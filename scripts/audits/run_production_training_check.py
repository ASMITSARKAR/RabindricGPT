"""
scripts/audits/run_production_training_check.py : Real-shard training verification with batch >= 16,384 tokens and production schedule.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.data.shards import ShardReader
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT
from src.rgpt.train.schedule import get_learning_rate


def evaluate_val_loss(model: torch.nn.Module, val_reader: ShardReader, max_chunks: int = 16, s_tokens: int = 512, vocab_size: int = 16384) -> float:
    model.eval()
    total_loss = 0.0
    chunks_eval = 0
    with torch.no_grad():
        for i in range(min(max_chunks, len(val_reader))):
            tokens = list(val_reader.get_chunk(i)["tokens"])
            if len(tokens) > s_tokens:
                tokens = tokens[:s_tokens]
            elif len(tokens) < s_tokens:
                tokens = tokens + [0] * (s_tokens - len(tokens))
            x = torch.tensor([tokens[:-1]], dtype=torch.long)
            y = torch.tensor([tokens[1:]], dtype=torch.long)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=0)
            total_loss += loss.item()
            chunks_eval += 1
    model.train()
    return total_loss / max(1, chunks_eval)


def main() -> None:
    train_shard_path = Path("shards/wikisource/bpe16/train")
    val_shard_path = Path("shards/wikisource/bpe16/val")

    assert train_shard_path.exists(), "Missing train shard!"
    assert val_shard_path.exists(), "Missing val shard!"

    train_meta = json.loads((train_shard_path / "index.json").read_text(encoding="utf-8"))
    val_meta = json.loads((val_shard_path / "index.json").read_text(encoding="utf-8"))

    total_train_tokens = train_meta["total_tokens"]
    total_chunks = train_meta["total_chunks"]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu")
    args = parser.parse_args()

    # Production parameters
    total_steps = args.steps
    batch_size_chunks = 32
    seq_len = 512
    tokens_per_step = batch_size_chunks * seq_len  # 16,384 tokens per step!
    warmup_steps = max(1, int(0.02 * total_steps))  # 2% warmup
    peak_lr = 1.0e-3
    min_lr = 1.0e-4
    vocab_size = 16384

    # Calculate exact epoch count
    total_tokens_trained = total_steps * tokens_per_step
    epoch_count = total_tokens_trained / total_train_tokens

    print("=" * 70)
    print("RabindricGPT Production Training Verification on Real Shards")
    print(f"Model: 10M BPE16 (vocab_size={vocab_size})")
    print(f"Dataset Size: {total_train_tokens:,} tokens ({total_chunks} chunks)")
    print(f"Batch Configuration: {batch_size_chunks} chunks x {seq_len} tokens = {tokens_per_step:,} tokens/step")
    print(f"Total Steps: {total_steps} | Total Tokens Trained: {total_tokens_trained:,}")
    print(f"Epoch Count Over Dataset: {epoch_count:.2f} epochs")
    print(f"Schedule: 2% warmup ({warmup_steps} steps), Peak LR: {peak_lr}, Min LR: {min_lr}")
    print("=" * 70)

    cfg = ModelConfig.from_size("10m", vocab_size=vocab_size)
    model = RabindricGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    train_reader = ShardReader(train_shard_path)
    val_reader = ShardReader(val_shard_path)

    # Initial validation loss
    initial_val_loss = evaluate_val_loss(model, val_reader, max_chunks=16, s_tokens=seq_len, vocab_size=vocab_size)
    print(f"Step      0 | Train Loss:   N/A  | Val Loss: {initial_val_loss:.4f} | LR: {0.0:.6f} | Tokens/step: {tokens_per_step}")

    chunk_cursor = 0
    log_records = []
    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        # Cosine learning rate schedule with warmup
        if step <= warmup_steps:
            lr = peak_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Prepare batch of 32 chunks x 512 tokens = 16,384 tokens
        batch_tokens = []
        for _ in range(batch_size_chunks):
            chunk = list(train_reader.get_chunk(chunk_cursor % total_chunks)["tokens"])
            chunk_cursor += 1
            if len(chunk) > seq_len:
                chunk = chunk[:seq_len]
            elif len(chunk) < seq_len:
                chunk = chunk + [0] * (seq_len - len(chunk))
            batch_tokens.append(chunk)

        batch_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        x = batch_tensor[:, :-1]
        y = batch_tensor[:, 1:]

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == total_steps:
            elapsed = max(1e-6, time.perf_counter() - t0)
            steps_in_window = step if step < 100 else (100 if step % 100 == 0 else step % 100)
            speed = (steps_in_window * tokens_per_step) / elapsed
            val_loss = evaluate_val_loss(model, val_reader, max_chunks=8, s_tokens=seq_len, vocab_size=vocab_size)
            print(f"Step {step:6d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f} | Tokens/step: {tokens_per_step} | Speed: {speed:.1f} tok/s", flush=True)
            log_records.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "lr": round(lr, 6),
                "tokens_per_step": tokens_per_step,
                "speed_tok_sec": round(speed, 1),
            })
            t0 = time.perf_counter()

    train_reader.close()
    val_reader.close()

    out_file = Path("results/audits/audit_production_training_check.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "dataset_tokens": total_train_tokens,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "epoch_count": round(epoch_count, 2),
        "logs": log_records,
    }, indent=2), encoding="utf-8")
    print(f"\nTraining log successfully saved to {out_file}")


if __name__ == "__main__":
    main()
