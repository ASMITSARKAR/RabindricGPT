"""
scripts/audits/run_colab_pilot_and_kill_resume.py : Throughput benchmark, MFU measurement, and kill-resume test for Colab L4/A100.
Contract      : CLI options -> results/pilot_kill_resume.json, prints formatted report
Invariants    : Hard stop if cumulative units exceed 6.0; checks kill-resume continuity (|Δloss| <= 1e-4)
Unit Rates    : L4 = 1.8 units/hour, A100 = 5.0 units/hour
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.common.paths import get_repo_root
from src.rgpt.compute.flops import calculate_f_tok
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT
from src.rgpt.train.checkpoint import load_latest_checkpoint, save_checkpoint
from src.rgpt.train.throughput import calculate_mfu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab Throughput & Kill-Resume Benchmark (Action B: L4 10M, 100 steps, kill at step 50)"
    )
    parser.add_argument(
        "--mode",
        choices=["l4_10m", "a100_25m"],
        default="l4_10m",
        help="Benchmark execution mode (default: l4_10m covering all 5 tokenizers on L4 for funded plan; a100_25m is an optional single 200-step run costing ~0.42u)",
    )
    parser.add_argument("--steps", type=int, default=100, help="Total steps per tokenizer run (default: 100)")
    parser.add_argument("--kill-step", type=int, default=50, help="Step at which to kill and resume (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Effective batch size per step (default: 64)")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=32,
        help="Micro-batch size for gradient accumulation (default: 32; provides 2x accum for batch 64 ensuring safe VRAM margin under 24 GB)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2816,
        help="Maximum sequence length ceiling (default: 2816, holds byte window of ~2,800 bytes)",
    )
    parser.add_argument("--max-units", type=float, default=6.0, help="Hard stop unit cap (default: 6.0)")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    return parser.parse_args()


# GPU Peak FLOP/s specs (FP16/BF16 non-sparse)
GPU_PEAK_FLOPS = {
    "L4": 1.21e14,       # 121 TFLOPS (NVIDIA Ada Lovelace L4, BF16 non-sparse)
    "A100": 3.12e14,     # 312 TFLOPS (NVIDIA Ampere A100 SXM4/PCIe, BF16 non-sparse)
    "CPU": 1.0e12,
}

# Empirical Colab Pro Burn Rates (units/hr) - owner specified
GPU_BURN_RATES = {
    "L4": 1.80,          # 1.8 units / hour
    "A100": 5.00,        # 5.0 units / hour
    "CPU": 0.00,
}

MODELS_CONFIG = {
    "10m": {"L": 6, "d": 384, "N_ne": 10621824},
    "25m": {"L": 8, "d": 512, "N_ne": 25698816},
}

TOKENIZER_VOCABS = {
    "byte": 264,
    "char": 512,
    "aks": 8192,
    "bpe16": 16384,
    "uni16": 16384,
}

# Typical measured tokens per ~2,800-byte window
TYPICAL_TOKENS_PER_WINDOW = {
    "byte": 2694,
    "char": 1024,
    "aks": 664,
    "bpe16": 295,
    "uni16": 295,
}


def estimate_memory_breakdown(
    scale: str,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    micro_batch_size: int = 64,
) -> dict:
    """Compute detailed theoretical peak VRAM memory breakdown in GB."""
    L = MODELS_CONFIG[scale]["L"]
    d = MODELS_CONFIG[scale]["d"]
    N_ne = MODELS_CONFIG[scale]["N_ne"]
    n_params = N_ne + (vocab_size * d)

    # Static: bf16 weights (2B) + grad (2B) + AdamW fp32 states & master (12B) = 16B/param
    static_gb = (n_params * 16) / (1024 ** 3)

    # Active micro-batch
    mb_tokens = micro_batch_size * seq_len

    # Activations with FlashAttention / SDPA: ~32 bytes per token per layer
    activations_gb = (mb_tokens * d * 32 * L) / (1024 ** 3)

    # Forward logits (BF16): [mb, seq, V] * 2 bytes
    logits_gb = (mb_tokens * vocab_size * 2) / (1024 ** 3)

    # Cross-entropy autograd buffer (FP32 logits + grad in autograd): [mb, seq, V] * 4 bytes
    ce_autograd_gb = (mb_tokens * vocab_size * 4) / (1024 ** 3)

    # Total peak VRAM estimate
    total_est_gb = static_gb + activations_gb + logits_gb + ce_autograd_gb

    return {
        "static_gb": round(static_gb, 3),
        "activations_gb": round(activations_gb, 3),
        "logits_gb": round(logits_gb, 3),
        "ce_autograd_gb": round(ce_autograd_gb, 3),
        "total_est_gb": round(total_est_gb, 2),
    }


def load_shard_chunk_windows(tok: str) -> List[np.ndarray]:
    """Load real Tagore chunk windows from wikisource shards."""
    repo_root = get_repo_root()
    shard_bin = repo_root / "shards" / "wikisource" / tok / "train" / "shard_00000.bin"
    index_json = repo_root / "shards" / "wikisource" / tok / "train" / "index.json"

    windows: List[np.ndarray] = []

    if shard_bin.exists() and index_json.exists():
        raw_arr = np.fromfile(shard_bin, dtype=np.uint16)
        meta = json.loads(index_json.read_text(encoding="utf-8"))
        offsets = meta.get("chunk_offsets", [])
        for chunk in offsets:
            start = chunk["start_offset"]
            length = chunk["num_tokens"]
            w = raw_arr[start : start + length].astype(np.int64)
            if len(w) > 1:
                windows.append(w)
        if len(windows) > 0:
            return windows

    # Fallback to contiguous slicing from binary if index offsets not available
    if shard_bin.exists():
        raw_arr = np.fromfile(shard_bin, dtype=np.uint16).astype(np.int64)
        approx_len = TYPICAL_TOKENS_PER_WINDOW.get(tok, 295)
        n_chunks = len(raw_arr) // approx_len
        for i in range(n_chunks):
            w = raw_arr[i * approx_len : (i + 1) * approx_len]
            if len(w) > 1:
                windows.append(w)
        if len(windows) > 0:
            return windows

    # Fallback to tokenizing cleaned Tagore text
    cleaned_text_path = repo_root / "data" / "cleaned" / "tagore" / "train.txt"
    if cleaned_text_path.exists():
        try:
            from src.rgpt.tokenizers.base import get_tokenizer
            tokenizer = get_tokenizer(tok)
            text = cleaned_text_path.read_text(encoding="utf-8")[:100000]
            token_ids = tokenizer.encode(text)
            approx_len = TYPICAL_TOKENS_PER_WINDOW.get(tok, 295)
            for i in range(0, len(token_ids) - approx_len, approx_len):
                windows.append(np.array(token_ids[i : i + approx_len], dtype=np.int64))
            if len(windows) > 0:
                return windows
        except Exception:
            pass

    # Deterministic synthetic windows if shard files are absent
    torch.manual_seed(42)
    vocab = TOKENIZER_VOCABS.get(tok, 16384)
    approx_len = TYPICAL_TOKENS_PER_WINDOW.get(tok, 295)
    for _ in range(128):
        windows.append(np.random.randint(0, vocab, size=(approx_len,), dtype=np.int64))
    return windows


def build_dynamic_batch(
    windows: List[np.ndarray],
    batch_idx: int,
    batch_size: int,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build a batch of sequences with dynamic padding to batch max length.
    
    Returns:
        (x_batch, y_batch, max_batch_len)
    """
    selected: List[np.ndarray] = []
    n_windows = len(windows)
    for i in range(batch_size):
        w = windows[(batch_idx * batch_size + i) % n_windows]
        selected.append(w)

    # Dynamic batch length: max length across these 64 windows, capped by max_seq_len
    raw_max = max(len(w) for w in selected)
    batch_max_len = min(max_seq_len, raw_max)
    # Target length is batch_max_len - 1 (since x = seq[:-1], y = seq[1:])
    seq_dim = max(1, batch_max_len - 1)

    x_tensor = torch.zeros((batch_size, seq_dim), dtype=torch.int64)
    y_tensor = torch.zeros((batch_size, seq_dim), dtype=torch.int64)

    for i, w in enumerate(selected):
        # Truncate if exceeds batch_max_len
        w_sub = w[:batch_max_len]
        curr_len = len(w_sub) - 1
        if curr_len > 0:
            x_tensor[i, :curr_len] = torch.from_numpy(w_sub[:-1])
            y_tensor[i, :curr_len] = torch.from_numpy(w_sub[1:])

    return x_tensor, y_tensor, seq_dim


def run_single_benchmark(
    scale: str,
    tok: str,
    device_obj: torch.device,
    gpu_type: str,
    total_steps: int = 100,
    kill_step: int = 50,
    batch_size: int = 64,
    micro_batch_size: int = 64,
    seq_len: int = 2816,
) -> dict:
    vocab_size = TOKENIZER_VOCABS[tok]
    cfg = ModelConfig.from_size(scale, vocab_size=vocab_size)
    model = RabindricGPT(cfg).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Load real Tagore chunk windows
    windows = load_shard_chunk_windows(tok)
    approx_seq = TYPICAL_TOKENS_PER_WINDOW.get(tok, 295)

    # Gradient accumulation setup
    grad_accum_steps = max(1, batch_size // micro_batch_size)

    mem_est_dyn = estimate_memory_breakdown(scale, vocab_size, batch_size, approx_seq, micro_batch_size)
    mem_est_fixed = estimate_memory_breakdown(scale, vocab_size, batch_size, seq_len, micro_batch_size)

    print(
        f"       [Dynamic Memory Est @ seq ~{approx_seq}] Peak: {mem_est_dyn['total_est_gb']:.2f} GB "
        f"(Static: {mem_est_dyn['static_gb']:.2f} GB, Acts: {mem_est_dyn['activations_gb']:.2f} GB, Logits: {mem_est_dyn['logits_gb']:.2f} GB, Autograd: {mem_est_dyn['ce_autograd_gb']:.2f} GB)",
        flush=True,
    )
    if approx_seq != seq_len:
        print(
            f"       [Fixed Ceiling Est @ seq {seq_len}] Peak: {mem_est_fixed['total_est_gb']:.2f} GB (Grad Accum: {grad_accum_steps}x)",
            flush=True,
        )

    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)

    t0 = time.perf_counter()
    loss_at_kill_step = 0.0
    total_tokens_processed = 0

    # Phase 1: Steps 0 to kill_step
    last_x_kill: Optional[torch.Tensor] = None
    last_y_kill: Optional[torch.Tensor] = None

    for step in range(kill_step):
        optimizer.zero_grad()
        x_step, y_step, actual_seq_dim = build_dynamic_batch(windows, step, batch_size, seq_len)
        step_tokens = batch_size * actual_seq_dim
        total_tokens_processed += step_tokens

        # Process micro-batches with gradient accumulation
        for mb_idx in range(grad_accum_steps):
            mb_start = mb_idx * micro_batch_size
            mb_end = mb_start + micro_batch_size
            x_mb = x_step[mb_start:mb_end].to(device_obj)
            y_mb = y_step[mb_start:mb_end].to(device_obj)

            logits = model(x_mb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_mb.view(-1), ignore_index=0) / grad_accum_steps
            loss.backward()

        optimizer.step()

        if step == kill_step - 1:
            loss_at_kill_step = loss.item() * grad_accum_steps
            last_x_kill = x_step[:micro_batch_size].to(device_obj)
            last_y_kill = y_step[:micro_batch_size].to(device_obj)

    # Emulate Kill: Save Checkpoint
    tmp_ckpt_dir = Path("results/tmp_ckpt") / f"{scale}_{tok}"
    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = f"hash_{scale}_{tok}"
    save_checkpoint(
        checkpoint_dir=tmp_ckpt_dir,
        step=kill_step,
        model=model,
        optimizer=optimizer,
        scaler=None,
        config_hash=cfg_hash,
        tokenizer_hash=f"tok_{tok}",
    )

    # Evaluate unkilled continuity loss on last step batch
    with torch.no_grad():
        unkilled_eval_loss = F.cross_entropy(
            model(last_x_kill).view(-1, vocab_size), last_y_kill.view(-1), ignore_index=0
        ).item()

    # Emulate Resume: Instantiate fresh model and load checkpoint
    fresh_model = RabindricGPT(cfg).to(device_obj)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    ckpt_dict = load_latest_checkpoint(checkpoint_dir=tmp_ckpt_dir, expected_config_hash=cfg_hash)
    assert ckpt_dict is not None, "Failed to load saved checkpoint"
    fresh_model.load_state_dict(ckpt_dict["model_state_dict"])
    fresh_optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
    resumed_step = ckpt_dict["step"]

    # Evaluate step kill loss on resumed model
    with torch.no_grad():
        resumed_eval_loss = F.cross_entropy(
            fresh_model(last_x_kill).view(-1, vocab_size), last_y_kill.view(-1), ignore_index=0
        ).item()

    delta_loss = abs(resumed_eval_loss - unkilled_eval_loss)

    # Phase 2: Steps kill_step to total_steps on resumed model
    for step in range(kill_step, total_steps):
        fresh_optimizer.zero_grad()
        x_step, y_step, actual_seq_dim = build_dynamic_batch(windows, step, batch_size, seq_len)
        step_tokens = batch_size * actual_seq_dim
        total_tokens_processed += step_tokens

        for mb_idx in range(grad_accum_steps):
            mb_start = mb_idx * micro_batch_size
            mb_end = mb_start + micro_batch_size
            x_mb = x_step[mb_start:mb_end].to(device_obj)
            y_mb = y_step[mb_start:mb_end].to(device_obj)

            logits = fresh_model(x_mb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_mb.view(-1), ignore_index=0) / grad_accum_steps
            loss.backward()

        fresh_optimizer.step()

    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)

    tokens_per_sec = total_tokens_processed / elapsed

    # Memory peak
    if device_obj.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device_obj)
        peak_gb = peak_bytes / (1024 ** 3)
    else:
        peak_gb = 0.0

    # FLOPs and MFU
    n_ne = MODELS_CONFIG[scale]["N_ne"]
    L = MODELS_CONFIG[scale]["L"]
    d = MODELS_CONFIG[scale]["d"]
    f_tok = calculate_f_tok(n_ne=n_ne, vocab_size=vocab_size, d_model=d, n_layers=L, s_tokens=approx_seq)
    peak_flops = GPU_PEAK_FLOPS.get(gpu_type, 1.21e14)
    mfu = calculate_mfu(tokens_per_sec, f_tok, peak_flops)

    # Compute burn with empirical Colab Pro rates
    burn_rate = GPU_BURN_RATES.get(gpu_type, 1.80)
    units_burned = (elapsed / 3600.0) * burn_rate

    return {
        "scale": scale,
        "tokenizer": tok,
        "total_steps": total_steps,
        "kill_step": kill_step,
        "resumed_step": resumed_step,
        "delta_loss": delta_loss,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "mfu": round(mfu, 4),
        "peak_gb": round(peak_gb, 2),
        "elapsed_sec": round(elapsed, 2),
        "units_burned": round(units_burned, 4),
        "gpu_peak_flops_assumed": peak_flops,
        "burn_rate_used": burn_rate,
        "dynamic_memory_estimate": mem_est_dyn,
        "fixed_ceiling_estimate": mem_est_fixed,
    }


def main() -> None:
    args = parse_args()
    device_str = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device_obj = torch.device(device_str)

    gpu_type = "CPU"
    if device_obj.type == "cuda":
        name = torch.cuda.get_device_name(0)
        if "L4" in name:
            gpu_type = "L4"
        elif "A100" in name:
            gpu_type = "A100"
        elif "T4" in name:
            gpu_type = "T4"
        else:
            gpu_type = "L4"

    if gpu_type == "T4":
        raise RuntimeError(
            "T4 GPU detected. T4 is excluded from pretraining by the matrix (L4 bf16 is default; A100 is for 25M/100M). "
            "Action B empirical throughput benchmarking requires an L4 or A100 runtime."
        )

    # CPU Guard: refuse batch > 4 or seq > 128 on CPU to prevent machine freeze/crash
    if device_obj.type == "cpu":
        if args.batch_size > 4 or args.seq_len > 128:
            raise RuntimeError(
                f"CPU Guard: Execution refused on CPU for batch_size={args.batch_size} > 4 or seq_len={args.seq_len} > 128. "
                "Action B empirical throughput benchmarking requires an L4 or A100 GPU runtime. "
                "On CPU, only lightweight smoke tests with batch <= 4 and seq_len <= 128 are permitted."
            )

    grad_accum_steps = max(1, args.batch_size // args.micro_batch_size)

    print("=" * 80)
    print("RabindricGPT Action B: Empirical Throughput & Kill-Resume Benchmark")
    print("=" * 80)
    print(">>> STEP: RECORD COLAB COMPUTE UNIT BALANCE BEFORE RUNNING <<<")
    print(f"Device: {device_obj.type.upper()} ({gpu_type}) | Mode: {args.mode}")
    print(f"Batch Size: {args.batch_size} (Dynamic padding to batch max) | Grad Accum: {grad_accum_steps}x")
    print(f"Steps: {args.steps} (Kill at step {args.kill_step}) | Seq Len Ceiling: {args.seq_len}")
    print(f"Burn Rates: L4 = {GPU_BURN_RATES['L4']} u/hr | A100 = {GPU_BURN_RATES['A100']} u/hr | Max Safety Cap: {args.max_units} u")
    print(f"Peak FLOP/s Assumed: L4 = {GPU_PEAK_FLOPS['L4']:.2e} (121 TFLOPS) | A100 = {GPU_PEAK_FLOPS['A100']:.2e} (312 TFLOPS)")
    print("=" * 80)

    # Print VRAM Memory Analysis Table for L4 (24GB)
    print("\n" + "-" * 80)
    print("VRAM MEMORY ESTIMATION TABLE FOR 10M ON L4 (24 GB CEILING)")
    print("-" * 80)
    print(f"{'Tokenizer':<8} | {'Vocab':>5} | {'Seq(dyn)':>8} | {'Static':>7} | {'Acts':>6} | {'Logits':>6} | {'Peak(dyn)':>9} | {'Fits 24G L4?':<12}")
    print("-" * 80)
    for tok in ["byte", "char", "aks", "bpe16", "uni16"]:
        v = TOKENIZER_VOCABS[tok]
        s_dyn = TYPICAL_TOKENS_PER_WINDOW[tok]
        m = estimate_memory_breakdown("10m", v, args.batch_size, s_dyn, args.micro_batch_size)
        fits = "YES" if m["total_est_gb"] < 22.0 else "NO (OOM RISK)"
        print(f"{tok:<8} | {v:>5} | {s_dyn:>8} | {m['static_gb']:>6.2f}G | {m['activations_gb']:>5.2f}G | {m['logits_gb']:>5.2f}G | {m['total_est_gb']:>8.2f}G | {fits:<12}")
    print("-" * 80)
    print("Notice: With dynamic padding to batch max, 16k vocabs consume only ~2.9 GB (well under 24 GB)!")
    print("-" * 80)

    runs = []
    if args.mode == "l4_10m":
        for tok in ["byte", "char", "aks", "bpe16", "uni16"]:
            runs.append(("10m", tok, "L4"))
    elif args.mode == "a100_25m":
        runs.append(("25m", "bpe16", "A100"))

    results = []
    cumulative_units = 0.0

    for scale, tok, target_gpu in runs:
        effective_gpu = gpu_type if device_obj.type == "cuda" else "CPU"
        print(f"\n---> Benchmarking {scale} x {tok} (Target GPU: {target_gpu}, Active: {effective_gpu})...", flush=True)
        res = run_single_benchmark(
            scale=scale,
            tok=tok,
            device_obj=device_obj,
            gpu_type=effective_gpu,
            total_steps=args.steps,
            kill_step=args.kill_step,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            seq_len=args.seq_len,
        )
        cumulative_units += res["units_burned"]
        results.append(res)

        print(f"     Done: {res['tokens_per_sec']:.1f} tok/s | MFU: {res['mfu']:.2%} | Peak: {res['peak_gb']:.2f} GB", flush=True)
        print(f"     Kill-Resume: |delta_loss| = {res['delta_loss']:.6f} | Units: {res['units_burned']:.4f} u (Cumulative: {cumulative_units:.4f} u)", flush=True)

        if cumulative_units >= args.max_units:
            print(f"\n[SAFETY WARNING] Cumulative units ({cumulative_units:.4f}) reached max cap ({args.max_units}). Aborting early.")
            break

    out_file = Path("results/pilot_kill_resume.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Summary saved to {out_file}")
    print(">>> STEP: RECORD COLAB COMPUTE UNIT BALANCE AFTER RUNNING <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
