"""
src/rgpt/eval/bpc.py : Fair Bits-Per-Character (BPC) and Bits-Per-Byte (BPB) calculation over chunked eval sets.
Backlog/Phase : I9 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; DR-2; 01 §1
Contract      : evaluate_bpc(model, eval_chunks) -> {"bpc": float, "bpb": float, ...}
Invariants    : BOS unscored; PAD masked; identical n_chars across all tokenizers on the same eval split
Failure modes : Empty eval set; division by zero characters
Status        : IMPLEMENTED
"""

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rgpt.data.shards import ShardReader


def evaluate_bpc(
    model: nn.Module,
    shard_reader: ShardReader,
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute exact BPC and BPB on a held-out shard set.

    Formulas:
        BPC = sum_nll_nats / (n_chars * ln 2)
        BPB = sum_nll_nats / (n_bytes * ln 2)

    Args:
        model: Evaluated PyTorch model.
        shard_reader: ShardReader for the evaluation set.
        batch_size: Evaluation batch size.
        device: Computation device.

    Returns:
        Dictionary containing 'bpc', 'bpb', 'n_chars', 'n_bytes', and 'total_nll_nats'.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    total_nll_nats = 0.0
    total_chars = 0
    total_bytes = 0
    chunk_metrics: List[Dict[str, Any]] = []

    num_chunks = len(shard_reader)
    if num_chunks == 0:
        raise ValueError("ShardReader contains 0 chunks for evaluation.")

    with torch.no_grad():
        for i in range(num_chunks):
            chunk = shard_reader.get_chunk(i)
            tokens = chunk["tokens"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            else:
                tokens = list(tokens)

            n_chars = chunk["n_chars"]
            n_bytes = chunk["n_bytes"]

            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=dev)
            targets = torch.tensor([tokens[1:]], dtype=torch.long, device=dev)

            logits = model(input_ids)
            vocab_size = logits.shape[-1]

            # Cross entropy loss in nats (unreduced)
            loss_nats = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                reduction="sum",
            ).item()

            total_nll_nats += loss_nats
            total_chars += n_chars
            total_bytes += n_bytes

            chunk_metrics.append({
                "chunk_id": chunk["chunk_id"],
                "work_id": chunk.get("work_id"),
                "nll_nats": loss_nats,
                "n_chars": n_chars,
                "n_bytes": n_bytes,
                "bpc": loss_nats / (n_chars * math.log(2)) if n_chars > 0 else 0.0,
            })

    if total_chars == 0:
        raise ValueError("Total characters across chunks is zero.")

    ln2 = math.log(2.0)
    overall_bpc = total_nll_nats / (total_chars * ln2)
    overall_bpb = total_nll_nats / (total_bytes * ln2)

    return {
        "bpc": overall_bpc,
        "bpb": overall_bpb,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_nll_nats": total_nll_nats,
        "chunk_metrics": chunk_metrics,
    }
