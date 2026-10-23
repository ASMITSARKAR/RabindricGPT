"""
scripts/09_pilot.py : Run throughput and MFU pilot on active GPU and write results/pilot.json.
Backlog/Phase : I10 / Phase L      Docs : 15_BUILD_DIRECTIVE.md Phase L; 08_COMPUTE_BUDGET.md §10
Contract      : CLI options -> results/pilot.json
Invariants    : Warmup and compile steps excluded from measurement; records units/hour read from Colab status bar
Failure modes : Measurement window too short; GPU out-of-memory
Status        : TESTED
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.rgpt.common.atomic_io import atomic_write_json
from src.rgpt.common.paths import get_repo_root
from src.rgpt.compute.flops import calculate_f_tok
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT
from src.rgpt.train.throughput import calculate_mfu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute throughput and MFU benchmark pilot")
    parser.add_argument("--units-per-hour", type=float, default=None, help="Measured burn rate from Colab status bar")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU identifier (T4, L4, A100)")
    parser.add_argument("--output", type=str, default="results/pilot.json", help="Output path")
    parser.add_argument("--steps", type=int, default=20, help="Benchmarking steps per model size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps to exclude from measurement")
    return parser.parse_args()


def benchmark_size(size_id: str, device: str, steps: int, warmup: int) -> dict:
    device_obj = torch.device(device)
    cfg = ModelConfig.from_size(size_id, vocab_size=264)
    model = RabindricGPT(cfg).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    b, s = 2, 64
    x = torch.randint(0, cfg.vocab_size, (b, s), device=device_obj)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    # Timed measurement
    t0 = time.perf_counter()
    tokens_total = 0

    for _ in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        tokens_total += b * s

    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)

    tok_per_sec = tokens_total / elapsed

    # Peak FLOPs: L4 is ~121 TFLOPs (1.21e14), T4 is ~65 TFLOPs
    peak_flops = 1.21e14
    f_tok = calculate_f_tok(
        n_ne=sum(p.numel() for p in model.parameters()) - cfg.vocab_size * cfg.d_model,
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        s_tokens=s,
    )
    mfu = calculate_mfu(tok_per_sec, f_tok, peak_flops)

    return {
        "size_id": size_id,
        "n_params": sum(p.numel() for p in model.parameters()),
        "tokens_per_sec": round(tok_per_sec, 2),
        "mfu": round(mfu, 4),
        "elapsed_seconds": round(elapsed, 4),
    }


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    out_file = Path(args.output)
    if not out_file.is_absolute():
        out_file = repo_root / out_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running 09_pilot.py on {device} (GPU arg: {args.gpu}, units/hr: {args.units_per_hour})")

    res_10m = benchmark_size("10m", device, args.steps, args.warmup)
    res_25m = benchmark_size("25m", device, args.steps, args.warmup)

    pilot_data = {
        "device": device,
        "gpu_type": args.gpu if args.gpu != "auto" else ("GPU" if device == "cuda" else "CPU"),
        "units_per_hour": args.units_per_hour,
        "results": {
            "10m": res_10m,
            "25m": res_25m,
        },
    }

    atomic_write_json(out_file, pilot_data)
    print(f"Pilot benchmark written to {out_file}")


if __name__ == "__main__":
    main()
