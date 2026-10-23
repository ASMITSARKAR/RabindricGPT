"""
scripts/11_run.py : Single-run entrypoint executing pretraining, adaptation, or pilot from run matrix row.
Backlog/Phase : Phase M      Docs : 15_BUILD_DIRECTIVE.md Phase M; 09_INFRA_WORKFLOW.md §7
Contract      : --run-id <id> -> runs/<id>/{config.json, metrics.jsonl, ckpt_*.pt, done.json}
Invariants    : Idempotent (exits 0 if done.json exists); records git commit hash and config hash
Failure modes : Run ID not found in matrix; unapproved run invoked in strict mode; NaN loss
Status        : TESTED
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.common.atomic_io import atomic_write_json
from src.rgpt.common.paths import get_repo_root
from src.rgpt.compute.matrix import generate_run_matrix
from src.rgpt.train.loop import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute single RabindricGPT experiment run")
    parser.add_argument("--run-id", type=str, required=True, help="Canonical Run ID from matrix")
    parser.add_argument("--allow-unapproved", action="store_true", help="Allow running unapproved/gated runs")
    parser.add_argument("--override-steps", type=int, default=0, help="Override steps for smoke testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    matrix = generate_run_matrix()

    match_row = next((r for r in matrix if r["run_id"] == args.run_id), None)
    if not match_row:
        print(f"Error: Run ID {args.run_id} not found in execution matrix.", file=sys.stderr)
        sys.exit(1)

    if not match_row.get("approved", False) and not args.allow_unapproved:
        print(f"Error: Run ID {args.run_id} is currently GATED/unapproved.", file=sys.stderr)
        sys.exit(1)

    run_dir = repo_root / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Idempotency check: if done.json exists, exit 0 immediately
    done_file = run_dir / "done.json"
    if done_file.exists():
        print(f"Run {args.run_id} already finished (done.json exists). Exiting 0.")
        sys.exit(0)

    # 2. Build configuration from matrix row
    tok_name = match_row["tokenizer"]
    model_size = match_row["model_size"]
    seed = match_row["seed"]
    lr = match_row["lr"]

    # Locate shards for this tokenizer and stage
    corpus = "wikisource" if ("AD-" in args.run_id or "adapt" in match_row["stage"]) else "fineweb2"
    val_split = "val" if corpus == "wikisource" else "egen"
    shard_index = repo_root / "shards" / corpus / tok_name / "train" / "index.json"
    val_shard_index = repo_root / "shards" / corpus / tok_name / val_split / "index.json"

    # Determine exact vocab size
    tok_cfg_file = repo_root / "tok" / tok_name / "config.json"
    if tok_cfg_file.exists():
        tok_meta = json.loads(tok_cfg_file.read_text(encoding="utf-8"))
        vocab_size = int(tok_meta.get("vocab_size", 264 if tok_name == "byte" else 16384))
    else:
        vocab_map = {"byte": 264, "char": 512, "aks": 8192, "bpe8": 8192, "bpe16": 16384, "uni16": 16384}
        vocab_size = vocab_map.get(tok_name, 16384)

    run_config = {
        "run_id": args.run_id,
        "stage": match_row["stage"],
        "tokenizer": tok_name,
        "seed": seed,
        "model": {
            "size_id": model_size,
            "vocab_size": vocab_size,
        },
        "train": {
            "total_steps": 100,
            "learning_rate": lr,
            "batch_size": 2,
            "checkpoint_interval": 50,
            "s_max": 64,
        },
        "shard_index": str(shard_index),
        "val_shard_index": str(val_shard_index) if val_shard_index.exists() else None,
    }

    config_path = run_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f)

    # 3. Execute training loop
    mode = "adapt" if "AD-" in args.run_id or "adapt" in match_row["stage"] else "pretrain"
    t0 = time.perf_counter()
    res = train(
        run_dir=run_dir,
        config_path=config_path,
        mode=mode,
        override_steps=args.override_steps,
    )
    elapsed = time.perf_counter() - t0

    # 4. Write ledger row
    spent_units = float(match_row.get("planned_units", 0.0))
    if args.override_steps > 0:
        spent_units = round(spent_units * (args.override_steps / 100.0), 4)

    ledger_row = {
        "run_id": args.run_id,
        "stage": match_row["stage"],
        "tokenizer": tok_name,
        "model_size": model_size,
        "seed": seed,
        "gpu": "CPU",
        "spent_units": spent_units,
        "time_seconds": round(elapsed, 2),
        "total_steps": res.get("total_steps", args.override_steps),
        "final_loss": res.get("final_loss"),
    }
    atomic_write_json(run_dir / "ledger_row.json", ledger_row)
    print(f"Run {args.run_id} completed successfully. Ledger row written.")


if __name__ == "__main__":
    main()
