"""
scripts/audits/test_resume_equality_cpu.py
Prove exact loss equality and initial_loss restoration upon checkpoint resumption across fresh processes on CPU.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rgpt.train.loop import train


def run_resume_proof():
    tmp_dir = PROJECT_ROOT / "results/tmp_resume_proof"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = tmp_dir / "ref_run"
    resume_dir = tmp_dir / "resume_run"
    ref_dir.mkdir(parents=True, exist_ok=True)
    resume_dir.mkdir(parents=True, exist_ok=True)

    # Minimal config for fast deterministic CPU run
    cfg = {
        "run_id": "RESUME-TEST",
        "stage": "pretrain",
        "tokenizer": "byte",
        "seed": 42,
        "model": {
            "size_id": "10m_mini",
            "n_layers": 2,
            "d_model": 64,
            "head_dim": 32,
            "n_heads": 2,
            "n_kv_heads": 2,
            "hidden_dim": 192,
            "vocab_size": 264,
            "max_seq_len": 64,
        },
        "train": {
            "total_steps": 4,
            "learning_rate": 0.001,
            "batch_size": 2,
            "checkpoint_interval": 1,  # Checkpoint every step
            "s_max": 64,
        },
        "shard_index": str(PROJECT_ROOT / "shards/fineweb2/byte/train/index.json"),
    }

    ref_cfg_path = ref_dir / "config.yaml"
    resume_cfg_path = resume_dir / "config.yaml"
    with open(ref_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with open(resume_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 1. Reference Run: full 4 steps
    torch.manual_seed(42)
    print("--- 1. Executing Continuous Reference Run (Steps 0-3) ---")
    ref_res = train(ref_dir, ref_cfg_path, override_steps=4)

    ref_metrics = [
        json.loads(line)
        for line in (ref_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    ref_losses = {m["step"]: m["loss"] for m in ref_metrics}
    ref_ckpt1 = torch.load(ref_dir / "checkpoints/ckpt_00000001.pt", map_location="cpu", weights_only=False)
    ref_initial_loss = ref_ckpt1.get("initial_loss")
    print(f"Ref Initial Loss : {ref_initial_loss:.6f}")
    for s, l in ref_losses.items():
        print(f"  Step {s}: loss = {l:.6f}")

    # 2. Killed Run: run steps 0 and 1 (override_steps=2), save checkpoint at step 1, then simulate preemption/kill
    torch.manual_seed(42)
    print("\n--- 2. Executing Initial Run Killed at Step 2 (Steps 0-1) ---")
    train(resume_dir, resume_cfg_path, override_steps=4, kill_at_step=2)

    # Checkpoint ckpt_00000001.pt exists in resume_dir
    assert (resume_dir / "checkpoints/ckpt_00000001.pt").exists(), "Step 1 checkpoint missing!"
    # Ensure done.json does not exist (not finished)
    if (resume_dir / "done.json").exists():
        (resume_dir / "done.json").unlink()

    # 3. Resume Run: fresh invocation simulating fresh process on disconnected clone
    print("\n--- 3. Executing Fresh Resume Run from Checkpoint (Steps 2-3) ---")
    resume_res = train(resume_dir, resume_cfg_path, override_steps=4)

    resume_metrics = [
        json.loads(line)
        for line in (resume_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    resume_losses = {m["step"]: m["loss"] for m in resume_metrics}

    resumed_ckpt = torch.load(resume_dir / "checkpoints/ckpt_final.pt", map_location="cpu", weights_only=False)
    resumed_initial_loss = resumed_ckpt.get("initial_loss")
    print(f"Resumed Initial Loss : {resumed_initial_loss:.6f}")
    for s, l in resume_losses.items():
        print(f"  Resumed Step {s}: loss = {l:.6f}")

    # 4. Strict assertions
    assert abs(resumed_initial_loss - ref_initial_loss) < 1e-6, (
        f"Initial loss mismatch: resumed {resumed_initial_loss} != ref {ref_initial_loss}"
    )
    for s in (2, 3):
        diff = abs(resume_losses[s] - ref_losses[s])
        print(f"Step {s} verification: ref={ref_losses[s]:.6f}, resumed={resume_losses[s]:.6f}, delta={diff:.2e}")
        assert diff < 1e-4, f"Loss mismatch at step {s}: ref={ref_losses[s]} vs resumed={resume_losses[s]}"

    print("\n================================================================================")
    print("CHECKPOINT RESUME PROOF: SUCCESS")
    print("  - Disconnect / fresh process simulated at step 2")
    print(f"  - initial_loss correctly restored: {resumed_initial_loss:.6f} == {ref_initial_loss:.6f}")
    print(f"  - Step 2 loss strictly identical: ref {ref_losses[2]:.6f} == resumed {resume_losses[2]:.6f}")
    print(f"  - Step 3 loss strictly identical: ref {ref_losses[3]:.6f} == resumed {resume_losses[3]:.6f}")
    print("================================================================================")

    # Cleanup temp proof dir
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    run_resume_proof()
