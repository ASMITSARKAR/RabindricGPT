"""
scripts/audits/freeze_data_snapshot.py : Generate immutable SHA-256 snapshot of all cleaned data and shards.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def hash_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file in 64KB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def generate_data_snapshot() -> Dict[str, Any]:
    git_sha = get_git_sha()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    cleaned_dir = PROJECT_ROOT / "data/cleaned"
    shards_dir = PROJECT_ROOT / "shards"

    cleaned_files: List[Dict[str, Any]] = []
    for p in sorted(cleaned_dir.glob("**/*")):
        if p.is_file():
            rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            size = p.stat().st_size
            sha = hash_file_sha256(p)
            cleaned_files.append({"path": rel, "size_bytes": size, "sha256": sha})

    shard_files: List[Dict[str, Any]] = []
    for p in sorted(shards_dir.glob("**/*")):
        if p.is_file():
            rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            size = p.stat().st_size
            sha = hash_file_sha256(p)
            shard_files.append({"path": rel, "size_bytes": size, "sha256": sha})

    snapshot = {
        "git_commit_sha": git_sha,
        "frozen_timestamp": timestamp,
        "num_cleaned_files": len(cleaned_files),
        "num_shard_files": len(shard_files),
        "data_cleaned": cleaned_files,
        "shards": shard_files,
    }
    return snapshot


def format_snapshot_text(snapshot: Dict[str, Any]) -> str:
    lines = [
        "=" * 80,
        "RABINDRICGPT FROZEN DATA & SHARD SNAPSHOT MANIFEST",
        "=" * 80,
        f"Git Commit SHA   : {snapshot['git_commit_sha']}",
        f"Frozen Timestamp : {snapshot['frozen_timestamp']}",
        f"Cleaned Files    : {snapshot['num_cleaned_files']}",
        f"Shard Files      : {snapshot['num_shard_files']}",
        "-" * 80,
        "\n1. Cleaned Data Files (data/cleaned/**):",
    ]
    for item in snapshot["data_cleaned"]:
        lines.append(f"  {item['sha256']}  {item['size_bytes']:>12,d} bytes  {item['path']}")

    lines.append("\n2. Shard Files & Indices (shards/**):")
    for item in snapshot["shards"]:
        lines.append(f"  {item['sha256']}  {item['size_bytes']:>12,d} bytes  {item['path']}")

    lines.append("-" * 80)
    lines.append("END OF MANIFEST")
    return "\n".join(lines)


def main() -> None:
    snapshot = generate_data_snapshot()
    json_path = PROJECT_ROOT / "results/data_snapshot.json"
    txt_path = PROJECT_ROOT / "results/data_snapshot.txt"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    txt_content = format_snapshot_text(snapshot)
    txt_path.write_text(txt_content, encoding="utf-8")

    print(f"Data snapshot frozen: {len(snapshot['data_cleaned'])} cleaned files, {len(snapshot['shards'])} shard files.")
    print(f"Saved to: {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
