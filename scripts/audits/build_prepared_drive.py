"""
scripts/audits/build_prepared_drive.py : Build the five funded shard sets and tokenizers
into prepared/ for Google Drive sync, complete with SHA-256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def build_prepared(output_dir: str = "prepared") -> dict:
    prep_dir = Path(output_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)
    prep_shards = prep_dir / "shards"
    prep_tok = prep_dir / "tok"
    prep_shards.mkdir(parents=True, exist_ok=True)
    prep_tok.mkdir(parents=True, exist_ok=True)

    funded = ["byte", "char", "aks", "bpe16", "uni16"]

    # 1. Copy shards for the 5 funded tokenizers
    for corpus in ["fineweb2", "wikisource"]:
        for tok in funded:
            src = Path("shards") / corpus / tok
            dst = prep_shards / corpus / tok
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    # 2. Copy tokenizer configs and models
    for tok in funded:
        src = Path("tok") / tok
        dst = prep_tok / tok
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # 3. Create zip archives for rapid Colab unpacking
    shards_zip = prep_dir / "shards.zip"
    with zipfile.ZipFile(shards_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in prep_shards.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(prep_dir))

    tok_zip = prep_dir / "tok.zip"
    with zipfile.ZipFile(tok_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in prep_tok.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(prep_dir))

    # 4. Compute SHA-256 hashes for all prepared files
    manifest = {}
    for p in sorted(prep_dir.rglob("*")):
        if p.is_file() and p.name != "hashes.json":
            content = p.read_bytes()
            h = hashlib.sha256(content).hexdigest()
            rel = str(p.relative_to(prep_dir)).replace("\\", "/")
            manifest[rel] = {
                "sha256": h,
                "bytes": len(content),
            }

    # Save manifest in prepared/ and results/
    (prep_dir / "hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    res_dir = Path("results")
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "prepared_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


if __name__ == "__main__":
    m = build_prepared()
    print("=" * 80)
    print("PREPARED DRIVE ARTIFACTS BUILD COMPLETE")
    print("=" * 80)
    print(f"Total entries hashed: {len(m)}")
    print(f"Shards zip size     : {m.get('shards.zip', {}).get('bytes', 0) / (1024**2):.2f} MB")
    print(f"Tokenizer zip size  : {m.get('tok.zip', {}).get('bytes', 0) / (1024**2):.2f} MB")
    print("Sample Hashes:")
    for k in sorted(m.keys())[:15]:
        v = m[k]
        print(f"  {k:<45} | {v['bytes']:>8} B | {v['sha256']}")
    print("=" * 80)
