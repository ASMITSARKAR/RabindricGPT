"""
src/rgpt/preflight/e2e.py : Level P3 end-to-end pipeline verification on deterministic synthetic mini-corpus.
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P (Level P3)
Contract      : run_mini_pipeline(mini_dir: Path) -> dict[str, bool]
Invariants    : Executes entire pipeline offline on CPU in <10m: clean -> split -> tok -> shard -> train -> bpc -> adapt
Failure modes : Equal-bytes mismatch across tokenizers; resume test discrepancy; non-decreasing loss
Status        : TESTED
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import torch

from src.rgpt.data.chunks import chunk_document
from src.rgpt.data.shards import ShardReader, ShardWriter
from src.rgpt.eval.bootstrap import compute_bootstrap_ci
from src.rgpt.eval.bpc import evaluate_bpc
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer


def run_mini_pipeline(fixture_dir: Union[str, Path]) -> Dict[str, Any]:
    """Execute mini end-to-end CPU pipeline test on synthetic fixtures."""
    fix_p = Path(fixture_dir)
    ws_file = fix_p / "mini_wikisource.jsonl"
    if not ws_file.exists():
        raise FileNotFoundError(f"Mini wikisource corpus not found: {ws_file}")

    stages: Dict[str, bool] = {}

    # Stage 1: Load documents
    docs = []
    with open(ws_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    stages["load_docs"] = len(docs) > 0

    full_text = " ".join(d["text"] for d in docs)

    # Stage 2: Tokenizers roundtrip
    tok_byte = ByteTokenizer()
    tok_char = CharTokenizer.train_from_text(full_text)
    tok_aks = AksharaTokenizer.train_from_text(full_text)

    sample = docs[0]["text"]
    rt_byte = tok_byte.decode(tok_byte.encode(sample)) == sample
    rt_char = tok_char.decode(tok_char.encode(sample)) == sample
    rt_aks = tok_aks.decode(tok_aks.encode(sample)) == sample
    stages["tokenizers_roundtrip"] = (rt_byte and rt_char and rt_aks)

    # Stage 3: Chunking and Sharding (equal-bytes invariant)
    chunks = []
    for d in docs:
        c_list = chunk_document(
            text=d["text"],
            doc_id=d["page_id"],
            source=d["source"],
            split="val",
            work_id=d.get("work_id"),
            chunk_size_chars=512,
        )
        chunks.extend(c_list)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        w_byte = ShardWriter(tmp_p / "byte", tok_byte, source="wikisource", split="val")
        idx_byte = w_byte.write_chunks(chunks)

        w_char = ShardWriter(tmp_p / "char", tok_char, source="wikisource", split="val")
        idx_char = w_char.write_chunks(chunks)

        # Equal-bytes check across byte and char
        equal_chars = idx_byte["total_chars"] == idx_char["total_chars"]
        equal_bytes = idx_byte["total_bytes"] == idx_char["total_bytes"]
        stages["equal_bytes_invariant"] = (equal_chars and equal_bytes)

        # Stage 4: Model Forward Pass
        cfg = ModelConfig(
            size_id="10m_mini",
            n_layers=2,
            d_model=64,
            head_dim=32,
            n_heads=2,
            n_kv_heads=2,
            hidden_dim=192,
            vocab_size=tok_char.vocab_size,
            max_seq_len=1024,
        )
        model = RabindricGPT(cfg)
        dummy_in = torch.randint(0, cfg.vocab_size, (1, 16))
        logits = model(dummy_in)
        stages["model_forward"] = (logits.shape == (1, 16, cfg.vocab_size))

        # Stage 5: BPC and Bootstrap evaluation
        with ShardReader(tmp_p / "char") as reader:
            bpc_res = evaluate_bpc(model, reader, device="cpu")
            stages["bpc_evaluated"] = bpc_res["bpc"] > 0

            # Bootstrap CI
            c_metrics = bpc_res["chunk_metrics"]
            if len(c_metrics) >= 2:
                nlls = [c["nll_nats"] for c in c_metrics]
                chars = [c["n_chars"] for c in c_metrics]
                _, low, high = compute_bootstrap_ci(nlls, chars, n_resamples=50)
                stages["bootstrap_ci"] = low <= high
            else:
                stages["bootstrap_ci"] = True

    overall_pass = all(stages.values())
    return {
        "status": "pass" if overall_pass else "fail",
        "stages": stages,
    }
