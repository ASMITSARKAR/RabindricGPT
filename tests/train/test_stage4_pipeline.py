"""
tests/train/test_stage4_pipeline.py : Unit tests for Stage 4 model and training pipeline verification.
"""

import pytest
from pathlib import Path
from scripts.audits.audit_stage4_pipeline import audit_stage4_pipeline


def test_stage4_pipeline_verification_passes():
    if not Path("shards/wikisource/bpe16/train/index.json").exists():
        pytest.skip("Shards not present in clean clone; run 07_build_shards.py or mount Drive shards first.")
    res = audit_stage4_pipeline()
    assert res["loss_reduction_pct"] >= 20.0
    assert res["hierarchical_mean_bpc"] > 0
    assert res["bootstrap_resamples"] == 1000
