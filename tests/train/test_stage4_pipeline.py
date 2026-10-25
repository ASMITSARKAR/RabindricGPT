"""
tests/train/test_stage4_pipeline.py : Unit tests for Stage 4 model and training pipeline verification.
"""

from scripts.audits.audit_stage4_pipeline import audit_stage4_pipeline


def test_stage4_pipeline_verification_passes():
    res = audit_stage4_pipeline()
    assert res["loss_reduction_pct"] >= 20.0
    assert res["hierarchical_mean_bpc"] > 0
    assert res["bootstrap_resamples"] == 1000
