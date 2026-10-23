"""
tests/preflight/test_e2e.py : Tests for preflight Level P3 mini end-to-end CPU pipeline test.
"""


from src.rgpt.common.paths import get_project_root
from src.rgpt.preflight.e2e import run_mini_pipeline


def test_mini_e2e_pipeline_execution() -> None:
    """Mini end-to-end pipeline must execute all stages on synthetic corpus within target time."""
    root = get_project_root()
    mini_dir = root / "tests" / "data" / "mini"

    res = run_mini_pipeline(mini_dir)
    assert res["status"] == "pass"
    assert res["stages"]["load_docs"]
    assert res["stages"]["tokenizers_roundtrip"]
    assert res["stages"]["equal_bytes_invariant"]
    assert res["stages"]["model_forward"]
    assert res["stages"]["bpc_evaluated"]
    assert res["stages"]["bootstrap_ci"]
