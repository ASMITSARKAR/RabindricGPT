"""
tests/data/test_stage2_shards.py : Unit tests for Stage 2 shard manifests, invariants, and round-trip assertions.
"""

from scripts.audits.audit_stage2_shards import audit_stage2_shards


def test_stage2_shards_all_invariants_pass():
    """Verify all P3 shard equality checks, whole-work splits, and round-trip pass."""
    res = audit_stage2_shards(n_roundtrip_chunks=50, seed=42)
    assert res["all_p3_passed"] is True, "P3 shard character equality check failed"
    for tid, r in res["roundtrip_results"].items():
        assert r["passed"] is True, f"Lossless roundtrip failed for {tid}"
    assert res["epochs_fw2_bytes"] > 0
    assert res["epochs_tagore_bytes"] > 0
