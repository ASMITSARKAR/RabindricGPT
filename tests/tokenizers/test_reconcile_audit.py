"""
tests/tokenizers/test_reconcile_audit.py : Automated unit tests verifying shard index
reconciliation, lossless round-trip, and zero UNK rate.
"""

from scripts.audits.audit_reconcile import audit_shards_reconciliation


def test_audit_reconciliation_zero_discrepancy() -> None:
    res = audit_shards_reconciliation()
    # 1. Shard agreement
    for r in res["shard_rows"]:
        assert r["agreement"], f"Discrepancy in {r['tokenizer']} {r['split']}: {r['diff_pct']:.4%}"

    # 2. Sample stats: zero CRLF
    assert res["sample_stats"]["crlf_count"] == 0

    # 3. UNK rate and round trip
    for r in res["unk_and_roundtrip"]:
        assert r["unk_rate"] == 0.0, f"Non-zero UNK in {r['tokenizer']} on {r['corpus']}"
        assert r["round_trip_lossless"], f"Lossy round-trip in {r['tokenizer']} on {r['corpus']}"
