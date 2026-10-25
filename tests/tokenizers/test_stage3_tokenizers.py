"""
tests/tokenizers/test_stage3_tokenizers.py : Tests for Stage 3 tokenizer audit.
"""

from scripts.audits.audit_stage3_tokenizers import audit_stage3_tokenizers


def test_stage3_tokenizers_audit_passes():
    res = audit_stage3_tokenizers()
    for spec in res["tok_specs"]:
        assert spec["matches"] is True
    for hr in res["heldout_results"]:
        assert hr["lossless"] is True
        assert hr["unk_pct"] == 0.0
        if hr["tokenizer"] == "bpe16" and hr["corpus"] == "tagore_val":
            assert hr["cpt"] >= 2.5
