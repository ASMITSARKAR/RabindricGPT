"""
tests/data/test_audit.py : Tests for data audit checklist verification for Gate G1.
"""

from pathlib import Path

from src.rgpt.data.audit import generate_audit_report


def test_audit_report_generation(tmp_path: Path) -> None:
    """Audit report must generate JSON and Markdown records covering all criteria in 03 §9."""
    wiki_dir = tmp_path / "wikisource"
    wiki_dir.mkdir()
    sample_file = wiki_dir / "sample.txt"
    sample_file.write_text(
        "রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালে কলকাতায় জন্মগ্রহণ করেন। তিনি বিশ্বকবি হিসেবে পরিচিত।\n"
        "গীতাঞ্জলি কাব্যগ্রন্থের জন্য ১৯১৩ সালে তিনি সাহিত্যে নোবেল লাভ করেন।",
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"
    report = generate_audit_report(wikisource_dir=wiki_dir, fineweb2_dir=tmp_path, output_dir=out_dir)

    assert (out_dir / "g1_audit.json").exists()
    assert (out_dir / "g1_audit.md").exists()
    assert report["gate"] == "G1"
    assert report["passed"] is True
    assert report["bytes_per_char"] > 1.5
    assert report["chars_per_akshara"] > 0.5
    assert report["bengali_script_ratio"] >= 0.80
    assert report["checklist"]["whole_work_leakage_free"] is True
