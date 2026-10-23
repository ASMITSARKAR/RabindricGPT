"""
tests/tokenizers/test_windows_gora_audit.py : Test correctness of Windows 0-99 and Gora compression profile audit.
"""

from pathlib import Path


def test_windows_gora_audit_pinned_file_exists():
    audit_file = Path("results/audits/audit_windows_gora.txt")
    assert audit_file.exists(), "audit_windows_gora.txt must exist"
    content = audit_file.read_text(encoding="utf-8")
    assert (
        "=== AUDIT: WINDOWS 0-99 & GORA COMPRESSION PROFILE ===" in content
    )
    assert "sonar_tori: 99 windows" in content
    assert "galpaguchchha_1: 2 windows" in content
    assert "Audit status: COMPLETE" in content


def test_gora_compression_profile_invariants():
    audit_file = Path("results/audits/audit_windows_gora.txt")
    content = audit_file.read_text(encoding="utf-8")

    # In Gora, all 100-window blocks have bpe16 chars/token >= 3.50
    lines = content.splitlines()
    gora_block_lines = [
        line_text
        for line_text in lines
        if line_text.startswith("Windows ") and "Windows 0-99" not in line_text
    ]
    assert len(gora_block_lines) == 9, (
        f"Expected 9 Gora blocks, found {len(gora_block_lines)}"
    )

    for line in gora_block_lines:
        parts = [p.strip() for p in line.split("|")]
        block_name = parts[0]
        c16 = float(parts[2])
        assert c16 >= 3.50, (
            f"Gora block {block_name} bpe16 chars/tok {c16} is below 3.50"
        )
