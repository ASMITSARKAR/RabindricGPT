from pathlib import Path

for p in sorted(Path("data/cleaned/tagore").glob("*.txt")):
    text = p.read_text(encoding="utf-8")
    print(f"=== File: {p.name} ({len(text):,} chars) ===")
    for pat in ["::", "{{", "[[", "'''", "<"]:
        count = text.count(pat)
        print(f"  Pattern '{pat}': {count} occurrences")
