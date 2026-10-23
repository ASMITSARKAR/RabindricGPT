"""
tests/common/test_hashing.py : Tests for SHA256 string, byte, and file hashing functions.
"""

from pathlib import Path

from src.rgpt.common.hashing import hash_bytes, hash_file, hash_text


def test_known_sha256_vectors() -> None:
    """Hashing standard NIST test strings must match known SHA256 digests."""
    # NIST test vectors
    # sha256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    assert hash_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert hash_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    # sha256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    assert hash_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    # Bengali text vector
    bengali_text = "গীতাঞ্জলি - রবীন্দ্রনাথ ঠাকুর"
    expected = hash_bytes(bengali_text.encode("utf-8"))
    assert hash_text(bengali_text) == expected


def test_file_hash_matches_bytes(tmp_path: Path) -> None:
    """Streaming file hash must match in-memory bytes hash."""
    test_file = tmp_path / "test_data.bin"
    payload = b"RabindricGPT deterministic hashing verification payload" * 1000
    test_file.write_bytes(payload)

    assert hash_file(test_file) == hash_bytes(payload)
