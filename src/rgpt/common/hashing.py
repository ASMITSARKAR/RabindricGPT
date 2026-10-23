"""
src/rgpt/common/hashing.py : Deterministic SHA256 hashing for files, text, and in-memory byte streams.
Backlog/Phase : I-none / Phase B      Docs : 15_BUILD_DIRECTIVE.md Phase B
Contract      : text: str | bytes | Path -> hex_digest: str
Invariants    : Standard NIST SHA256 test vectors match; streaming file hash matches in-memory hash
Failure modes : File unreadable; non-existent path
Status        : TESTED
"""

import hashlib
from pathlib import Path
from typing import Union


def hash_text(text: str) -> str:
    """Compute hex-encoded SHA256 digest of UTF-8 encoded text.

    Args:
        text: String to hash.

    Returns:
        Hex-encoded SHA256 digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    """Compute hex-encoded SHA256 digest of byte buffer.

    Args:
        data: Raw byte array.

    Returns:
        Hex-encoded SHA256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute hex-encoded SHA256 digest of file by chunked streaming.

    Args:
        path: Path to file.
        chunk_size: Read buffer size in bytes (default 64KB).

    Returns:
        Hex-encoded SHA256 digest string.

    Raises:
        FileNotFoundError: If path does not exist.
        IsADirectoryError: If path is a directory.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
