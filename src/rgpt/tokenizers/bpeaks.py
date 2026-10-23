"""
src/rgpt/tokenizers/bpeaks.py : BPE over Aksharas tokenizer via reversible PUA mapping (stubbed until Gate G4).
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md §2, Phase E; 04_TOKENIZER_PLAN.md §4
Contract      : encode(text: str) -> list[int]; decode(ids: list[int]) -> str
Invariants    : Only activated after Gate G4; uses PUA range U+E000–U+F8FF without collisions
Failure modes : Called prior to G4 gate approval (raises loud error); PUA character in source data
Status        : TESTED
"""

from pathlib import Path
from typing import List, Union

from src.rgpt.tokenizers.base import BaseTokenizer


class BPEAksharaTokenizer(BaseTokenizer):
    """BPE over akshara units via reversible PUA code point mapping."""

    def __init__(self) -> None:
        pass

    def encode(self, text: str) -> List[int]:
        # TODO(phase E): Stubbed until Gate G4
        raise NotImplementedError("BPE over aksharas is scheduled only after Gate G4 passes.")

    def decode(self, ids: List[int]) -> str:
        # TODO(phase E): Stubbed until Gate G4
        raise NotImplementedError("BPE over aksharas is scheduled only after Gate G4 passes.")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Phase E (Post-G4)")

    def to_files(self, directory: Union[str, Path]) -> None:
        raise NotImplementedError("Phase E (Post-G4)")

    @classmethod
    def from_files(cls, directory: Union[str, Path]) -> "BPEAksharaTokenizer":
        raise NotImplementedError("Phase E (Post-G4)")

    def content_hash(self) -> str:
        raise NotImplementedError("Phase E (Post-G4)")
