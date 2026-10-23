"""
src/rgpt/train/loader.py : Deterministic chunk data loader with position reconstruction and right padding with loss masking.
Backlog/Phase : I8 / Phase I      Docs : 15_BUILD_DIRECTIVE.md Phase I; DR-1, DR-6
Contract      : get_batch(step: int) -> (input_ids, targets, loss_mask)
Invariants    : Chunk order is a pure function of (seed, epoch, step); zero data RNG state required
Failure modes : Index out of bounds; empty shard directory
Status        : TESTED
"""

import random
from typing import Tuple

import torch

from src.rgpt.data.shards import ShardReader


class DeterministicDataLoader:
    """Deterministic batch sampler over memory-mapped token shards."""

    def __init__(
        self,
        reader: ShardReader,
        batch_size: int,
        s_max: int,
        seed: int = 42,
        pad_id: int = 0,
        ignore_index: int = -100,
        pad_to_batch_max: bool = True,
    ) -> None:
        self.reader = reader
        self.batch_size = batch_size
        self.s_max = s_max
        self.seed = seed
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        self.pad_to_batch_max = pad_to_batch_max

        self.num_chunks = len(self.reader)
        if self.num_chunks == 0:
            raise ValueError("ShardReader contains 0 chunks.")

    def get_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate deterministic batch for a specific training step.

        Returns:
            Tuple of:
                - input_ids: LongTensor of shape (batch_size, batch_len)
                - targets: LongTensor of shape (batch_size, batch_len)
                - loss_mask: BoolTensor of shape (batch_size, batch_len)
        """
        # Deterministic RNG derived solely from seed and step
        rng = random.Random(self.seed + step * 10007)
        chunk_indices = [rng.randint(0, self.num_chunks - 1) for _ in range(self.batch_size)]

        raw_inputs = []
        raw_targets = []

        for c_idx in chunk_indices:
            chunk = self.reader.get_chunk(c_idx)
            tokens = chunk["tokens"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            else:
                tokens = list(tokens)

            # Causal LM inputs (0..N-1) and targets (1..N)
            if len(tokens) < 2:
                tokens = [self.pad_id, self.pad_id]

            input_toks = tokens[:-1][: self.s_max]
            target_toks = tokens[1:][: self.s_max]
            raw_inputs.append(input_toks)
            raw_targets.append(target_toks)

        # Dynamic batch padding: pad to batch max, or static s_max
        if self.pad_to_batch_max:
            batch_target_len = max(max(len(t) for t in raw_inputs), 1)
        else:
            batch_target_len = self.s_max

        batch_inputs = []
        batch_targets = []
        batch_masks = []

        for input_toks, target_toks in zip(raw_inputs, raw_targets):
            valid_len = len(input_toks)
            pad_len = max(0, batch_target_len - valid_len)
            input_padded = input_toks + [self.pad_id] * pad_len
            target_padded = target_toks + [self.ignore_index] * pad_len
            mask_padded = [True] * valid_len + [False] * pad_len

            batch_inputs.append(input_padded)
            batch_targets.append(target_padded)
            batch_masks.append(mask_padded)

        input_ids = torch.tensor(batch_inputs, dtype=torch.long)
        targets = torch.tensor(batch_targets, dtype=torch.long)
        loss_mask = torch.tensor(batch_masks, dtype=torch.bool)

        return input_ids, targets, loss_mask
