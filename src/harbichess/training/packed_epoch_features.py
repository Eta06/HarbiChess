"""Lossless, epoch-owned float32 features; no RNG or persisted format change."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedEpochFeatures:
    size: int
    binary: np.ndarray
    exception_indices: np.ndarray
    exception_bits: np.ndarray

    @classmethod
    def encode(cls, values):
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        bits = array.view(np.uint32)
        indices = np.flatnonzero((bits != 0) & (bits != 0x3F800000)).astype(np.uint32)
        binary = np.packbits(bits == 0x3F800000)
        exceptions = bits[indices].copy()
        for stored in (binary, indices, exceptions):
            stored.setflags(write=False)
        return cls(array.size, binary, indices, exceptions)

    @property
    def nbytes(self):
        return self.binary.nbytes + self.exception_indices.nbytes + self.exception_bits.nbytes

    def decode(self):
        values = np.unpackbits(self.binary, count=self.size).astype(np.float32)
        values.view(np.uint32)[self.exception_indices] = self.exception_bits
        values.setflags(write=False)
        return values
