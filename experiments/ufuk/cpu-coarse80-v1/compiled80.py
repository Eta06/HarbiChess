"""ctypes adapter for the tiny no-fast-math coarse80 residual kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path


class Compiled80:
    def __init__(self, shared_object):
        self.path = Path(shared_object).resolve()
        self.library = ctypes.CDLL(str(self.path))
        self.function = self.library.coarse80_residual
        double_p = ctypes.POINTER(ctypes.c_double)
        uint64_p = ctypes.POINTER(ctypes.c_uint64)
        self.function.argtypes = [double_p, uint64_p, ctypes.c_int]
        self.function.restype = ctypes.c_double

    def __call__(self, theta, masks, mover_white):
        if len(theta) != 80 or len(masks) != 10:
            raise ValueError("coarse80 compiled input shape differs")
        weights = (ctypes.c_double * 80)(*(float(v) for v in theta))
        bitboards = (ctypes.c_uint64 * 10)(*(int(v) for v in masks))
        return float(self.function(weights, bitboards, int(bool(mover_white))))
