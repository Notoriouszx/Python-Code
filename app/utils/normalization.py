from typing import Dict, Iterable, Tuple

import numpy as np


def l2_normalize(vec: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(vec, dtype=np.float64)
    denom = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / denom


def minmax_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sigma = float(np.std(x)) + eps
    return (x - mu) / sigma


def stack_modalities(
    parts: Iterable[Tuple[str, np.ndarray]],
) -> Tuple[np.ndarray, list[str]]:
    names: list[str] = []
    blocks: list[np.ndarray] = []
    for name, arr in parts:
        names.append(name)
        blocks.append(np.asarray(arr, dtype=np.float64).ravel())
    if not blocks:
        return np.array([]), []
    return np.concatenate(blocks), names
