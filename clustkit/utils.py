"""Shared utilities: logging, GPU memory management, and helpers."""

import logging
import time
from contextlib import contextmanager

import numpy as np


def get_logger(name: str = "clustkit") -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


@contextmanager
def timer(label: str):
    """Context manager that logs elapsed time for a block."""
    start = time.perf_counter()
    logger.info(f"{label}...")
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{label} done ({elapsed:.2f}s)")


def gpu_available() -> bool:
    """Check if CuPy and a CUDA GPU are available."""
    try:
        import cupy as cp  # noqa: F401

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def to_gpu(array: np.ndarray):
    """Move a numpy array to GPU via CuPy."""
    import cupy as cp

    return cp.asarray(array)


def to_cpu(array) -> np.ndarray:
    """Move a CuPy array back to CPU."""
    import cupy as cp

    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def auto_lsh_params(threshold: float, sensitivity: str) -> dict:
    """Choose LSH parameters (num_tables, num_bands) based on threshold and sensitivity.

    Higher threshold → pairs are more similar → easier to detect → fewer tables needed.
    Higher sensitivity → more tables, fewer bands → higher recall, more candidates.
    """
    # Base parameters tuned for ~90% identity threshold
    base_tables = {
        "low": 16,
        "medium": 32,
        "high": 64,
    }
    base_bands = {
        "low": 8,
        "medium": 4,
        "high": 2,
    }

    L = base_tables.get(sensitivity, 32)
    b = base_bands.get(sensitivity, 4)

    # Adjust for threshold: lower thresholds need more tables for recall
    if threshold < 0.7:
        L = int(L * 1.5)
        b = max(1, b - 1)
    elif threshold < 0.5:
        L = L * 2
        b = max(1, b - 2)

    return {"num_tables": L, "num_bands": b}
