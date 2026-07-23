"""CUDA runtime bindings for cuda-python 12.x and 13.x."""

try:
    from cuda.bindings import runtime as cu
except ImportError:
    from cuda import cudart as cu  # type: ignore[no-redef]

__all__ = ["cu"]
