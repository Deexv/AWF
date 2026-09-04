"""AWF — Algorithmic Weight Fabric.

A new way to represent neural network weights: store a generative program
instead of a tensor.
"""
__version__ = "0.1.0"

from .core import (
    CoordGenerator,
    AWFLinear,
    AWFLayerConfig,
    AWFTransformer,
    DenseTransformer,
    num_params,
)

__all__ = [
    "CoordGenerator",
    "AWFLinear",
    "AWFLayerConfig",
    "AWFTransformer",
    "DenseTransformer",
    "num_params",
    "__version__",
]
