"""AWF — Algorithmic Weight Fabric.

A new way to represent neural network weights: store a generative program
instead of a tensor.
"""
__version__ = "0.5.0"

from .core import (
    CoordGenerator,
    AWFLinear,
    AWFLayerConfig,
    AWFTransformer,
    DenseTransformer,
    num_params,
)
from .event_training import (
    GradientEventTrainer,
    StandardTrainer,
    ActivationCollector,
)

__all__ = [
    "CoordGenerator",
    "AWFLinear",
    "AWFLayerConfig",
    "AWFTransformer",
    "DenseTransformer",
    "num_params",
    "GradientEventTrainer",
    "StandardTrainer",
    "ActivationCollector",
    "__version__",
]
