"""AxisForgeBO public CPU API. Research backends are imported on demand."""

from .acquisitions import (
    expected_improvement,
    log_expected_improvement,
    probability_improvement,
)
from .context import ContextMemory
from .evaluation import Evaluation, MockEvaluator
from .models import ExactGP, StationaryKernel
from .optimizer import BayesianOptimizer, Candidate, register_acquisition
from .space import Box, Categorical, Integer, MixedSpace, Real

__all__ = [
    "BayesianOptimizer",
    "Box",
    "Candidate",
    "Categorical",
    "ContextMemory",
    "Evaluation",
    "ExactGP",
    "Integer",
    "MixedSpace",
    "MockEvaluator",
    "Real",
    "StationaryKernel",
    "expected_improvement",
    "log_expected_improvement",
    "probability_improvement",
    "register_acquisition",
]
