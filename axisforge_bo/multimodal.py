"""CPU float64 multimodal deep-kernel GP over precomputed external features.

Embeddings are supplied by callers; no foundation model is loaded or trained.
"""

import hashlib
import re
from dataclasses import dataclass

import torch
from torch import nn

from .deep_kernel import DeepKernelGP


@dataclass(frozen=True)
class Modality:
    name: str
    input_dim: int
    feature_dim: int
    trainable: bool
    source_version: str
    source_sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.source_version, str)
            or not self.source_version
        ):
            raise ValueError("name and source version are required")
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0
            for x in (self.input_dim, self.feature_dim)
        ) or not isinstance(self.trainable, bool):
            raise ValueError("invalid dimensions or trainable flag")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("source SHA-256 must be 64 lowercase hex characters")


def sha256_bytes(data):
    """Digest caller-supplied model artifact bytes, not input embedding values."""
    if not isinstance(data, bytes):
        raise TypeError("artifact must be bytes")
    return hashlib.sha256(data).hexdigest()


class _ModalityFeatures(nn.Module):
    def __init__(self, modalities):
        super().__init__()
        self.sizes = tuple(m.input_dim for m in modalities)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(m.input_dim, m.feature_dim), nn.Tanh())
                for m in modalities
            ]
        )
        for branch, modality in zip(self.branches, modalities):
            for p in branch.parameters():
                p.requires_grad_(modality.trainable)

    def forward(self, x):
        parts = torch.split(x, self.sizes, dim=-1)
        return torch.cat(
            [branch(part) for branch, part in zip(self.branches, parts)], dim=-1
        )


class MultimodalDeepKernelGP(DeepKernelGP):
    """Concatenate inputs in declared modality order; transform each separately.

    `fit` jointly learns unfrozen branches and GP hyperparameters on supplied
    data; `predict` returns latent mean and variance. Source SHA/version are
    asserted metadata only: caller must verify bytes and split data correctly.
    Freeze policy is fixed at construction. Checkpoint codec is not provided.
    """

    def __init__(self, modalities, *, seed=0, jitter=1e-8, pretrained_branches=None):
        modalities = tuple(modalities)
        if (
            not modalities
            or any(not isinstance(m, Modality) for m in modalities)
            or len({m.name for m in modalities}) != len(modalities)
        ):
            raise ValueError("modalities must be nonempty with unique names")
        super().__init__(sum(m.input_dim for m in modalities), seed=seed, jitter=jitter)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.features = _ModalityFeatures(modalities).double()
        self.modalities = modalities
        pretrained_branches = {} if pretrained_branches is None else pretrained_branches
        if not isinstance(pretrained_branches, dict) or not set(
            pretrained_branches
        ).issubset({m.name for m in modalities}):
            raise ValueError("unknown pretrained branch")
        for m, branch in zip(modalities, self.features.branches):
            if m.name in pretrained_branches:
                try:
                    branch.load_state_dict(pretrained_branches[m.name], strict=True)
                except (RuntimeError, TypeError) as exc:
                    raise ValueError(f"invalid pretrained branch: {m.name}") from exc
                if any(not torch.isfinite(p).all() for p in branch.parameters()):
                    raise ValueError("pretrained weights must be finite")

    def provenance(self):
        return tuple(
            {
                "name": m.name,
                "version": m.source_version,
                "sha256": m.source_sha256,
                "trainable": m.trainable,
                "input_dim": m.input_dim,
                "feature_dim": m.feature_dim,
            }
            for m in self.modalities
        )
