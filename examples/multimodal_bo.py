"""Run: OMP_NUM_THREADS=1 python examples/multimodal_bo.py (requires torch)."""

import numpy as np
from axisforge_bo.multimodal import Modality, MultimodalDeepKernelGP, sha256_bytes
from axisforge_bo.optimizer import BayesianOptimizer
from axisforge_bo.space import Box

# Illustrative bytes only: use the actual externally supplied encoder artifact.
digest = sha256_bytes(b"example encoder version 1")
model = MultimodalDeepKernelGP(
    [
        Modality("decision", 1, 2, True, "example-v1", digest),
        Modality("external_embedding", 384, 2, False, "example-v1", digest),
    ],
    seed=2,
)
bo = BayesianOptimizer(
    Box([[0.0, 1.0]]),
    model,
    context_dimension=384,
    seed=42,
    n_initial=2,
    pool_size=32,
    max_evaluations=4,
)
context = np.zeros(384)  # replace with an embedding available before the decision
for _ in range(4):
    candidate = bo.ask(context)
    bo.tell(candidate, (candidate.x[0] - 0.4) ** 2)
print(bo.best(context))
print(model.provenance())
