"""Tiny deterministic transfer smoke experiment; not a scientific benchmark."""

import json
import numpy as np
from axisforge_bo.context import ContextMemory
from axisforge_bo.meta_learning import meta_step, task_meta_gradient

rng = np.random.default_rng(17)
mem = ContextMemory(2, 1)
for group in ("train", "positive", "negative"):
    for i in range(8):
        x = rng.uniform(-1, 1, 2)
        y = x[0] if group != "negative" else x[1]
        mem.append(x, [1], y, i, group)


def task(group):
    records = [r for r in mem.before(9) if r.group == group]
    x = np.array([r.decision for r in records])
    y = np.array([r.value for r in records])
    return x[:4], y[:4], x[4:], y[4:]


theta = np.array([[0.7], [0.7]])
train = task("train")
scenarios = {key: task(key) for key in ("positive", "negative")}
before = {key: task_meta_gradient(theta, *data)[0] for key, data in scenarios.items()}
for _ in range(5):
    theta, _ = meta_step(theta, [train], learning_rate=0.2)
after = {key: task_meta_gradient(theta, *data)[0] for key, data in scenarios.items()}
print(
    json.dumps(
        {
            "seed": 17,
            "steps": 5,
            "metric": "joint query NLL",
            "before": before,
            "after": after,
            "theta": theta.tolist(),
        },
        indent=2,
    )
)
