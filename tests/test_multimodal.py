import numpy as np
import pytest

torch = pytest.importorskip("torch")
from axisforge_bo.multimodal import Modality, MultimodalDeepKernelGP, sha256_bytes
from axisforge_bo.optimizer import BayesianOptimizer
from axisforge_bo.space import Box


def test_multimodal_fit_freeze_gradients_and_optimizer():
    sha = sha256_bytes(b"example encoder artifact")
    specs = [
        Modality("decision", 1, 2, True, "v1", sha),
        Modality("vision", 384, 2, False, "encoder-v1", sha),
    ]
    model = MultimodalDeepKernelGP(specs, seed=5)
    old = [p.detach().clone() for p in model.features.branches[1].parameters()]
    train_before = [p.detach().clone() for p in model.features.branches[0].parameters()]
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 385))
    y = np.array([0.1, 0.3, -0.5, 0.2])
    model.fit(x, y, steps=8)
    assert all(
        torch.equal(a, b) for a, b in zip(old, model.features.branches[1].parameters())
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(train_before, model.features.branches[0].parameters())
    )
    assert model.log_lengthscale.grad is not None and torch.isfinite(
        model.log_lengthscale.grad
    )
    mean, var = model.predict(x)
    assert np.isfinite(mean).all() and (var >= 0).all()
    opt = BayesianOptimizer(
        Box([[0, 1]]),
        model,
        seed=3,
        context_dimension=384,
        n_initial=1,
        pool_size=8,
        max_evaluations=2,
    )
    context = np.zeros(384)
    for _ in range(2):
        c = opt.ask(context)
        opt.tell(c, (c.x[0] - 0.4) ** 2)
    assert len(opt.records) == 2
    assert model.provenance()[1]["sha256"] == sha


def test_modality_rejects_invalid_metadata():
    with pytest.raises(ValueError):
        Modality("x", 1, 1, True, "v", "bad")
    with pytest.raises(ValueError):
        MultimodalDeepKernelGP([])
    with pytest.raises(TypeError):
        sha256_bytes("not bytes")


def test_pretrained_branch_and_metadata_validation():
    sha = sha256_bytes(b"weights")
    spec = Modality("vision", 384, 2, False, "v1", sha)
    state = {
        "0.weight": torch.full((2, 384), 0.125, dtype=torch.double),
        "0.bias": torch.zeros(2, dtype=torch.double),
    }
    model = MultimodalDeepKernelGP([spec], pretrained_branches={"vision": state})
    state["0.weight"].fill_(1.0)
    assert model.features.branches[0][0].weight[0, 0].item() == 0.125
    with pytest.raises(ValueError):
        MultimodalDeepKernelGP([spec], pretrained_branches={"unknown": state})
    with pytest.raises(ValueError):
        MultimodalDeepKernelGP([spec], pretrained_branches={"vision": {}})


def test_multimodal_configuration_rejects_invalid_and_nonfinite_weights():
    sha = sha256_bytes(b"x")
    with pytest.raises(ValueError):
        Modality("", 1, 1, True, "v", sha)
    with pytest.raises(ValueError):
        Modality("a", 0, 1, True, "v", sha)
    spec = Modality("a", 1, 1, False, "v", sha)
    with pytest.raises(ValueError):
        MultimodalDeepKernelGP([spec, spec])
    state = {"0.weight": torch.tensor([[float("nan")]]), "0.bias": torch.tensor([0.0])}
    with pytest.raises(ValueError):
        MultimodalDeepKernelGP([spec], pretrained_branches={"a": state})
