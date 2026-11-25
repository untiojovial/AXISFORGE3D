import pytest

torch = pytest.importorskip("torch")
from axisforge_bo.implicit import implicit_hypergradient


def test_implicit_gradient_matches_analytic_and_finite_difference():
    theta = torch.tensor([1.2], dtype=torch.double)
    inner = torch.tensor([2.4], dtype=torch.double)

    def f(t, u):
        return ((u - 2 * t) ** 2).sum() / 2

    def g(t, u):
        return (u**2 + 3 * t).sum()

    grad = implicit_hypergradient(f, g, theta, inner).item()

    def objective(t):
        return (2 * t) ** 2 + 3 * t

    eps = 1e-5
    assert grad == pytest.approx(8 * 1.2 + 3, rel=1e-12)
    assert grad == pytest.approx(
        (objective(1.2 + eps) - objective(1.2 - eps)) / (2 * eps), rel=1e-9
    )


def test_implicit_rejects_nonstationary_and_singular():
    t = torch.tensor([1.0], dtype=torch.double)
    with pytest.raises(ValueError, match="stationary"):
        implicit_hypergradient(
            lambda a, b: ((b - a) ** 2).sum(), lambda a, b: (a + b).sum(), t, t * 2
        )
    with pytest.raises(ArithmeticError, match="singular"):
        implicit_hypergradient(
            lambda a, b: (b**4).sum(), lambda a, b: (a + b).sum(), t, t * 0
        )


def test_implicit_two_dimensional_closed_form_and_input_validation():
    theta = torch.tensor([1.0, 2.0], dtype=torch.double)
    inner = 2 * theta

    def f(t, u):
        return ((u - 2 * t) ** 2).sum() / 2

    def g(t, u):
        return (u * u + t).sum()

    grad = implicit_hypergradient(f, g, theta, inner)
    torch.testing.assert_close(grad, 8 * theta + 1)
    with pytest.raises(ValueError):
        implicit_hypergradient(f, g, theta, inner, stationarity_tol=0)
    with pytest.raises(ValueError):
        implicit_hypergradient(f, g, theta.float(), inner)
    with pytest.raises(ValueError):
        implicit_hypergradient(f, g, theta * float("nan"), inner)


def test_implicit_rejects_nonfinite_outer_derivative():
    t = torch.tensor([0.0], dtype=torch.double)

    def inner(a, b):
        return ((b - a) ** 2).sum()

    def outer(a, b):
        return torch.sqrt(a).sum() + b.sum()

    with pytest.raises(ArithmeticError, match="nonfinite derivatives"):
        implicit_hypergradient(inner, outer, t, t)
