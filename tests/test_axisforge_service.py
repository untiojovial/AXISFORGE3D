from pathlib import Path
import shutil
import pytest
from axisforge_bo.integrations.axisforge3d import WasmService

WASM = Path("target/wasm32-unknown-unknown/wasm-release/axisforge_wasm.wasm")


@pytest.mark.skipif(
    not WASM.exists() or shutil.which("node") is None,
    reason="existing WASM artifact and node required",
)
def test_actual_service():
    service = WasmService(WASM)
    for operation in ("describe", "list_extensions"):
        result = service.request(operation)
        assert result.status == "success" and len(result.wasm_sha256) == 64
    with pytest.raises(ValueError):
        service.request("plan_pick")


def test_timeout_and_missing_node(tmp_path):
    binary = tmp_path / "wasm"
    binary.write_bytes(b"not wasm")
    result = WasmService(binary, node="missing-axisforge-node").request("describe")
    assert result.status == "error"


@pytest.mark.skipif(
    not WASM.exists() or shutil.which("node") is None,
    reason="existing WASM artifact and node required",
)
def test_snapshot_evaluator():
    from axisforge_bo.integrations.axisforge3d import SnapshotEvaluator
    from axisforge_bo.optimizer import Candidate

    e = SnapshotEvaluator(WasmService(WASM))
    outcome = e.evaluate(Candidate(0, (0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0), ()))
    assert outcome.status == "success" and outcome.value > 0
    assert len(e.last_result.wasm_sha256) == 64
    assert e.evaluate(Candidate(1, (1.0,), ())).status == "error"


def test_changed_artifact_rejected(tmp_path):
    binary = tmp_path / "wasm"
    binary.write_bytes(b"old")
    service = WasmService(binary)
    binary.write_bytes(b"new")
    result = service.request("describe")
    assert result.status == "error" and "changed" in result.detail


@pytest.mark.skipif(
    not WASM.exists() or shutil.which("node") is None,
    reason="existing WASM artifact and node required",
)
def test_snapshot_evaluator_units_and_cost():
    from axisforge_bo.integrations.axisforge3d import SnapshotEvaluator
    from axisforge_bo.optimizer import Candidate

    outcome = SnapshotEvaluator(WasmService(WASM)).evaluate(
        Candidate(1, (0.0, 0.3, 1.0, 0.0, 1.84, 0.0, 0.0, 0.0), ())
    )
    assert (
        outcome.units
        and outcome.cost_seconds >= 0
        and outcome.provenance.startswith("sha256:")
    )


def test_service_errors(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from axisforge_bo.integrations.axisforge3d import SnapshotEvaluator, ServiceResult
    from axisforge_bo.optimizer import Candidate

    wasm = tmp_path / "w"
    wasm.write_bytes(b"bytes")
    with pytest.raises(ValueError):
        WasmService(wasm, timeout=0)
    service = WasmService(wasm)
    with pytest.raises(ValueError):
        service.request("snapshot", [float("nan")])

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("node", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert service.request("describe").status == "timeout"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="error", stdout=""),
    )
    assert service.request("describe").status == "error"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stderr="", stdout="{}"),
    )
    assert service.request("describe").status == "error"
    e = SnapshotEvaluator(service)
    with pytest.raises(TypeError):
        e.evaluate(None)
    monkeypatch.setattr(
        service,
        "request",
        lambda *a: ServiceResult("success", {"data": {}}, "", 0, "x"),
    )
    assert e.evaluate(Candidate(1, (0,), ())).provenance == "sha256:x"
    monkeypatch.setattr(
        service,
        "request",
        lambda *a: ServiceResult(
            "success", {"data": {"conditioning": {"manipulability": None}}}, "", 0, "x"
        ),
    )
    assert e.evaluate(Candidate(1, (0,), ())).provenance == "sha256:x"
