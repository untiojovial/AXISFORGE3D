"""Isolated adapter to the existing AxisForge WASM JSON service.

Each request creates a fresh demo session. This validates service connectivity;
no arbitrary grasp-evaluation or physical compatibility claim is made.
"""

import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..evaluation import Evaluation
from ..optimizer import Candidate


@dataclass(frozen=True)
class ServiceResult:
    status: str
    data: dict | None
    detail: str
    elapsed_seconds: float
    wasm_sha256: str


class WasmService:
    def __init__(self, wasm_path, timeout=10.0, node="node"):
        self.wasm_path = Path(wasm_path).resolve(strict=True)
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        self.timeout = float(timeout)
        self.node = node
        self.sha256 = hashlib.sha256(self.wasm_path.read_bytes()).hexdigest()
        self.bridge = Path(__file__).with_name("axisforge_wasm_bridge.js")

    def request(self, operation: str, values=None) -> ServiceResult:
        if operation not in {"describe", "list_extensions", "snapshot"}:
            raise ValueError("unsupported service operation")
        started = time.monotonic()
        try:
            current_hash = hashlib.sha256(self.wasm_path.read_bytes()).hexdigest()
            if current_hash != self.sha256:
                return ServiceResult(
                    "error",
                    None,
                    "WASM artifact changed since adapter construction",
                    time.monotonic() - started,
                    current_hash,
                )
        except OSError as exc:
            return ServiceResult(
                "error", None, str(exc), time.monotonic() - started, self.sha256
            )
        try:
            request = {"op": operation}
            if operation == "snapshot":
                if (
                    not isinstance(values, (tuple, list))
                    or not values
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        for v in values
                    )
                ):
                    raise ValueError("snapshot requires finite joint values")
                request.update({"values": values, "carry": None, "grasping": None})
            proc = subprocess.run(
                [self.node, str(self.bridge), str(self.wasm_path)],
                input=json.dumps(request, allow_nan=False),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ServiceResult(
                "timeout",
                None,
                "worker timed out",
                time.monotonic() - started,
                self.sha256,
            )
        except OSError as exc:
            return ServiceResult(
                "error", None, str(exc), time.monotonic() - started, self.sha256
            )
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            return ServiceResult(
                "error", None, proc.stderr[:1024], elapsed, self.sha256
            )
        try:
            envelope = json.loads(
                proc.stdout,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            if not isinstance(envelope, dict) or envelope.get("sha256") != self.sha256:
                raise ValueError("executed WASM digest mismatch")
            result = envelope.get("reply")
            kind = {
                "describe": "description",
                "list_extensions": "extensions",
                "snapshot": "snapshot",
            }[operation]
            if (
                not isinstance(result, dict)
                or result.get("ok") is not True
                or result.get("kind") != kind
                or "data" not in result
            ):
                raise ValueError("unexpected service response")
            if not isinstance(
                result["data"], list if operation == "list_extensions" else dict
            ):
                raise ValueError("unexpected service data")
        except (ValueError, TypeError) as exc:
            return ServiceResult("error", None, str(exc), elapsed, self.sha256)
        return ServiceResult("success", result, "", elapsed, self.sha256)


class SnapshotEvaluator:
    """Evaluate Jacobian manipulability for a full joint vector in a fresh demo.

    The metric is dimensionful and is not a grasp quality or safety measure.
    Returned service result (including provenance) is available as last_result.
    """

    def __init__(self, service):
        self.service = service
        self.last_result = None

    def evaluate(self, candidate: Candidate) -> Evaluation:
        if not isinstance(candidate, Candidate):
            raise TypeError("expected Candidate")
        self.last_result = self.service.request("snapshot", candidate.x)
        result = self.last_result
        if result.status != "success":
            return Evaluation(
                result.status,
                detail=result.detail,
                cost_seconds=result.elapsed_seconds,
                provenance="sha256:" + result.wasm_sha256,
            )
        data = result.data["data"]
        conditioning = data.get("conditioning")
        if not isinstance(conditioning, dict):
            return Evaluation(
                "error",
                detail="conditioning unavailable",
                cost_seconds=result.elapsed_seconds,
                provenance="sha256:" + result.wasm_sha256,
            )
        value = conditioning.get("manipulability")
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            return Evaluation(
                "error",
                detail="invalid manipulability",
                cost_seconds=result.elapsed_seconds,
                provenance="sha256:" + result.wasm_sha256,
            )
        return Evaluation(
            "success",
            value=float(value),
            units="Jacobian determinant metric in mixed SI units",
            cost_seconds=result.elapsed_seconds,
            provenance="sha256:" + result.wasm_sha256,
        )
