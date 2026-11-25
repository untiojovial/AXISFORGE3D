"""Extract frozen, offline features from one object's explicitly ordered RGB views."""

import argparse
import json
import platform
import resource
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from .cache import FeatureCache, cache_key
from .dino import extract_tokens, load_local_dino, sha256_file
from .preprocess import preprocess_rgb


def extract_views(
    paths,
    directory,
    weights_sha,
    config_sha,
    family,
    recipe,
    device,
    cache_path,
    repeats,
):
    import torch
    from PIL import Image

    if not paths or repeats < 1 or device not in ("cpu", "cuda"):
        raise ValueError(
            "provide at least one view and a positive repeat count on cpu or cuda"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA hardware is unavailable")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    sync()
    start = time.perf_counter()
    model = load_local_dino(directory, weights_sha, config_sha, family, device=device)
    sync()
    load_s = time.perf_counter() - start
    cache = FeatureCache(cache_path)
    version = {
        pkg: metadata.version(pkg)
        for pkg in ("torch", "transformers", "safetensors", "numpy", "Pillow")
    }
    hardware = torch.cuda.get_device_name() if device == "cuda" else platform.machine()
    specs = {
        "implementation": sha256_file(__file__),
        "loader_implementation": sha256_file(Path(__file__).with_name("dino.py")),
        "preprocess_implementation": sha256_file(
            Path(__file__).with_name("preprocess.py")
        ),
        "family": family,
        "weights_sha256": weights_sha,
        "config_sha256": config_sha,
        "device": device,
        "hardware": hardware,
        "libraries": version,
        "dtype": str(next(model.parameters()).dtype),
    }
    views = []
    cls_tokens = []
    for path in map(Path, paths):
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(
                    f"expected RGB source, got {image.mode} for {path.name}"
                )
            rgb = np.array(image)
        inputs, affine, prep = preprocess_rgb(rgb, **recipe)
        provenance = {**specs, "image_sha256": sha256_file(path), "preprocessing": prep}
        key = cache_key(provenance, {"pixels": rgb})
        values = cache.get(key)
        hit = values is not None
        timings = []
        warm_s = None
        cuda_peak = None
        if not hit:
            inputs = inputs.to(device)
            sync()
            start = time.perf_counter()
            cls, patches = extract_tokens(model, inputs)
            sync()
            warm_s = time.perf_counter() - start
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for _ in range(repeats):
                sync()
                start = time.perf_counter()
                cls, patches = extract_tokens(model, inputs)
                sync()
                timings.append(time.perf_counter() - start)
            if device == "cuda":
                cuda_peak = int(torch.cuda.max_memory_allocated())
            if not np.isfinite(cls).all() or not np.isfinite(patches).all():
                raise ArithmeticError("nonfinite model output")
            values = {"cls": cls[0], "patches": patches[0], "affine": affine}
            cache.put(key, values)
        cls_tokens.append(values["cls"])
        views.append(
            {
                "name": path.name,
                **provenance,
                "key": key,
                "cache_hit": hit,
                "patch_shape": list(values["patches"].shape),
                "affine": values["affine"].tolist(),
                "warmup_seconds": warm_s,
                "inference_and_host_copy_seconds": timings,
                "peak_cuda_allocated_bytes": cuda_peak,
            }
        )
    group = {
        "ordered_view_keys": [v["key"] for v in views],
        "pooling": "unweighted_mean_cls_v1",
    }
    aggregate_key = cache_key(group, {})
    cache.put(aggregate_key, {"mean_cls": np.mean(np.stack(cls_tokens), axis=0)})
    return {
        "views": views,
        "aggregate": {**group, "key": aggregate_key},
        "model_load_seconds": load_s,
        "device": device,
        "hardware": hardware,
        "process_peak_rss_kib_linux": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--family", choices=["dinov2", "dinov3"], required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = extract_views(
        args.images,
        args.model_dir,
        args.weights_sha256,
        args.config_sha256,
        args.family,
        json.loads(args.recipe.read_text()),
        args.device,
        args.cache,
        args.repeats,
    )
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
