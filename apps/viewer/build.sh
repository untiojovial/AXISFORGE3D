#!/usr/bin/env bash
# Builds the engine for WebAssembly and embeds it in the viewer page.
# Usage: apps/viewer/build.sh <output.html>
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?usage: build.sh <output.html>}"
cargo build --manifest-path "$root/Cargo.toml" -p axisforge-wasm \
  --target wasm32-unknown-unknown --profile wasm-release
wasm="$root/target/wasm32-unknown-unknown/wasm-release/axisforge_wasm.wasm"
python3 - "$root/apps/viewer/index.template.html" "$wasm" "$out" <<'PY'
import base64, pathlib, sys
template, wasm, out = map(pathlib.Path, sys.argv[1:])
page = template.read_text(encoding="utf-8")
assert page.count("__WASM_BASE64__") == 1
out.write_text(page.replace("__WASM_BASE64__", base64.b64encode(wasm.read_bytes()).decode()), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} kB)")
PY
