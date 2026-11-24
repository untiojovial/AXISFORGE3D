#!/usr/bin/env bash
# Builds the engine for WebAssembly and assembles the viewer into one page.
# Usage: apps/viewer/build.sh <output.html>
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?usage: build.sh <output.html>}"
cargo build --manifest-path "$root/Cargo.toml" -p axisforge-wasm \
  --target wasm32-unknown-unknown --profile wasm-release
wasm="$root/target/wasm32-unknown-unknown/wasm-release/axisforge_wasm.wasm"
python3 - "$root/apps/viewer" "$wasm" "$out" <<'PY'
import base64, pathlib, sys
viewer, wasm, out = map(pathlib.Path, sys.argv[1:])
read = lambda name: (viewer / name).read_text(encoding="utf-8")
head, body, script = read("styles.part.css.html"), read("body.part.html"), read("script.part.js")
import html, json
sample = viewer / "samples" / "ur5"
samples = {"ur5": {
    "urdf": (sample / "robot.urdf").read_text(encoding="utf-8"),
    "files": {m.name: base64.b64encode(m.read_bytes()).decode() for m in sorted((sample / "meshes").glob("*.stl"))},
}}
assert body.count("__THIRD_PARTY_LICENSE__") == 1 and script.count("__SAMPLES__") == 1
body = body.replace("__THIRD_PARTY_LICENSE__", html.escape((sample / "LICENSE").read_text(encoding="utf-8")).strip())
# json.dumps escapes every non-ASCII character, and "</" is broken up so the
# embedded text can never close the surrounding <script> element.
script = script.replace("__SAMPLES__", json.dumps(samples).replace("</", "<\\/"))
# The markup uses entities; the script is escaped here, so the page does not
# depend on the charset it happens to be served with.
assert all(ord(c) < 128 for c in head + body), "non-ASCII in markup: use entities"
script = "".join(c if ord(c) < 128 else "\\u%04x" % ord(c) for c in script)
assert script.count("__WASM_BASE64__") == 1
script = script.replace("__WASM_BASE64__", base64.b64encode(wasm.read_bytes()).decode())
out.write_text(head + body + "<script>\n" + script + "</script>\n", encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} kB)")
PY
