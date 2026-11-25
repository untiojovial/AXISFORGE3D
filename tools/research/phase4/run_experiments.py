"""Run the authorized feature-extraction stage of the development DAG."""
import argparse
import json
from pathlib import Path
from research_grasping.phase4.orchestrator import run_extraction, export_bundle, validate_declarative

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--job', type=Path, required=True); p.add_argument('--work', type=Path, required=True)
p.add_argument('--bundle', type=Path, required=True); p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--config', type=Path, help='declarative development JSON')
p.add_argument('--model', choices=['openclip','dinov2','dinov3'])
p.add_argument('--revision'); p.add_argument('--device'); p.add_argument('--batch-size', type=int)
p.add_argument('--weights-sha256'); p.add_argument('--config-sha256'); p.add_argument('--processor-sha256')
p.add_argument('--mode', choices=['development','confirmatory'])
a = p.parse_args()
c = validate_declarative(json.loads(a.config.read_text())) if a.config else {}
mode = a.mode or c.get('mode', 'development')
if mode == 'confirmatory': raise SystemExit('BLOCKED: confirmatory protocol and independent approval required')
model_kind = a.model or c.get('model'); revision = a.revision or c.get('model_revision')
device = a.device or c.get('device', 'cpu'); batch_size = a.batch_size if a.batch_size is not None else c.get('batch_size', 8)
if not model_kind or not revision: p.error('--model and --revision required without declarative config')
from research_grasping.phase4.frozen_models import load_frozen
model = load_frozen(model_kind, a.checkpoint, config_sha256=a.config_sha256, weights_sha256=a.weights_sha256, processor_sha256=a.processor_sha256, device=device, model_revision=revision)
run_extraction(a.job,a.work,model,model_revision=revision,device=device,batch_size=batch_size)
export_bundle(a.job,a.work,a.bundle)
print('COMPLETED',a.bundle)
