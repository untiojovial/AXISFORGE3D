"""Build the self-contained Colab notebook for the frozen visual features.

The notebook embeds the runtime code, the sealed development job and the CPU
reference for DINOv2, so it needs no repository checkout. It runs one or more
of OpenCLIP, DINOv2 and DINOv3 in a single pass and leaves, for each, a
bundle the repository's importer verifies.
"""
import base64, gzip, json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
files = ['research_grasping/__init__.py', 'research_grasping/phase4/__init__.py', 'research_grasping/phase4/jobs.py',
         'research_grasping/phase4/bundle_io.py', 'research_grasping/phase4/orchestrator.py',
         'research_grasping/phase4/frozen_models.py', 'research_grasping/phase4/spatial.py',
         'research_grasping/phase3/__init__.py', 'research_grasping/phase3/vision_openclip.py',
         'research_grasping/representations/__init__.py', 'research_grasping/representations/geometry.py',
         'research_grasping/representations/dino.py', 'research/phase3/dataset/manifest.lock.json',
         'research/phase4/automation/rgb_views.lock.json',
         'research/phase4/embedding_experiment/development-images.zip',
         'research/phase4/embedding_experiment/dinov2/global_embeddings.npz',
         'tools/research/phase4/download_models.py', 'tools/research/phase4/run_experiments.py',
         'research_grasping/phase4/visual_prior.py']
payload = {p: base64.b64encode(gzip.compress((root / p).read_bytes(), mtime=0)).decode() for p in files}


def md(s): return {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}


def code(s): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(True)}


INTRO = '''# AxisForge · características visuales congeladas (OpenCLIP, DINOv2, DINOv3)

Ejecuta todo (*Entorno de ejecución → Ejecutar todo*). Procesa las 352 vistas de desarrollo ya selladas
(22 objetos × 16 vistas) y deja, por cada modelo, un ZIP de características que el repositorio importa y verifica.
No hace falta clonar el repositorio: el código y las imágenes van dentro del notebook. Con GPU va más rápido;
en CPU también funciona (DINOv2-small tarda menos de un minuto).

**DINOv2** (`facebook/dinov2-small`, Apache-2.0) no necesita cuenta.

**DINOv3** (`facebook/dinov3-vits16-pretrain-lvd1689m`) está restringido por Meta. Una vez por cuenta:

1. Entra en https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m con tu cuenta, lee la licencia y
   solicita acceso. Meta lo aprueba a mano; hasta entonces la descarga devuelve 401/403.
2. Crea un token de lectura en https://huggingface.co/settings/tokens.
3. En Colab, abre el panel de la llave (*Secrets*), crea `HF_TOKEN` con ese token y activa *Notebook access*.

Después añade `'dinov3'` a `MODELS` en la celda de configuración. El notebook no acepta licencias en tu nombre
ni imprime el token. La redistribución de las características debe respetar las licencias del modelo y de los
datos (vistas derivadas de Google Scanned Objects, adaptación MuJoCo de Kevin Zakka, CC BY 4.0).

**Prior visual para el planificador de agarres.** Si además subes un `views.zip` de tus piezas (lo genera
`tools/research/phase4/visual_prior.py views` a partir de tus proyectos `.axisforge`), el notebook devuelve un
fichero de descriptores por modelo. Con él, `visual_prior.py prior` escribe el fichero que lee
`axisforge plan --prior`: las piezas ya resueltas que más se parecen a la nueva ordenan sus candidatos.
'''

INSTALL = '''# install: dependencias, sin reinstalar el PyTorch que ya trae Colab
import importlib.metadata as metadata, subprocess, sys
wanted = ['open_clip_torch==4.2.2', 'transformers==4.57.3', 'safetensors>=0.8,<0.9',
          'huggingface_hub>=0.36,<1.0', 'pillow', 'numpy']
try:
    print('torch', metadata.version('torch'), '· torchvision', metadata.version('torchvision'))
except metadata.PackageNotFoundError:
    wanted += ['torch==2.9.1', 'torchvision==0.24.1']
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *wanted], check=True)
'''

CONFIG = '''# Configuración. Añade 'dinov3' cuando tu cuenta tenga el acceso aprobado.
MODELS = ['dinov2']      # cualquiera de 'openclip', 'dinov2', 'dinov3'
REVISIONS = {}           # opcional, p. ej. {'dinov2': '<SHA de 40 caracteres>'}; por defecto, la revisión oficial actual
WORK_DIR = '/content/axisforge-work'      # pon una carpeta de Drive montado para reanudar entre sesiones
JOB_PATH = '/content/research/phase4/embedding_experiment/development-images.zip'
OUT_DIR = '/content/axisforge-bundles'
BATCH_SIZE = 16
VIEWS_ZIP = '/content/views.zip'          # opcional: vistas de tus piezas para el prior visual de agarre
'''

JOB = '''from pathlib import Path
if not Path(JOB_PATH).is_file():
    from google.colab import files
    print('Sube el ZIP del trabajo de imágenes preparado en local.')
    uploaded = files.upload()
    if len(uploaded) != 1: raise ValueError('Sube exactamente un ZIP')
    Path(JOB_PATH).write_bytes(next(iter(uploaded.values())))
from research_grasping.phase4.bundle_io import load_job
job = load_job(JOB_PATH)
print('Trabajo verificado', job['job_id'], '·', len(job['objects']), 'objetos')
'''

ACCESS = '''import os
OFFICIAL = {'dinov2': 'facebook/dinov2-small', 'dinov3': 'facebook/dinov3-vits16-pretrain-lvd1689m'}
unknown = sorted(set(MODELS) - {'openclip', 'dinov2', 'dinov3'})
if unknown or not MODELS: raise ValueError(f'MODELS admite openclip, dinov2 y dinov3; revisa {unknown or MODELS}')
STEPS = ('Pasos: (1) solicita acceso en https://huggingface.co/' + OFFICIAL['dinov3'] + ' con tu cuenta y espera la '
         'aprobación de Meta; (2) crea un token de lectura; (3) guárdalo como secreto HF_TOKEN de Colab con '
         '"Notebook access" activado. O quita dinov3 de MODELS.')

def secret_token():
    if os.getenv('HF_TOKEN'):
        return os.environ['HF_TOKEN']
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
    except Exception as exc:  # secreto ausente, sin permiso del notebook, o fuera de Colab
        raise RuntimeError('No hay un HF_TOKEN disponible para DINOv3. ' + STEPS) from exc
    if not token:
        raise RuntimeError('El secreto HF_TOKEN está vacío. ' + STEPS)
    return token

if 'dinov3' in MODELS:
    os.environ['HF_TOKEN'] = secret_token()
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    try:
        hf_hub_download(OFFICIAL['dinov3'], 'config.json', token=os.environ['HF_TOKEN'],
                        local_dir='/content/models/dinov3-access-check')
    except GatedRepoError as exc:
        raise RuntimeError('Tu cuenta todavía no tiene acceso aprobado a DINOv3. ' + STEPS) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError('Hugging Face rechazó el token para DINOv3 (' + type(exc).__name__ + '). ' + STEPS) from exc
    print('Acceso a DINOv3 comprobado.')
'''

RUN = '''import json, re, subprocess, sys
from pathlib import Path
import torch
from research_grasping.phase4.jobs import canonical, sha_file
from research_grasping.phase4.frozen_models import load_frozen
from research_grasping.phase4.orchestrator import atomic, run_extraction, export_bundle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Dispositivo:', device, '· torch', torch.__version__)

def official_checkpoint(kind):
    from huggingface_hub import HfApi
    repo = OFFICIAL[kind]
    token = os.getenv('HF_TOKEN') if kind == 'dinov3' else None
    pin_path = Path(WORK_DIR) / kind / 'model-source.json'
    revision = REVISIONS.get(kind)
    old = json.loads(pin_path.read_text()) if pin_path.exists() else None
    if old:
        if old['repo_id'] != repo or (revision and revision != old['revision']):
            raise ValueError('WORK_DIR ya tiene fijado otro modelo; usa un WORK_DIR nuevo')
        revision = old['revision']
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=token)
    if not re.fullmatch('[a-f0-9]{40}', info.sha): raise ValueError('El Hub no resolvió una revisión exacta')
    weights = next(f for f in info.siblings if f.rfilename == 'model.safetensors')
    if weights.lfs is None or not re.fullmatch('[a-f0-9]{64}', weights.lfs.sha256):
        raise ValueError('Falta el digest oficial de los pesos')
    pins = {'repo_id': repo, 'revision': info.sha, 'weights_sha256': weights.lfs.sha256}
    if old and pins != old: raise ValueError('Los metadatos oficiales difieren del fijado anterior')
    atomic(pin_path, canonical(pins))
    checkpoint = Path('/content/models') / kind
    subprocess.run([sys.executable, 'tools/research/phase4/download_models.py', '--model', kind,
                    '--revision', info.sha, '--out', str(checkpoint)], check=True)
    hashes = {n: sha_file(checkpoint / n) for n in ('config.json', 'model.safetensors', 'preprocessor_config.json')}
    if hashes['model.safetensors'] != pins['weights_sha256']:
        raise ValueError('Los pesos descargados no coinciden con el digest oficial')
    return checkpoint, info.sha, hashes

def run_model(kind):
    if kind == 'openclip':
        import urllib.request
        checkpoint = Path('/content/vit_b_32-quickgelu-laion400m_e31-d867053b.pt')
        if not checkpoint.exists():
            urllib.request.urlretrieve('https://github.com/mlfoundations/open_clip/releases/download/'
                                       'v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt', checkpoint)
        revision = 'v0.2-weights'
        adapter = load_frozen(kind, checkpoint, device=device)
        pins = {'revision': revision, 'checkpoint_sha256': adapter.provenance['weights_sha256']}
        ADAPTERS[kind] = adapter
    else:
        checkpoint, revision, hashes = official_checkpoint(kind)
        adapter = load_frozen(kind, checkpoint, config_sha256=hashes['config.json'],
                              weights_sha256=hashes['model.safetensors'],
                              processor_sha256=hashes['preprocessor_config.json'],
                              device=device, model_revision=revision)
        ADAPTERS[kind] = adapter
        pins = {'revision': revision, 'checkpoint_sha256': hashes['model.safetensors'],
                'processor_sha256': hashes['preprocessor_config.json'], 'config_sha256': hashes['config.json']}
    state = run_extraction(JOB_PATH, Path(WORK_DIR) / kind, adapter, model_revision=revision,
                           device=device, batch_size=BATCH_SIZE)
    bundle = Path(OUT_DIR) / f'{kind}_development.zip'
    index = export_bundle(JOB_PATH, Path(WORK_DIR) / kind, bundle)
    pins.update(model=kind, bundle=bundle.name, bundle_sha256=sha_file(bundle), status=state['status'],
                device=device, torch=torch.__version__, objects=len(index['objects']))
    return pins

RESULTS = {}
ADAPTERS = {}
for kind in MODELS:
    print('==', kind)
    RESULTS[kind] = run_model(kind)
    print(json.dumps(RESULTS[kind], indent=1))
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
(Path(OUT_DIR) / 'pins.json').write_text(json.dumps(RESULTS, indent=1, sort_keys=True) + '\\n')
for kind, pin in RESULTS.items():
    flags = '' if kind == 'openclip' else (f" --expected-revision {pin['revision']}"
             f" --expected-checkpoint-sha256 {pin['checkpoint_sha256']}"
             f" --expected-processor-sha256 {pin['processor_sha256']}")
    print(f"\\nImportar {kind} en el repositorio:\\nPYTHONPATH=. python tools/research/phase4/import_embedding_bundle.py "
          f"--job research/phase4/embedding_experiment/development-images.zip --bundle {pin['bundle']} "
          f"--destination imported-{kind} --analysis-out {kind}_analysis.json{flags}")
'''

COMPARE = '''# Si se ejecutó DINOv2, compara sus vectores globales con la referencia CPU del repositorio.
import io, zipfile
import numpy as np
if 'dinov2' in RESULTS:
    reference = np.load('research/phase4/embedding_experiment/dinov2/global_embeddings.npz', allow_pickle=False)
    with zipfile.ZipFile(Path(OUT_DIR) / RESULTS['dinov2']['bundle']) as bundle:
        index = json.loads(bundle.read('index.json'))
        ran = {row['object_id']: np.load(io.BytesIO(bundle.read(row['file'])))['global_embeddings'] for row in index['objects']}
    worst = 1.0
    for name, expected in zip(reference['ids'].tolist(), reference['global_embeddings']):
        got = ran[name]
        cosine = (got * expected).sum(axis=1) / np.linalg.norm(got, axis=1) / np.linalg.norm(expected, axis=1)
        worst = min(worst, float(cosine.min()))
    print(f'Coseno mínimo frente a la referencia CPU: {worst:.6f} (1 en CPU; en GPU cerca de 1 por redondeos)')
    if worst < 0.999: print('AVISO: diferencia mayor de lo esperado; revisa versión de torch/transformers y procesado.')
'''

VISUAL = '''# Prior visual de agarre: descriptores de tus piezas con cada modelo ejecutado.
from research_grasping.phase4 import visual_prior
DESCRIPTORS = []
if Path(VIEWS_ZIP).exists():
    for kind, adapter in ADAPTERS.items():
        out = Path(OUT_DIR) / f'{kind}_descriptors.npz'
        names = visual_prior.write_descriptors(adapter, VIEWS_ZIP, out, kind)
        DESCRIPTORS.append(out)
        print(f'{kind}: {len(names)} piezas -> {out.name}')
    print('En el repositorio: visual_prior.py library ... --descriptors <fichero> y visual_prior.py prior --model <modelo>')
else:
    print(f'Sin {VIEWS_ZIP}: se omite el prior visual (súbelo con el panel de archivos si lo quieres).')
'''

DOWNLOAD = '''from google.colab import files
for pin in RESULTS.values():
    files.download(str(Path(OUT_DIR) / pin['bundle']))
files.download(str(Path(OUT_DIR) / 'pins.json'))
for path in DESCRIPTORS:
    files.download(str(path))
'''

OUTRO = '''## Llevar los resultados al repositorio

La celda de ejecución imprime, por modelo, la orden de importación con la revisión y los hashes que la
verifican. El importador comprueba la seguridad del ZIP, la identidad del trabajo, el orden de las vistas y
los hashes de cada fichero antes de lanzar el análisis de desarrollo, y deja el análisis comparable con
`research/phase4/embedding_experiment/foundation_comparison.json`:

```bash
PYTHONPATH=. python -m research_grasping.phase4.model_comparison \\
  --inputs research/phase4/embedding_experiment/openclip_analysis.json \\
           research/phase4/embedding_experiment/dinov2/analysis.json dinov3_analysis.json \\
  --out foundation_comparison.json
```

Para el planificador de agarres, con los descriptores descargados:

```bash
T="python tools/research/phase4/visual_prior.py"
$T library --binary target/release/axisforge --part Banana cells/Banana.axisforge ... \\
  --descriptors dinov3_descriptors.npz --out library.json
$T prior --library library.json --binary target/release/axisforge --part Pera cells/Pera.axisforge \\
  --descriptors dinov3_descriptors.npz --model dinov3 --out pera-prior.json
target/release/axisforge plan cells/Pera.axisforge Pera --prior pera-prior.json
```

Los hashes prueban integridad, no que un entorno ajeno usara de verdad el modelo declarado: audita la
procedencia antes de usar los resultados científicamente.
'''

nb = {'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'}},
      'cells': [md(INTRO), code(INSTALL),
                code("import base64, gzip, json, os, pathlib\nos.chdir('/content')\npayload = " + repr(payload) +
                     "\nfor name, value in payload.items():\n    target=pathlib.Path(name); target.parent.mkdir(parents=True, exist_ok=True)\n"
                     "    target.write_bytes(gzip.decompress(base64.b64decode(value)))\n"),
                code(CONFIG), code(JOB), code(ACCESS), code(RUN), code(COMPARE), code(VISUAL), code(DOWNLOAD),
                md(OUTRO)]}
for i, cell in enumerate(nb['cells']):
    cell['id'] = f'axisforge-{i}'
(root / 'notebooks/AxisForge_Phase4_Frozen_Foundation_Models.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
