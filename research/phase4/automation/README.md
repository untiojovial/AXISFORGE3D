# Portable frozen embedding jobs — development only

`development.json` is the declarative reference configuration; `validate_declarative` rejects incompatible and confirmatory settings. The currently executable path is image preparation → frozen extraction → bundle verification → development BO callback. Other registered DAG nodes are plans, not a claim that candidate generation, physics, statistics or audit have run automatically. Confirmatory execution is blocked. The local runner accepts the equivalent command-line values. No external paid services or tunnels are used.

From the repository root, using a Python environment with numpy, Pillow, torch 2.9.1, torchvision 0.24.1, open_clip_torch 4.2.2 and transformers 4.57.3:

```bash
PYTHONPATH=. python tools/research/phase4/prepare_embedding_job.py --root /workspace/axisforge-phase3-data --observations /workspace/axisforge-phase3-data/vision-development-v3 --output /workspace/axisforge-phase3-data/phase4-development-job.zip
PYTHONPATH=. python tools/research/phase4/run_experiments.py --job /workspace/axisforge-phase3-data/phase4-development-job.zip --work /workspace/axisforge-phase3-data/phase4-work --bundle /workspace/axisforge-phase3-data/phase4-bundle.zip --model openclip --revision v0.2-weights --checkpoint /workspace/axisforge-phase3-data/weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt
PYTHONPATH=. python tools/research/phase4/import_embedding_bundle.py --job /workspace/axisforge-phase3-data/phase4-development-job.zip --bundle /workspace/axisforge-phase3-data/phase4-bundle.zip --destination /workspace/axisforge-phase3-data/phase4-import
```

The notebook `notebooks/AxisForge_Phase4_Frozen_Foundation_Models.ipynb` embeds its runtime and sealed job; use Run all. It defaults to OpenCLIP; DINO requires an exact official revision. DINOv3 requires user-authorized access and potentially HF_TOKEN set in the environment, never committed or printed. If access is denied, the job remains blocked rather than replacing weights with random weights. For persistence mount authorized Drive and set WORK_DIR before execution. Transfer the resulting ZIP back to the originating machine; the import command invokes development BO only after archive and object/view/hash verification. A missing local analysis dataset reports BLOCKED and describes resumption. The reference job and source RGB are development only, with 16 ordered views for each of the 22 objects. It excludes depth, geometry, credentials, weights and final test assets.

Output arrays are `global_embeddings` (V,D), `patch_embeddings` (V,P,D) and `view_ids` (V). `index.json` records each object/split/file/hash, job ID, dataset manifest, checkpoint and processor hashes, model revision, token grid/pixel transform and per-object costs. GPU OOM reduces batch size; each successful object and run state is written atomically. Compatible resume verifies hashes. An integrity check does not establish that a remote party actually ran the claimed checkpoint; external execution and licensing require independent audit. Do not treat a partial job or smoke test as a model comparison.

For externally supplied DINO bundles, pass `--expected-checkpoint-sha256`, `--expected-revision` and `--expected-processor-sha256` from an independently verified official checkout. The default OpenCLIP hash is pinned in the importer. This comparison checks a claim against a trusted digest; it does not prove that remote code actually used those weights. Capture and independently audit execution and processor provenance before a scientific claim.

The input PNGs are rendered from Google Scanned Objects, attributed to Downs et al. and the MuJoCo adaptation by Kevin Zakka; the asset provider declares CC BY 4.0 for the meshes. The source and frozen split are documented in `research/phase3/dataset/README.md`. Derived RGB inputs and embedded notebook job share this attribution; consult model-specific terms before redistributing derived embeddings.

The separate CPU experiment workflow is available through `PYTHONPATH=. python tools/research/phase4/run_workflow.py --help`; its profiles, dependencies and blocked stages are recorded by that entrypoint. A registered DAG node is not itself evidence that an experiment ran.

A local OpenCLIP execution processed 22 objects/352 views, then a separate CPU execution reproduced all 22 NPZ byte for byte (max absolute array difference 0) with checkpoint SHA-256 `d867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e`. Bundle and development callback outputs remain local at `/workspace/axisforge-phase4-data/openclip-rerun.zip`, `openclip-rerun-import/`, and `openclip-rerun-analysis/analysis.json`. This is repeatability on one environment, not GPU reproducibility or proof of remote executions. The notebook's code cells were syntax checked and their embedded code and job compared byte for byte with the repository; an actual Colab execution remains pending.

If analysis fails after installation (for example because a dependency is missing), the importer exits with code 2. Restore the dependency or input and rerun exactly the same import command. The installer reuses an existing directory only after revalidating the bundle against the job, checking its exact installed file set, canonical index and all feature hashes, and rejecting symlinks; a corrupted or different installation is rejected. A callback exception outside the explicitly handled missing-data/dependency cases propagates as a failure and can likewise be retried.

To regenerate and check the embedded notebook runtime and 22-object job without network access:

```bash
python tools/research/phase4/build_colab_notebook.py
PYTHONPATH=. python tools/research/phase4/verify_notebook_offline.py
PYTHONPATH=. python -m pytest -q tests/phase4/test_jobs_bundle.py
```

The offline check executes the notebook's bootstrap, settings and sealed-job validation cells in a temporary workspace with sockets disabled; it also compiles the remaining Python cells and compares all embedded files with the repository. With no checkpoint argument this short check skips installation and inference. The additional offline full CPU check is documented below. The exact per-object NPZ SHA-256 values from both CPU runs and the job, bundle and notebook hashes are in `openclip_reproduction.json`. ZIP bundles differ because their run metadata differs; the 22 feature NPZ files match byte for byte. Generated notebook content must be refreshed with the builder after edits to an embedded runtime file or the job.

## Notebook autónomo y ejecución completa comprobada en CPU

La versión final incluye también el módulo de geometría que requiere el adaptador al cargar. Ejecutamos sin red sus cinco celdas Python de trabajo en un proceso aislado, con el checkpoint OpenCLIP previamente descargado. Las 352 vistas produjeron los mismos bytes NPZ que el runner independiente. La prueba usa un sustituto local para la interfaz de descarga del navegador. La instalación con pip, el servicio Colab y CUDA requieren su propia ejecución. El artefacto de reproducción anterior conserva la huella del notebook que comprobó en aquel momento. [Ejecución del notebook final](notebook_cpu.json)

Para usar DINO, cambiar `MODEL` a `dinov2` o `dinov3` y ejecutar todas las celdas. En la primera ejecución se obtiene la revisión oficial de Hugging Face y el hash LFS de sus pesos. Se fija la revisión para reanudar y se acepta opcionalmente una revisión concreta desde el principio. Para DINOv3 hace falta que la cuenta tenga acceso concedido y que `HF_TOKEN` esté en su entorno privado o en los secretos de Colab con acceso concedido al notebook. Los tres valores esperados del importador deben comprobarse frente al checkout oficial independiente antes de ejecutar el análisis. [Notebook](../../../notebooks/AxisForge_Phase4_Frozen_Foundation_Models.ipynb)

```bash
python tools/research/phase4/verify_notebook_offline.py --openclip "$WEIGHTS" --compare-bundle "$BUNDLE" --report notebook-cpu.json
```
