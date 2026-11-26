"""Machine-readable register of every external component (datasets, models, weights, tools, assets).

    python tools/licenses/build_manifest.py --out research/phase7_extension/third_party_manifest.json

The component records below are maintained by hand from the evidence named in each record; the script adds
the SHA-256 of the affected committed files and of the licence texts kept in LICENSES/, so that any later change
to them is visible. Verification date of every record: 2026-11-26 unless stated otherwise.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DATE = '2026-11-26'

COMPONENTS = [
    dict(name='AxisForge3D source code, documentation and own results', version='repository', source='this repository',
         license='Apache-2.0', evidence='LICENSE (official text from https://www.apache.org/licenses/LICENSE-2.0.txt); author authorisation of 2026-11-26',
         redistribution='yes', attribution='copyright holder: the AxisForge3D author (name pending)',
         affected=['LICENSE', 'Cargo.toml', 'pyproject.toml'], status='compliant',
         notes='Apache-2.0 covers only material the author has rights to; the third-party items below keep their own terms.'),
    dict(name='YCB Object and Model Set (16 objects of the GRASPA layouts)', version='YCB-Video models as packaged by pybullet-object-models @ 57cc4007607f6bcbb0a5364ccab7636faf7a91e4',
         source='https://github.com/eleramp/pybullet-object-models (ycb_objects/Ycb*/); upstream YCB: https://ycb-benchmarks.s3.amazonaws.com/index.html',
         license='YCB dataset: CC BY 4.0 (YCB code: MIT). Packaging repository: LGPL-2.1 (its LICENSE file). YCB-Video model terms: not verified',
         evidence=('YCB S3 index page ("License for the data set: Creative Commons Attribution 4.0 International (CC BY 4.0)", '
                   '"License for the code: MIT license") and AWS Open Data Registry entry (license: CC BY 4.0), both read on 2026-11-26; '
                   'pybullet-object-models LICENSE (LGPL-2.1) and README ("The YCB Video dataset (posecnn, YCB_Video_toolbox)") read on 2026-11-26. '
                   'Provenance check: the 16 meshes downloaded from the pinned commit reproduce the Phase IV objects exactly '
                   '(identical world bounds, 0.0 mm, and identical candidate counts); the project-file hashes differ only by the project format.'),
         redistribution='meshes and renders: not redistributed; derived numeric data (bounds, grasp candidates, descriptors, statistics): included with attribution',
         attribution='Calli et al. 2015 (YCB Object and Model Set); YCB-Video models of the PoseCNN project (https://rse-lab.cs.washington.edu/projects/posecnn/); pybullet-object-models (https://github.com/eleramp/pybullet-object-models); modifications: placed at (0.42, 0, 0.42) m in the demo cell, mass from the package URDF, no geometric change',
         affected=['tools/verify/ycb_pick.sh', 'research/phase4/grasp_prior/library.json', 'research/phase4/grasp_prior/dinov2.npz',
                   'research/phase4/grasp_prior/dinov3.npz', 'research/phase4/grasp_prior/openclip.npz', 'research/phase4/grasp_prior/vit-random.npz',
                   'research/phase5/results/descriptors.npz'],
         status='resolved: licence and exact provenance verified; meshes not redistributed. Residual: whether LGPL-2.1 terms of the packaging repository attach to numeric data derived from its meshes (author decision D-YCB)'),
    dict(name='Google Scanned Objects (GSO)', version='MuJoCo adaptation kevinzakka/mujoco_scanned_objects @ 6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c',
         source='https://github.com/kevinzakka/mujoco_scanned_objects; upstream https://arxiv.org/abs/2204.11918',
         license='OBJ and PNG: CC BY 4.0; MJCF XML: MIT (as stated in the adaptation README)', evidence='research/phase3/dataset/README.md (Phase III verification of the pinned README)',
         redistribution='meshes not redistributed; renders of 22 objects (research/phase4/embedding_experiment/development-images.zip) are adaptations under CC BY 4.0; numeric data included with attribution',
         attribution='Downs et al. 2022 (Google Scanned Objects); MuJoCo adaptation by K. Zakka; modifications: grey renders, CoACD/V-HACD collision pieces, placement in the demo cell',
         affected=['research/phase4/embedding_experiment/development-images.zip', 'research/phase5/results/descriptors.npz', 'research/phase5/results/parts.json'],
         status='compliant with attribution (CC BY 4.0 notice in THIRD_PARTY_NOTICES.md and LICENSES/CC-BY-4.0.txt)'),
    dict(name='EGAD (Evolved Grasping Analysis Dataset)', version='training set (2 281 meshes) and evaluation set (49); archives hashed in research/phase7/egad_archives.sha256',
         source='https://github.com/dougsm/egad and the QUT data record', license='BSD-3-Clause', evidence='research/phase7/EGAD_LICENSE.txt (copied to LICENSES/BSD-3-Clause-EGAD.txt)',
         redistribution='meshes not redistributed; names, hashes and derived statistics included', attribution='Morrison, Corke, Leitner 2020; copyright D. Morrison, ACRV/QUT 2020; modifications: rescaled per the authors\' rule, stable pose, CoACD pieces',
         affected=['research/phase7/egad_selection.json', 'research/phase7/results/*', 'research/phase7_extension/results/*'], status='compliant'),
    dict(name='DINOv3 ViT-S/16 (Meta) and its embeddings', version='timm copy used in Phases IV-VII (weights not in the repository)',
         source='https://github.com/facebookresearch/dinov3', license='DINOv3 License (Meta), last updated 2025-08-19 — not Apache-2.0',
         evidence='LICENSES/DINOv3-License.md (fetched from https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md on 2026-11-26)',
         redistribution='weights: no. Embeddings computed with the model: kept in the repository and distributed only under the DINOv3 License with a copy of it (clause 1.b.i), which satisfies the licence whether or not embeddings are derivative works; packaged separately from the Apache-2.0 material',
         attribution='acknowledge the use of DINO Materials in any publication (clause 1.b.ii)',
         affected=['research/phase4/embedding_experiment/dinov3/global_embeddings.npz', 'research/phase4/grasp_prior/dinov3.npz', 'research/phase4/grasp_prior/new_parts/dinov3.npz',
                   'research/phase4/grasp_prior/scale/development/dinov3-metric.npz', 'research/phase4/grasp_prior/scale/test/dinov3.npz',
                   'research/phase4/grasp_prior/scale/test/dinov3-metric.npz', 'research/phase5/results/descriptors.npz',
                   'research/phase4/grasp_prior/new_parts/library.json.gz', 'research/phase4/grasp_prior/scale/test/test-library.json.gz',
                   'research/phase5/near_duplicates.json'],
         status='compliant under the stricter reading (licence copy + DINOv3-License labelling); not relicensed'),
    dict(name='DINOv2 ViT-S/14 (Meta)', version='facebook/dinov2-small (revision recorded in the descriptor provenance)', source='https://huggingface.co/facebook/dinov2-small',
         license='Apache-2.0 (model card)', evidence='research/phase7/PHASE7_REPOSITORY_AUDIT.md §5', redistribution='weights: no; embeddings: yes, with attribution',
         attribution='Oquab et al. 2023 (DINOv2)', affected=['research/phase4/grasp_prior/dinov2.npz', 'research/phase5/results/descriptors.npz'], status='compliant'),
    dict(name='OpenCLIP ViT-B-32-quickgelu, LAION-400M epoch 31', version='open_clip_torch 4.2.2; checkpoint vit_b_32-quickgelu-laion400m_e31-d867053b.pt',
         source='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt',
         license='weights: MIT (model card of the registry\'s mirror timm/vit_base_patch32_clip_224.laion400m_e31, "license: mit"); code: MIT. Training data LAION-400M: its own terms, not assessed (not redistributed)',
         evidence=('open_clip src/open_clip/pretrained.py (main, read 2026-11-26) maps laion400m_e31 of ViT-B-32-quickgelu to that URL and to hf_hub timm/vit_base_patch32_clip_224.laion400m_e31; '
                   'the Hugging Face model card README declares license: mit (read 2026-11-26); the checkpoint used has SHA-256 '
                   'd867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e and 605 159 845 bytes (research_grasping/phase3/fetch_openclip.py), whose prefix matches the official file name'),
         redistribution='weights: no (fetch script with pinned URL, size and hash); embeddings: yes', attribution='Ilharco et al. (OpenCLIP); Schuhmann et al. 2021 (LAION-400M)',
         affected=['research_grasping/phase3/fetch_openclip.py', 'research/phase4/grasp_prior/openclip.npz', 'research/phase4/embedding_experiment/openclip_development.zip'],
         status='resolved: checkpoint identified (e31, not e32) and its declared licence verified'),
    dict(name='GPD (Grasp Pose Detection) and its LeNet weights', version='commit 6327f20eabfcba41a05fdd2e2ba408153dc2e958 (2021-07-15)', source='https://github.com/atenpas/gpd',
         license='BSD-2-Clause (repository LICENSE.md, copyright 2018 Andreas ten Pas); the weights ship in the same repository',
         evidence='local clone, LICENSE.md', redistribution='not redistributed; built locally; the adapter in tools/research/phase7_extension/gpd_adapter is own code (Apache-2.0) that links to the unmodified library',
         attribution='ten Pas et al. 2017 (GPD)', affected=['tools/research/phase7_extension/gpd_adapter/*'], status='compliant'),
    dict(name='UR5 model (example-robot-data)', version='robots/ur_description (branch master at copy time)', source='https://github.com/Gepetto/example-robot-data',
         license='BSD-3-Clause', evidence='apps/viewer/samples/ur5/LICENSE, crates/project-io/tests/data/ur5/LICENSE', redistribution='yes, with the licence alongside',
         attribution='LAAS-CNRS, University of Edinburgh, INRIA', affected=['apps/viewer/samples/ur5/*', 'crates/project-io/tests/data/ur5/*'], status='compliant'),
    dict(name='three.js', version='0.150.1', source='https://cdnjs.cloudflare.com/ajax/libs/three.js/0.150.1/three.min.js', license='MIT',
         evidence='loaded from the CDN at run time by apps/viewer (not vendored)', redistribution='not redistributed', attribution='three.js authors',
         affected=['apps/viewer/body.part.html'], status='compliant'),
    dict(name='MuJoCo, CoACD, trimesh, PyTorch, scikit-learn and other Python packages; Rust crates', version='see research/phase7_extension/results/dependency_licences.json',
         source='PyPI / crates.io', license='permissive (MIT, BSD, Apache-2.0, PSF, MIT-CMU); no copyleft dependency found',
         evidence='tools/licenses/audit_dependencies.py output', redistribution='not redistributed (installed by pip and cargo)', attribution='THIRD_PARTY_NOTICES.md',
         affected=['Cargo.lock', 'pyproject.toml'], status='compliant'),
    dict(name='PCL, Eigen, OpenCV (system packages used to build GPD)', version='Ubuntu packages of the build machine', source='apt',
         license='PCL BSD-3-Clause; Eigen MPL-2.0; OpenCV Apache-2.0 (as declared by the projects)', evidence='not verified against the package copyright files in this pass',
         redistribution='not redistributed', attribution='—', affected=[], status='not applicable to the repository (build-time only)'),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = []
    for c in COMPONENTS:
        files = {}
        for pat in c['affected']:
            for f in sorted(ROOT.glob(pat)):
                if f.is_file():
                    files[str(f.relative_to(ROOT))] = sha(f)
        out.append(dict(c, verified=DATE, affected_sha256=files))
    lic = {str(f.relative_to(ROOT)): sha(f) for f in sorted((ROOT/'LICENSES').glob('*'))}
    lic['LICENSE'] = sha(ROOT/'LICENSE')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    Path(a.out).write_text(json.dumps(dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), repository_commit_when_built=commit,
                                           licence_texts_sha256=lic, components=out), indent=1) + '\n')
    print(len(out), 'components;', sum(len(c['affected_sha256']) for c in out), 'files hashed')


if __name__ == '__main__':
    main()
