"""Integration checks using the existing development-only artifacts, when present."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/workspace/axisforge-phase3-data')
MANIFEST = ROOT/'research/phase3/pilot/manifest.prepilot.json'
CATS = DATA/'pilot-expanded/catalogs'
HISTORY = DATA/'run-development'
PHYSICAL = ROOT/'research/phase4/robotics/physical_pilot.json'
PROTOCOL = ROOT/'research/phase4/robotics/physical_protocol.json'


@unittest.skipUnless(MANIFEST.exists() and CATS.exists() and HISTORY.exists(), 'development assets unavailable')
class RealDataIntegrationTests(unittest.TestCase):
    def test_calibration_cli_real_train_folds(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch unavailable')
        from research_grasping.phase3.experiment import digest
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'calibration.json'
            subprocess.run([sys.executable,'-m','research_grasping.phase4.calibration','--manifest',str(MANIFEST),
                            '--catalogs',str(CATS),'--history',str(HISTORY),'--out',str(out)],
                           cwd=ROOT,env=env,check=True,capture_output=True,text=True,timeout=180)
            result=json.loads(out.read_text());manifest=json.loads(MANIFEST.read_text())
            names={r['object_id'] for r in manifest['objects'] if r['split']=='train'}
            self.assertEqual({f['object_id'] for f in result['folds']},names)
            self.assertEqual(result['manifest_sha256'],digest(MANIFEST))
            self.assertEqual(len(result['folds']),16)
            for fold in result['folds']:
                self.assertEqual(len(fold['y']),len(fold['candidate_ids']))
                for model in fold['models'].values():
                    self.assertEqual(model['n'],len(fold['y']))
                    self.assertEqual(len(model['prediction_mean']),model['n'])
                    self.assertTrue(all(v>=0 for v in model['prediction_variance']))
                    self.assertTrue(0<=model['coverage90']<=1)

    @unittest.skipUnless(PHYSICAL.exists() and PROTOCOL.exists(), 'complete physics unavailable')
    def test_fidelity_cli_real_catalogs_and_hashes(self):
        from research_grasping.phase3.experiment import digest
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fidelity.json'
            subprocess.run([sys.executable,'-m','research_grasping.phase4.fidelity_analysis','--result',str(PHYSICAL),
                            '--protocol',str(PROTOCOL),'--catalogs',str(CATS),'--out',str(out)],
                           cwd=ROOT,check=True,capture_output=True,text=True,timeout=90)
            result=json.loads(out.read_text())
            self.assertEqual(result['input_sha256'],digest(PHYSICAL))
            self.assertEqual(result['protocol_sha256'],digest(PROTOCOL))
            self.assertEqual(len(result['rows']),84)
            self.assertEqual(sum(x['n'] for x in result['by_object'].values()),84)
            self.assertEqual(result['aggregate']['n'],84)
            self.assertEqual(result['aggregate']['nominal_completed']+result['aggregate']['unknown'],84)


if __name__=='__main__':unittest.main()
