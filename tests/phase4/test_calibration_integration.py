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


@unittest.skipUnless(MANIFEST.exists() and CATS.exists() and HISTORY.exists() and (ROOT/'research/phase4/methodology/calibration_results.json').exists(), 'development calibration assets unavailable')
class RecalibrationIntegrationTests(unittest.TestCase):
    def test_cli_real_recalibration_and_tampering(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch unavailable')
        from research_grasping.phase3.experiment import digest
        outer_path=ROOT/'research/phase4/methodology/calibration_results.json'
        original=json.loads(outer_path.read_text())
        base=[sys.executable,'-m','research_grasping.phase4.calibration_recalibrate','--manifest',str(MANIFEST),
              '--catalogs',str(CATS),'--history',str(HISTORY)]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'out.json'
            completed=subprocess.run(base+['--outer',str(outer_path),'--out',str(out)],cwd=ROOT,env=env,
                                     check=True,capture_output=True,text=True,timeout=120)
            result=json.loads(out.read_text())
            self.assertEqual(result['outer_sha256'],digest(outer_path))
            self.assertEqual(len(result['folds']),16)
            self.assertEqual(len({f['object_id'] for f in result['folds']}),16)
            for fold in result['folds']:
                self.assertEqual([len(g) for g in fold['inner_groups']],[5,5,5])
                self.assertNotIn(fold['object_id'],sum(fold['inner_groups'],[]))
                self.assertTrue(1<=fold['factor']<=100)
                self.assertEqual(fold['raw']['mae'],fold['adjusted']['mae'])
                self.assertGreaterEqual(fold['adjusted']['width90'],fold['raw']['width90'])
            variants=[]
            bad=json.loads(json.dumps(original));bad['manifest_sha256']='0'*64;variants.append((bad,'manifest mismatch'))
            bad=json.loads(json.dumps(original));bad['sources'][next(iter(bad['sources']))]['n']+=1;variants.append((bad,'history/catalogue mismatch'))
            bad=json.loads(json.dumps(original));bad['folds'][0]=bad['folds'][1];variants.append((bad,'outer folds'))
            bad=json.loads(json.dumps(original));bad['folds'][0]['candidate_ids'][0]='not-the-same';variants.append((bad,'outer predictions'))
            bad=json.loads(json.dumps(original));bad['folds'][0]['y'][0]+=1;variants.append((bad,'outer predictions'))
            for i,(data,message) in enumerate(variants):
                path=root/f'bad{i}.json';path.write_text(json.dumps(data));dest=root/f'bad{i}-out.json'
                proc=subprocess.run(base+['--outer',str(path),'--out',str(dest)],cwd=ROOT,env=env,
                                    capture_output=True,text=True,timeout=60)
                self.assertNotEqual(proc.returncode,0)
                self.assertIn(message,proc.stderr)
                self.assertFalse(dest.exists())


@unittest.skipUnless(PHYSICAL.exists() and PROTOCOL.exists() and CATS.exists(), 'complete physics unavailable')
class FidelityIntegrityTests(unittest.TestCase):
    def test_incomplete_duplicate_and_wrong_candidate_are_rejected(self):
        import copy
        from research_grasping.phase4.fidelity_analysis import analyze
        result=json.loads(PHYSICAL.read_text());protocol=json.loads(PROTOCOL.read_text())
        bad=copy.deepcopy(result);bad['rows'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete'):analyze(bad,protocol,CATS)
        bad=copy.deepcopy(result);bad['rows'][-1]=copy.deepcopy(bad['rows'][0])
        with self.assertRaisesRegex(ValueError,'duplicate or unregistered'):analyze(bad,protocol,CATS)
        bad=copy.deepcopy(result);bad['rows'][0]['f1_status']='invalid'
        with self.assertRaisesRegex(ValueError,'unknown F1 status'):analyze(bad,protocol,CATS)
        bad=copy.deepcopy(result);bad['rows'][0]['candidate_id']='invalid'
        with self.assertRaisesRegex(ValueError,'duplicate or unregistered'):analyze(bad,protocol,CATS)
        bad=copy.deepcopy(protocol);bad['objects'][0]['catalog_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'catalog hash mismatch'):analyze(result,bad,CATS)
