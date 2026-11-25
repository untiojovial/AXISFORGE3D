"""Exercise placement study on one real train object; do not touch final test."""
from pathlib import Path
import pytest
from research_grasping.phase4 import placement_study
from research_grasping.phase3.experiment import DevelopmentData, load

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/workspace/axisforge-phase3-data')
CATALOGUES=Path('/workspace/axisforge-phase4-data/catalogues')
BINARY=ROOT/'target/release/axisforge'
NAME='2_of_Jenga_Classic_Game'
pytestmark=pytest.mark.skipif(not (DATA/'manifest.json').exists() or not (CATALOGUES/'diverse128'/f'{NAME}.json').exists() or not BINARY.exists(), reason='prepared development assets unavailable')


@pytest.mark.skipif(not (DATA/'manifest.json').exists() or not (CATALOGUES/'diverse128'/f'{NAME}.json').exists()
                    or not BINARY.exists(), reason='prepared development assets unavailable')
def test_placement_run_real_train_object(tmp_path,monkeypatch):
    class OneObjectGuard(DevelopmentData):
        def __init__(self,*args):
            super().__init__(*args)
            self.objects={NAME:self.objects[NAME]}
    monkeypatch.setattr(placement_study,'DevelopmentData',OneObjectGuard)
    work=tmp_path/'work'
    work.mkdir()
    (work/'catalogues').symlink_to(CATALOGUES, target_is_directory=True)
    output=tmp_path/'placement.json'
    placement_study.run(DATA,BINARY,work,output)
    result=load(output)
    assert len(result['objects'])==1
    obj=result['objects'][0]
    assert obj['object_id']==NAME and obj['table_supported']
    assert len(obj['ids'])==10 and len(obj['measurements'])==10
    assert any(row['status']=='paired' for row in obj['measurements'])
    for row in obj['measurements']:
        if row['status']=='paired':
            assert row['before']['id'] in obj['ids']
            assert abs(row['before']['epsilon']-row['after']['epsilon']) <= 1e-6
            assert row['f2_agrees']
        else:
            assert row['status']=='unmatched' and row['alternatives']>=0


def test_unsupported_table_position_short_circuits_measurement(tmp_path,monkeypatch):
    class OneObjectGuard(DevelopmentData):
        def __init__(self,*args):
            super().__init__(*args)
            self.objects={NAME:self.objects[NAME]}
    monkeypatch.setattr(placement_study,'DevelopmentData',OneObjectGuard)
    monkeypatch.setattr(placement_study,'supported',lambda *args:False)
    def unexpected(*args,**kwargs):raise AssertionError('unsupported scene cannot be evaluated')
    monkeypatch.setattr(placement_study,'load_catalog',unexpected)
    work=tmp_path/'work';work.mkdir();(work/'catalogues').symlink_to(CATALOGUES,target_is_directory=True)
    output=tmp_path/'unsupported.json'
    placement_study.run(DATA,BINARY,work,output)
    obj=load(output)['objects'][0]
    assert not obj['table_supported'] and obj['measurements']==[]


def test_ambiguous_pose_does_not_receive_an_arbitrary_label(tmp_path,monkeypatch):
    from copy import deepcopy
    class OneObjectGuard(DevelopmentData):
        def __init__(self,*args):
            super().__init__(*args)
            self.objects={NAME:self.objects[NAME]}
        def catalogue(self,*args):
            cat=super().catalogue(*args)
            cat=deepcopy(cat);cat['ids']=['0']
            return cat
    monkeypatch.setattr(placement_study,'DevelopmentData',OneObjectGuard)
    expected=load(ROOT/'research/phase4/candidates/selection.json')['objects'][NAME]['catalogues']['diverse128']['catalogue_digest']
    monkeypatch.setattr(placement_study,'fingerprint',lambda _:expected)
    v=lambda x,y,z:{'x':x,'y':y,'z':z}
    before={'candidate':{'center':v(.4,0,.5),'approach':v(0,0,-1),'closing':v(1,0,0),
                         'width':.04,'contacts':[{'point':v(.42,0,.5),'normal':v(1,0,0)}]}}
    shifted=deepcopy(before)
    shifted['candidate']['center']['x']-=.1
    shifted['candidate']['contacts'][0]['point']['x']-=.1
    wrong=deepcopy(shifted);wrong['candidate']['center']['x']+=.04
    def fake_catalog(binary,project,body):
        return {'records':[before] if 'placement' not in str(project) else [wrong,shifted,deepcopy(shifted)]}
    monkeypatch.setattr(placement_study,'load_catalog',fake_catalog)
    def unexpected(*args,**kwargs):raise AssertionError('ambiguous candidate must not be evaluated')
    monkeypatch.setattr(placement_study,'evaluate_candidate',unexpected)
    work=tmp_path/'work';work.mkdir();(work/'catalogues').symlink_to(CATALOGUES,target_is_directory=True)
    output=tmp_path/'ambiguous.json'
    placement_study.run(DATA,BINARY,work,output)
    assert load(output)['objects'][0]['measurements']==[{'id':0,'status':'unmatched','alternatives':2}]
