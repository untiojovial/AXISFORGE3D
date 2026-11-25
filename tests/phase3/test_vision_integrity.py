import numpy as np
import pytest
from research_grasping.phase3.vision_integrity import semantic_npz_sha256, verify_visual, verify_texture, file_sha256


def test_hash_excludes_time_but_rejects_changed_tokens_and_texture(tmp_path):
    f=tmp_path/'x.npz'; texture=tmp_path/'texture.png'; texture.write_bytes(b'authentic')
    def write(value,t): np.savez_compressed(f,patches=np.array([value]),elapsed_seconds=t,manifest_sha256='id')
    write(1,1.0); lock={'objects':{'one':{'vision_semantic_sha256':semantic_npz_sha256(f),'source_texture_sha256':file_sha256(texture)}}}
    write(1,99.); verify_visual(f,'one',lock); verify_texture(texture,'one',lock)
    write(2,99.)
    with pytest.raises(ValueError): verify_visual(f,'one',lock)
    texture.write_bytes(b'tampered')
    with pytest.raises(ValueError): verify_texture(texture,'one',lock)
    with pytest.raises(ValueError): verify_visual(f,'unrecognized',lock)


def test_prediction_invariant_to_target_labels():
    from research_grasping.phase3.contact_similarity import evaluate
    a=[dict(geometry=[0.],global_embedding=[1.,0.],local=[1.,0.]),dict(geometry=[1.],global_embedding=[0.,1.],local=[0.,1.])]
    labels=[dict(status='ok',details={'epsilon':x}) for x in (.2,.9)]
    first=evaluate(a,a,labels,labels)
    altered=[dict(status='physical_failure',details={'epsilon':100+x}) for x in (0,1)]
    second=evaluate(a,a,labels,altered)
    for name in first:
        assert first[name]['epsilon_predictions']==second[name]['epsilon_predictions']
        assert first[name]['f1_probabilities']==second[name]['f1_probabilities']
