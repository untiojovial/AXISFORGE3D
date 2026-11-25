import numpy as np
import pytest
from research_grasping.phase3.contact_similarity import evaluate


def test_constant_uses_train_labels_only():
    row = {'geometry':[0., 1.], 'global_embedding':[1.,0.], 'local':[1.,0.]}
    train = [row, row]
    target = [row]
    train_labels = [{'details':{'epsilon':.2},'status':'ok'},
                    {'details':{'epsilon':.6},'status':'physical_failure'}]
    first = evaluate(train, target, train_labels, [{'details':{'epsilon':.4}, 'status':'ok'}])
    second = evaluate(train, target, train_labels, [{'details':{'epsilon':.9}, 'status':'physical_failure'}])
    assert first['train_constant']['epsilon_predictions'] == pytest.approx([.4])
    assert second['train_constant']['epsilon_predictions'] == pytest.approx([.4])
    assert second['train_constant']['f1_probabilities'] == [.5]
    assert first['train_constant']['epsilon_mae'] == pytest.approx(0.)
