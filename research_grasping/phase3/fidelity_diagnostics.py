"""Descriptive F2/F3 paired diagnostics; no automatic B15 promotion."""
import numpy as np
from scipy.stats import spearmanr


def summarize_pairs(rows, *, f2_threshold, f3_threshold):
    """Rows: object_id, f2, f3, status; thresholds fixed before inspection.

    A single global correlation is descriptive; per-object ranks expose grouping.
    Missing/failed infrastructure rows remain counted and are not mapped to zero.
    """
    if not np.isfinite([f2_threshold, f3_threshold]).all():
        raise ValueError('finite predeclared thresholds required')
    valid = []
    counts = {'ok': 0, 'physical_failure': 0, 'infrastructure_failure': 0, 'missing': 0}
    physical_failure_unpaired = 0
    for row in rows:
        status = row.get('status')
        if not row.get('object_id') or status not in counts:
            raise ValueError('invalid paired record')
        counts[status] += 1
        if status in {'ok', 'physical_failure'}:
            pair = [row.get('f2'), row.get('f3')]
            if any(v is None for v in pair):
                if status == 'ok':
                    raise ValueError('successful pairs require finite values')
                physical_failure_unpaired += 1
                continue
            if not np.isfinite(pair).all():
                raise ValueError('paired values must be finite')
            valid.append((row['object_id'], float(pair[0]), float(pair[1])))
    def corr(items):
        if len(items) < 3:
            return None
        x = np.array([v[1] for v in items]); y = np.array([v[2] for v in items])
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            return None
        return float(spearmanr(x, y).statistic)
    tp = fp = tn = fn = 0
    for _, x, y in valid:
        predicted, actual = x >= f2_threshold, y >= f3_threshold
        tp += int(predicted and actual)
        fp += int(predicted and not actual)
        tn += int(not predicted and not actual)
        fn += int(not predicted and actual)
    objects = sorted({v[0] for v in valid})
    return {'status_counts': counts, 'physical_failure_unpaired': physical_failure_unpaired, 'n_pairs': len(valid), 'n_objects': len(objects),
            'pooled_spearman_descriptive': corr(valid),
            'per_object_spearman': {obj: corr([v for v in valid if v[0] == obj]) for obj in objects},
            'confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'false_positive_rate': fp/(fp+tn) if fp+tn else None,
            'false_negative_rate': fn/(fn+tp) if fn+tp else None,
            'f2_threshold': float(f2_threshold), 'f3_threshold': float(f3_threshold),
            'b15_status': 'PENDING'}
