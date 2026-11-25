"""Summarize completed development F1/F2 traces without accessing hidden labels."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t, spearmanr


def _mean_ci(values):
    a = np.asarray(values, float)
    if not len(a):
        return None
    mean = float(a.mean())
    if len(a) < 2:
        return {'n': len(a), 'mean': mean, 'low': None, 'high': None}
    half = float(t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {'n': len(a), 'mean': mean, 'low': mean-half, 'high': mean+half}


def summarize_run(run):
    definition, rows = run['definition'], run['rows']
    if run.get('pending') is not None or len(rows) != definition['budget']:
        return None
    if len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate candidate in run')
    statuses = {'ok': 0, 'physical_failure': 0, 'infrastructure_failure': 0}
    best, curve, epsilon_all, epsilon_feasible = None, [], [], []
    nominal, fresh = 0., 0.
    cache_hits, cache_unknown = 0, 0
    for r in rows:
        status = r['status']
        if status not in statuses or not np.isfinite(r['cost']) or r['cost'] < 0:
            raise ValueError('invalid status or cost')
        statuses[status] += 1
        details = r.get('details', {})
        nominal += r['cost']
        if details.get('cached') is False:
            fresh += r['cost']
        elif details.get('cached') is True:
            cache_hits += 1
        else:
            cache_unknown += 1
        if status != 'infrastructure_failure':
            value = r['utility']
            if value is None or not np.isfinite(value):
                raise ValueError('missing measured utility')
            best = value if best is None else max(best, value)
            epsilon = details.get('epsilon')
            if epsilon is not None:
                if not np.isfinite(epsilon):
                    raise ValueError('invalid epsilon')
                epsilon_all.append(float(epsilon))
                if status == 'ok':
                    epsilon_feasible.append(float(epsilon))
        curve.append(best)
    return {'object_id': definition['object_id'], 'method': definition['method'],
            'seed': definition['seed'], 'budget': definition['budget'], 'statuses': statuses,
            'curve': curve, 'auc': float(np.mean(curve)) if all(x is not None for x in curve) else None,
            'last': curve[-1], 'epsilon_all': epsilon_all, 'epsilon_feasible': epsilon_feasible,
            'nominal_online_s': nominal, 'fresh_online_s': fresh, 'cache_hits': cache_hits,
            'cache_unknown': cache_unknown}


def summarize_logs(runs, history):
    complete = [v for r in runs if (v := summarize_run(r)) is not None]
    groups = {}
    for run in complete:
        key = (run['object_id'], run['method'])
        groups.setdefault(key, []).append(run)
    table = []
    for (object_id, method), rows in sorted(groups.items()):
        budget = rows[0]['budget']
        if any(r['budget'] != budget for r in rows):
            raise ValueError('mixed budgets for same object and method')
        status = {k: sum(r['statuses'][k] for r in rows) for k in rows[0]['statuses']}
        n = sum(status.values())
        f1_total = status['ok'] + status['physical_failure']
        curve = [_mean_ci([r['curve'][i] for r in rows if r['curve'][i] is not None]) for i in range(budget)]
        table.append({'object_id': object_id, 'method': method, 'n_seeds': len(rows),
                      'auc': _mean_ci([r['auc'] for r in rows if r['auc'] is not None]),
                      'last': _mean_ci([r['last'] for r in rows if r['last'] is not None]),
                      'curve': curve, 'status_counts': status,
                      'f1_rejection_pct': 100*status['physical_failure']/f1_total if f1_total else None,
                      'infrastructure_error_pct': 100*status['infrastructure_failure']/n if n else None,
                      'epsilon_unconditional': _mean_ci([v for r in rows for v in r['epsilon_all']]),
                      'epsilon_feasible_only': _mean_ci([v for r in rows for v in r['epsilon_feasible']]),
                      'nominal_online_s': _mean_ci([r['nominal_online_s'] for r in rows]),
                      'fresh_online_s': _mean_ci([r['fresh_online_s'] for r in rows]),
                      'cache_hits': sum(r['cache_hits'] for r in rows),
                      'cache_unknown': sum(r['cache_unknown'] for r in rows)})
    source_rows = [row for h in history for row in h['rows']]
    return {'completed_runs': len(complete), 'incomplete_runs_excluded': len(runs)-len(complete),
            'objects': len({r['object_id'] for r in complete}),
            'historical': {'objects':len(history), 'attempts':len(source_rows),
              'nominal_s':sum(r['cost'] for r in source_rows),
              'fresh_s':sum(r['cost'] for r in source_rows if r.get('details',{}).get('cached') is False),
              'cache_hits':sum(r.get('details',{}).get('cached') is True for r in source_rows),
              'cache_unknown':sum('cached' not in r.get('details',{}) for r in source_rows)},
            'groups': table}


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    objects = sorted({g['object_id'] for g in summary['groups']})
    if not objects:
        return
    fig, axes = plt.subplots(len(objects), 1, figsize=(8, max(3, 2.8*len(objects))), squeeze=False)
    for ax, obj in zip(axes[:,0], objects):
        for group in summary['groups']:
            if group['object_id'] != obj:
                continue
            x = np.arange(1, len(group['curve'])+1)
            mean = [v['mean'] if v else np.nan for v in group['curve']]
            line, = ax.plot(x, mean, label=group['method'])
            lo = [v['low'] if v and v['low'] is not None else np.nan for v in group['curve']]
            hi = [v['high'] if v and v['high'] is not None else np.nan for v in group['curve']]
            ax.fill_between(x, lo, hi, alpha=.12, color=line.get_color())
        ax.set_title(obj, fontsize=9)
        ax.set_xlabel('F1/F2 attempts (including initialization)')
        ax.set_ylabel('Observed best score')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    for ext in ('svg','png'):
        fig.savefig(output / ('best_so_far.'+ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_paths = sorted((args.run_dir/'runs').glob('*/*.json'))
    history_paths = sorted((args.run_dir/'history').glob('*.json'))
    runs = [json.loads(p.read_text()) for p in run_paths]
    history = [json.loads(p.read_text()) for p in history_paths]
    summary = summarize_logs(runs, history)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (args.out/'summary.csv').open('w', newline='') as file:
        fields = ['object_id','method','n_seeds','auc_mean','last_mean','f1_rejection_pct','infrastructure_error_pct','epsilon_unconditional_mean','epsilon_feasible_mean','nominal_online_s_mean','fresh_online_s_mean','cache_hits']
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for g in summary['groups']:
            mean = lambda key: g[key]['mean'] if g[key] else None
            writer.writerow(dict(object_id=g['object_id'],method=g['method'],n_seeds=g['n_seeds'],
                auc_mean=mean('auc'),last_mean=mean('last'),f1_rejection_pct=g['f1_rejection_pct'],
                infrastructure_error_pct=g['infrastructure_error_pct'],epsilon_unconditional_mean=mean('epsilon_unconditional'),
                epsilon_feasible_mean=mean('epsilon_feasible_only'),nominal_online_s_mean=mean('nominal_online_s'),
                fresh_online_s_mean=mean('fresh_online_s'),cache_hits=g['cache_hits']))
    plot(summary, args.out)

if __name__ == '__main__':
    main()
