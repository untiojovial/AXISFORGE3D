"""Figures for the manuscript, generated only from the stored result files (no manual edits).

    python tools/research/phase7/make_figures.py --results research/phase7/results \
        --out publication/intelligent_service_robotics/figures [--confirmation-dir conf/]

Writes PDF and SVG. Development figures need only research/phase7/results; confirmation figures need
confirmation_results.json and, for the success-at-budget curves, the run's orders.json.
Colours: the validated reference categorical palette (light), one fixed hue per method, plus a distinct
marker per method and direct labels, so identity never rests on colour alone.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHOD = {'M0': ('#f47230', 'o', 'M0 original'), 'M1': ('#00b894', 's', 'M1 learned P(F)'),
          'M4': ('#f2a600', '^', 'M4 P(F)+centring'), 'M9': ('#ee7fac', 'D', 'M9 P(F)·P(H|F)'),
          'Joint': ('#257de0', 'P', 'FORTIS-R Joint')}
INK, INK2, GRID = '#191919', '#666666', '#e6e6e6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'legend.frameon': False,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, out, name):
    for ext in ('pdf', 'svg'):
        fig.savefig(Path(out)/f'{name}.{ext}')
    plt.close(fig)
    print('wrote', name)


def fig_torsion(res, out):
    fs = json.loads((Path(res)/'friction_sensitivity.json').read_text())
    tors = [float(t) for t in fs['torsions_m']]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True)
    for ax, key, title in zip(axes, ('robust', 'load'), ('Robust (six standard conditions)', 'Load tolerant (0.5, 1, 2 kg)')):
        for m, (c, mk, lab) in METHOD.items():
            y = [fs['by_torsion'][str(t)]['rates'][m][key] for t in fs['torsions_m']]
            ax.plot(tors, y, color=c, marker=mk, ms=4.5, lw=1.6, label=lab)
        ax.set_xscale('log'); ax.set_xticks(tors); ax.set_xticklabels([f'{t:g}' for t in tors])
        ax.set_xlabel('Torsional friction coefficient (m)'); ax.set_title(title, fontsize=8, color=INK, loc='left')
        ax.axvline(.005, color=INK2, lw=.8, ls=':'); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Share of first grasps (v3)')
    axes[1].legend(loc='lower right', fontsize=6.5, ncol=1)
    fig.text(.01, -.02, f"Development, {fs['development_parts']} GSO parts; dotted line = value used in Phases VI-VII",
             fontsize=7, color=INK2, ha='left', va='top')
    save(fig, out, 'fig_dev_torsion')


def forest(ax, rows, xlabel, ref=1., log=True):
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color='#257de0', lw=1.6, solid_capstyle='round')
        ax.plot(v, y, 'o', color='#257de0', ms=5, mec='white', mew=1)
        ax.annotate(f'{v:.3f}', (hi, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=6.5, color=INK)
    ax.axvline(ref, color=INK2, lw=.8)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale('log')
        lo_, hi_ = min(r[2] for r in rows), max(r[3] for r in rows)
        ticks = [t for t in (.1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.25, 1.3, 1.5, 2.) if lo_*.95 <= t <= hi_*1.05 or t == ref]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis='y', visible=False)


def fig_representation(res, out):
    v = json.loads((Path(res)/'visual_ablation.json').read_text())
    m = json.loads((Path(res)/'mlp_benchmark.json').read_text())
    lo_ = json.loads((Path(res)/'local_features_benchmark.json').read_text())
    pick = {('L7', 'L2'): 'L7 DINOv2 vs L2', ('L8', 'L2'): 'L8 DINOv3 vs L2', ('L11', 'L2'): 'L11 OpenCLIP vs L2',
            ('L10', 'L2'): 'L10 DINOv3 linear vs L2', ('L10', 'L10p'): 'L10 vs permuted (L10p)',
            ('L10', 'L10r'): 'L10 vs untrained ViT (L10r)', ('L8', 'L8p'): 'L8 vs permuted (L8p)',
            ('L2', 'L0'): 'L2 MLP vs gradient boosting'}
    allc = {(c['a'], c['b']): c for src in (v['contrasts']['N_F'], m['contrasts']['N_F'], lo_['contrasts']['N_F']) for c in src}
    rows = [(lab, allc[k]['ratio'], *allc[k]['ci95']) for k, lab in pick.items() if k in allc]
    for k, lab in ((('LG', 'L2'), 'LG local geometry vs L2'), (('LD', 'L2'), 'LD local DINOv3 vs L2')):
        if k in allc:
            rows.append((lab, allc[k]['ratio'], *allc[k]['ci95']))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    forest(ax, rows, 'N_F ratio (geometric mean, 95% bootstrap CI); < 1 = fewer candidates')
    ax.set_title(f"Development, {m['parts']} parts, out-of-fold; CIs unadjusted", fontsize=7, color=INK2, loc='left')
    save(fig, out, 'fig_dev_representation')


def fig_articulated(res, out):
    a = json.loads((Path(res)/'articulated_simulator_validation.json').read_text())
    conds = list(a['agreement_by_condition'])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=.75))
    x = np.arange(len(conds))
    ag = [a['agreement_by_condition'][c]['kappa'] for c in conds]
    axes[0].bar(x, ag, width=.6, color='#257de0', edgecolor='white', linewidth=2)
    for xi, k in zip(x, ag):
        axes[0].annotate(f'{k:.2f}', (xi, k), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=6.5, color=INK)
    axes[0].set_xticks(x); axes[0].set_xticklabels(conds, rotation=35, ha='right'); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Cohen's κ, articulated vs floating"); axes[0].grid(axis='x', visible=False)
    axes[0].set_title(f"{a['agreement_by_condition']['nominal']['grasps']} first grasps, 151 parts", fontsize=7, color=INK2, loc='left')
    keys = [('Joint - M0', 'robust4'), ('Joint - M1', 'robust4'), ('Joint - M0', 'load'), ('Joint - M1', 'load')]
    labels = ['Joint−M0 robust', 'Joint−M1 robust', 'Joint−M0 load', 'Joint−M1 load']
    for j, (lvl, col, mk) in enumerate((('v3 floating gripper', '#f47230', 'o'), ('articulated', '#257de0', 's'))):
        for i, (k, w) in enumerate(keys):
            c = a['contrasts'][lvl][f'{k}, {w}']
            y = len(keys)-1-i + (.15 if j == 0 else -.15)
            axes[1].plot(c['ci95'], [y, y], color=col, lw=1.6)
            axes[1].plot(c['difference'], y, mk, color=col, ms=4.5, mec='white', mew=.8, label=lvl if i == 0 else None)
    axes[1].axvline(0, color=INK2, lw=.8); axes[1].set_yticks(range(len(keys))[::-1]); axes[1].set_yticklabels(labels)
    axes[1].set_xlabel('Paired difference (95% CI)'); axes[1].grid(axis='y', visible=False)
    axes[1].legend(fontsize=6.5, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2)
    save(fig, out, 'fig_dev_articulated')


def fig_confirmation(res, out, conf_dir):
    c = json.loads((Path(res)/'confirmation_results.json').read_text())
    S = c['sets']['primary']
    prim = S['primary']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
    rows = [(k.split(':')[0], v['ratio'], *v['ci95']) for k, v in prim.items() if 'ratio' in v]
    forest(axes[0], rows, 'N_F ratio (95% CI)')
    axes[0].axvline(1.25, color=INK2, lw=.8, ls='--')
    p2 = next(v for k, v in prim.items() if k.startswith('P2'))
    axes[1].plot(p2['ci95'], [0, 0], color='#257de0', lw=1.6); axes[1].plot(p2['difference'], 0, 'o', color='#257de0', ms=5, mec='white')
    axes[1].axvline(0, color=INK2, lw=.8); axes[1].axvline(-.05, color=INK2, lw=.8, ls='--')
    axes[1].set_yticks([0]); axes[1].set_yticklabels(['P2 robust, Joint−M0']); axes[1].set_xlabel('Paired difference (95% CI); dashed = margin')
    axes[1].grid(axis='y', visible=False)
    fig.suptitle(f"EGAD primary set: {S['viable']} viable of {S['objects']} objects", fontsize=7, color=INK2, x=.01, ha='left')
    save(fig, out, 'fig_conf_primary')
    # success at budget (population, all objects)
    pop = S['population_success_at_budget']
    B = [int(k.split('@')[1]) for k in pop['M0']]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in ('M0', 'M1', 'Joint'):
        col, mk, lab = METHOD[m]
        y = [pop[m][f'success@{b}'] for b in B]
        ax.plot(B, y, color=col, marker=mk, ms=4.5, lw=1.6, label=lab)
        ax.annotate(m, (B[-1], y[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7, color=INK)
    ax.set_xscale('log'); ax.set_xticks(B); ax.set_xticklabels(B); ax.set_ylim(0, 1)
    ax.set_xlabel('Candidate budget B'); ax.set_ylabel('Share of all objects with a grasp within B')
    ax.legend(fontsize=6.5, loc='lower right')
    ax.set_title(f"EGAD primary, all {S['objects']} objects", fontsize=7, color=INK2, loc='left')
    save(fig, out, 'fig_conf_success_at_budget')
    # rates per method
    r = S['rates_on_viable']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for ax, key, title in zip(axes, ('robust_v3', 'load_tolerant_v3'), ('Robust (v3)', 'Load tolerant (v3)')):
        ms = list(METHOD)
        for i, m in enumerate(ms):
            col, mk, lab = METHOD[m]
            ax.bar(i, r[m][key], width=.6, color=col, edgecolor='white', linewidth=2)
            ax.annotate(f"{r[m][key]:.2f}", (i, r[m][key]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=6.5, color=INK)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylim(0, 1); ax.grid(axis='x', visible=False)
        ax.set_title(title, fontsize=8, color=INK, loc='left')
    axes[0].set_ylabel(f"Share of {S['viable_with_dynamics']} viable objects")
    save(fig, out, 'fig_conf_rates')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', required=True); p.add_argument('--out', required=True); p.add_argument('--confirmation-dir')
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig_torsion(a.results, a.out)
    fig_representation(a.results, a.out)
    fig_articulated(a.results, a.out)
    if (Path(a.results)/'confirmation_results.json').exists():
        fig_confirmation(a.results, a.out, a.confirmation_dir)


if __name__ == '__main__':
    main()
