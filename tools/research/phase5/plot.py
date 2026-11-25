"""Phase V figures: success within a budget, and the protocol's contrasts.

    python tools/research/phase5/plot.py --analysis analysis.json --rows test.json --out phase5.png
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import stats  # noqa: E402

INK, MUTED, GRID, GRAY = '#191919', '#666666', '#e6e6e6', '#9c9c9c'
SERIES = [('learned', 'Clasificador aprendido', '#257de0', 'o'), ('afat', 'AFAT', '#f47230', 'D'),
          ('box', 'Vecinas por tamaño', '#00b894', 's'), ('visual:dinov3', 'Vecinas DINOv3', '#f2a600', '^')]
REFERENCE = [('default', 'Orden por defecto', '--'), ('self', 'Sus propios agarres (techo)', ':')]
SHORT = {'learned': 'aprendido', 'box': 'tamaño', 'afat': 'AFAT', 'afat-novis': 'AFAT sin imagen',
         'visual:dinov3': 'DINOv3', 'visual:vit-random': 'ViT aleatorio', 'default': 'defecto',
         'fusion:dinov3': 'DINOv3 + tamaño', 'pooled': 'todas', 'random-order': 'orden aleatorio',
         'visual:dinov3-metric': 'DINOv3 con escala', 'visual:d2': 'D2', 'heuristic:antipodal': 'antipodal'}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True); p.add_argument('--rows', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    analysis = json.loads(Path(a.analysis).read_text())
    rows = stats.select(json.loads(Path(a.rows).read_text())['rows'], 'B', 80)
    budgets = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.1]))
    for ax in (left, right):
        style(ax)
    for m, label, color, marker in SERIES:
        b = stats.budget(rows, m, budgets)
        left.plot(budgets, [b['budgets'][str(x)]['rate'] for x in budgets], color=color, lw=2, marker=marker, ms=5,
                  markeredgecolor='white', markeredgewidth=.8, label=label)
    for m, label, ls in REFERENCE:
        b = stats.budget(rows, m, budgets)
        left.plot(budgets, [b['budgets'][str(x)]['rate'] for x in budgets], color=GRAY, ls=ls, lw=1.6, label=label)
    left.legend(loc='lower right', frameon=False, fontsize=8, labelcolor=INK)
    left.set_xscale('log'); left.set_xlim(.8, budgets[-1]*1.3)
    left.set_xticks(budgets, [str(x) for x in budgets]); left.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    left.set_ylim(0, 1); left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    left.set_xlabel('Presupuesto: candidatos que el planificador puede examinar', color=MUTED, fontsize=8.5)
    left.set_ylabel('Piezas con agarre viable en el presupuesto', color=MUTED, fontsize=8.5)
    n_parts = len({r['part'] for r in rows})
    left.set_title(f'Éxito frente a presupuesto · {n_parts} piezas nuevas (sin agarre = fallo)', fontsize=10,
                   color=INK, loc='left')

    shown = list(analysis['primary']) + [c for c in analysis['secondary'] if c['experiment'] == 'B' and c['gripper'] == 80
                                         and c['target'] == 1 and (c['a'], c['b']) in
                                         {('learned', 'default'), ('box', 'default'), ('visual:dinov3', 'box'),
                                          ('fusion:dinov3', 'box')}]
    y = list(range(len(shown)))[::-1]
    for yi, c in zip(y, shown):
        primary = c in analysis['primary']
        color = '#f47230' if primary else '#257de0'
        right.plot(c['ci95'], [yi, yi], color=color, lw=2, solid_capstyle='round')
        right.scatter([c['ratio']], [yi], s=46, color=color, edgecolor='white', linewidth=1.2, zorder=3)
        verdict = c.get('decision', '')
        verdict = {'different': ' · distinto', 'equivalent': ' · equivalente', 'not shown': ' · no demostrado'}.get(verdict, '')
        right.text(c['ci95'][1]*1.08, yi, f"{c['ratio']:.2f} [{c['ci95'][0]:.2f}–{c['ci95'][1]:.2f}]{verdict}",
                   va='center', fontsize=7.5, color=INK, bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none'))
    right.axvline(1, color=MUTED, lw=.8)
    m = analysis['margin']
    right.axvspan(1/m, m, color='#f2f2f2', zorder=0)
    right.set_yticks(y, [f"{SHORT.get(c['a'], c['a'])} ÷ {SHORT.get(c['b'], c['b'])}" for c in shown], color=INK)
    right.set_xscale('log')
    lo = min(c['ci95'][0] for c in shown); hi = max(c['ci95'][1] for c in shown)
    right.set_xlim(lo/1.3, hi*6)
    ticks = [t for t in (.1, .2, .5, 1, 2, 5) if lo/1.3 <= t <= hi*6]
    right.set_xticks(ticks, [f'{t:g}' for t in ticks]); right.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    right.set_xlabel('Candidatos hasta el primer viable, cociente por pieza (IC 95 %);\n'
                     f'banda gris: margen de equivalencia [{1/m:.2f}, {m:.2f}]', color=MUTED, fontsize=8)
    right.set_title('Contrastes (naranja: principales del protocolo)', fontsize=10, color=INK, loc='left')
    fig.suptitle('Fase V · prueba con biblioteca congelada (116 piezas) y pinza de 80 mm', fontsize=11, color=INK,
                 x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, facecolor='white')


if __name__ == '__main__':
    main()
