"""Figure for the visual grasp-prior evaluation (leave one part out)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {'default': 'Orden por defecto', 'random_expectation': 'Azar (esperado)',
          'random': '3 vecinos al azar', 'box': '3 vecinos por tamaño de caja',
          'pooled': 'Todas las piezas resueltas', 'visual:vit-random': '3 vecinos ViT aleatorio',
          'visual:openclip': '3 vecinos OpenCLIP', 'visual:dinov2': '3 vecinos DINOv2',
          'visual:dinov3': '3 vecinos DINOv3', 'gate:dinov3': 'DINOv3 entre 6 de tamaño parecido',
          'fusion:dinov3': 'DINOv3 + tamaño (fusión de puestos)',
          'visual:vit-random-metric': '3 vecinos ViT aleatorio, con escala',
          'visual:dinov2-metric': '3 vecinos DINOv2, con escala', 'visual:dinov3-metric': '3 vecinos DINOv3, con escala',
          'fusion:dinov3-metric': 'DINOv3 con escala + tamaño',
          'self': 'Sus propios agarres (techo)'}
SHORT = {'box': 'tamaño', 'pooled': 'todas', 'default': 'defecto', 'visual:dinov3': 'DINOv3',
         'visual:dinov2': 'DINOv2', 'visual:openclip': 'OpenCLIP', 'visual:vit-random': 'ViT aleatorio',
         'fusion:dinov3': 'DINOv3 + tamaño', 'fusion:vit-random': 'ViT aleatorio + tamaño',
         'gate:dinov3': 'DINOv3 entre 6 por tamaño', 'visual:dinov3-metric': 'DINOv3 esc.',
         'fusion:dinov3-metric': 'DINOv3 esc. + tamaño', 'visual:vit-random-metric': 'ViT aleat. esc.',
         'visual:dinov2-metric': 'DINOv2 esc.'}
INK, MUTED, GRID = '#191919', '#666666', '#e6e6e6'
BLUE, ORANGE, GRAY = '#257de0', '#f47230', '#9c9c9c'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', required=True); p.add_argument('--out', required=True)
    p.add_argument('--analysis', help='paired statistics: the right panel shows their contrasts instead')
    p.add_argument('--contrast', nargs=2, action='append', metavar=('A', 'B'), help='contrast to show (repeatable)')
    p.add_argument('--highlight', default='visual:dinov2')
    p.add_argument('--title', default='Prior visual de agarre en el planificador nativo · 16 piezas YCB, brazo y pinza de la demo')
    a = p.parse_args()
    summary = json.loads(Path(a.evaluation).read_text())['summary']['1']
    order = [m for m in LABELS if m in summary]
    ratio = [summary[m]['geometric_mean_ratio_to_default'] for m in order]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for ax in (left, right):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    y = np.arange(len(order))[::-1]
    colors = [ORANGE if m == a.highlight else GRAY if m in ('default', 'random_expectation', 'self') else BLUE
              for m in order]
    for yi, r, c in zip(y, ratio, colors):
        left.plot([1, r], [yi, yi], color=c, lw=2, solid_capstyle='round', zorder=2)
        left.scatter([r], [yi], s=46, color=c, edgecolor='white', linewidth=1.2, zorder=3)
        left.text(r*(1.12 if r >= 1 else 1/1.12), yi+.28, f'{r:.2f}', ha='left' if r >= 1 else 'right',
                  va='center', fontsize=8, color=INK)
    left.axvline(1, color=MUTED, lw=.8, zorder=1)
    left.set_yticks(y, [LABELS[m] for m in order], color=INK)
    left.set_xscale('log')
    left.set_xlim(.025, 4)
    left.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    left.set_xlabel('Candidatos examinados hasta el primer agarre viable, ÷ orden por defecto\n'
                    '(media geométrica; a la izquierda de 1, menos trabajo)', color=MUTED, fontsize=8.5)
    left.grid(axis='x', color=GRID, lw=.6); left.set_axisbelow(True)
    left.set_title(f"Media de las {summary['default']['parts']} piezas con agarre viable", fontsize=10,
                   color=INK, loc='left')
    if a.analysis:
        forest(right, json.loads(Path(a.analysis).read_text()), a.contrast or [])
        finish(fig, a)
        return

    parts = sorted(summary['default']['per_part'], key=lambda q: summary['default']['per_part'][q])
    yy = np.arange(len(parts))[::-1]
    show = [('default', GRAY, 'o', 'Orden por defecto'), ('pooled', BLUE, 's', 'Todas las piezas resueltas'),
            ('visual:dinov2', ORANGE, 'D', '3 vecinos DINOv2')]
    for (m, c, marker, label), size in zip(show, (170, 64, 30)):
        x = [summary[m]['per_part'][q] for q in parts]
        right.scatter(x, yy, s=size, marker=marker, color=c, label=label,
                      edgecolor='white', linewidth=.8, zorder=3)
    for q, yi in zip(parts, yy):
        xs = [summary[m]['per_part'][q] for m, *_ in show]
        right.plot([min(xs), max(xs)], [yi, yi], color=GRID, lw=1.4, zorder=1)
    right.set_yticks(yy, parts, color=INK)
    right.set_xscale('log')
    right.set_xlabel('Candidatos examinados hasta el primer agarre viable (escala log)', color=MUTED, fontsize=8.5)
    right.grid(axis='x', color=GRID, lw=.6); right.set_axisbelow(True)
    right.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    right.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.45, -.13), ncol=3,
                 labelcolor=INK, handletextpad=.3, columnspacing=1.2)
    right.set_title('Por pieza, dejando esa pieza fuera de la biblioteca', fontsize=10, color=INK, loc='left')
    finish(fig, a)


def finish(fig, a):
    fig.suptitle(a.title, fontsize=11, color=INK, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, facecolor='white')


def forest(ax, analysis, wanted):
    """Paired contrasts to the first feasible grasp: ratio and 95 % bootstrap interval."""
    rows = [c for c in analysis['primary'] + analysis['secondary'] if c['target'] == 1
            and (not wanted or [c['a'], c['b']] in wanted)]
    rows.sort(key=lambda c: wanted.index([c['a'], c['b']]) if wanted else 0)
    y = np.arange(len(rows))[::-1]
    primary = [(c['a'], c['b']) for c in analysis['primary']]
    for yi, c in zip(y, rows):
        color = ORANGE if (c['a'], c['b']) in primary else BLUE
        ax.plot(c['ci95'], [yi, yi], color=color, lw=2, solid_capstyle='round', zorder=2)
        ax.scatter([c['ratio']], [yi], s=46, color=color, edgecolor='white', linewidth=1.2, zorder=3)
        ax.text(c['ci95'][1]*1.08, yi, f"{c['ratio']:.2f}  [{c['ci95'][0]:.2f}–{c['ci95'][1]:.2f}]"
                f"  {c['better']}/{c['equal']}/{c['worse']}", va='center', fontsize=7.5, color=INK, zorder=4,
                bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none'))
    ax.axvline(1, color=MUTED, lw=.8, zorder=1)
    ax.set_yticks(y, [f"{SHORT.get(c['a'], c['a'])} ÷ {SHORT.get(c['b'], c['b'])}" for c in rows], color=INK)
    ax.set_xscale('log')
    lo = min(c['ci95'][0] for c in rows); hi = max(c['ci95'][1] for c in rows)
    ax.set_xlim(lo/1.3, hi*5)
    ticks = [t for t in (.1, .2, .5, 1, 2, 5, 10) if lo/1.3 <= t <= hi*6]
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
    note = '; esc. = con escala' if any('-metric' in c['a'] for c in rows) else ''
    ax.set_xlabel('Cociente por pieza: media geométrica, IC bootstrap 95 %\n'
                  f'(texto: cociente [IC]  mejor/igual/peor{note})', color=MUTED, fontsize=8)
    ax.set_title('Contrastes del protocolo (naranja: principales)', fontsize=10, color=INK, loc='left')


if __name__ == '__main__':
    main()
