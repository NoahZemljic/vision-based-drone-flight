#!/usr/bin/env python3
"""Regenerate final-report Figures 1 and 2 from the canonical fleet JSON files."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'run/vbdf'))
from report_results import FLEET, SEEDS, summarize

COLORS = {'roll': '#D55E00', 'pitch': '#E69F00', 'yaw': '#0072B2',
          'state': '#0072B2', 'vision': '#D55E00', 'gray': '#8a8a8a'}
MARKERS = ('o', 's', '^')


def seed_legend():
    return [Line2D([], [], marker=m, linestyle='', color='#202020', markersize=3,
                   label=f'Seed {s}') for s, m in zip(SEEDS, MARKERS)]


def bars(ax, x, rows, key, color, width):
    values = np.array([rows[str(seed)][key] for seed in SEEDS], dtype=float)
    avg, lo, hi = values.mean(), values.min(), values.max()
    ax.bar(x, avg, width=width, color=color, alpha=0.38, linewidth=0, zorder=2)
    ax.errorbar(x, avg, yerr=[[avg-lo], [hi-avg]], fmt='none', color=color,
                elinewidth=0.9, capsize=2, zorder=3)
    for offset, value, marker in zip((-0.28, 0, 0.28), values, MARKERS):
        ax.scatter(x + offset * width, value, s=9, marker=marker,
                   color='#202020', linewidths=0, zorder=4)


def style(ax, title, ylabel=None):
    ax.set_title(title, loc='left', weight='bold', pad=5)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', color='#c8c8c8', linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.tick_params(length=3, pad=2)


def figure_1(cells):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5))
    specs = [
        ('all_envs_dist_err', r'$|d_t-d^*|$  (m)', 'a  Distance error'),
        ('all_envs_omex_std2', 'Roll-rate std.  (rad/s)', 'b  Roll-rate variation'),
        ('all_envs_crash_episodes', 'Crashes per 32 simulations', 'c  Number of crashes'),
        ('all_envs_crash_n_envs', 'Simulations  (of 32)', 'd  Simulations with a crash'),
    ]
    groups = [('state/S5-RS', 'state/S5-100m', 'state'),
              ('vision/V3-final', 'vision/V3-100m', 'vision')]
    for ax, (key, ylabel, title) in zip(axes.flat, specs):
        for x, (cell50, cell100, color) in enumerate(groups):
            bars(ax, x-0.16, cells[cell50]['seeds'], key, COLORS['gray'], 0.295)
            bars(ax, x+0.16, cells[cell100]['seeds'], key, COLORS[color], 0.295)
        style(ax, title, ylabel)
        ax.set_xticks([0, 1], ['S5', 'V3'])
        ax.set_xlim(-0.55, 1.55)
        ax.margins(y=0.16)
    handles = [Patch(facecolor=COLORS[c], alpha=0.55, label=label)
               for c, label in [('gray', '50 M'), ('state', '100 M  (S5)'), ('vision', '100 M  (V3)')]]
    fig.legend(handles=handles+seed_legend(), loc='upper center', ncol=6,
               handlelength=1, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.05, w_pad=1.5)
    return fig


def figure_2(cells):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=True)
    groups = [('state', ['S0-world', 'S1-body', 'S2-AC', 'S3-PV', 'S4-LPF', 'S5-RS'],
               'a  State-based tracking'),
              ('vision', ['V0-base', 'V1-AC', 'V2-LPF', 'V3-final'], 'b  Bounding-box tracking')]
    for ax, (stage, names, title) in zip(axes, groups):
        for x, name in enumerate(names):
            for i, (axis, color) in enumerate(zip('xyz', ['roll', 'pitch', 'yaw'])):
                bars(ax, x+(i-1)*0.24, cells[f'{stage}/{name}']['seeds'],
                     f'all_envs_ome{axis}_std2', COLORS[color], 0.22)
        style(ax, title)
        ax.set_xticks(range(len(names)), [n.split('-')[0] for n in names])
        ax.set_xlim(-0.55, len(names)-0.45)
    axes[0].set_ylim(0, max(ax.dataLim.ymax for ax in axes) * 1.05)
    axes[0].set_ylabel('Body-rate std.  (rad/s)')
    handles = [Patch(facecolor=COLORS[c], alpha=0.55, label=c) for c in ['roll', 'pitch', 'yaw']]
    fig.legend(handles=handles+seed_legend(), loc='upper center', ncol=6,
               handlelength=1, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=1.35)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=FLEET)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    # Fail on incomplete/mismatched evidence instead of drawing partial bars.
    cells = summarize(args.results)['cells']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'legend.frameon': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 0.65})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, make in [('figure_1_training_budget', figure_1), ('figure_2_body_rate_variation', figure_2)]:
        fig = make(cells)
        path = args.out / f'{name}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
        plt.close(fig)
        print(path)


if __name__ == '__main__':
    main()
