#!/usr/bin/env python3
"""Figures S3-S5: identification rate against k, one figure per beacon size.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K_VALUES = [2, 3, 5, 7, 10]
SIZES    = [10, 20, 40, 63]
AGHDAM   = 3.3

STRATEGIES = [
    ('S1: Majority Voting',     's1',  'o-', '#e41a1c'),   # red
    ('S2a: Random Partition',   's2a', 's-', '#ff7f00'),   # orange
    ('S2b: MAF Partition',      's2b', '^-', '#4C72B0'),   # blue
    ('S2c: Category Partition', 's2c', 'D-', '#984ea3'),   # purple
    ('S2d: Strided Partition',  's2d', 'P-', '#2ca02c'),   # green
]

SINGLE_ATTACKER = (
    'Single Attacker (scaled queries)', 'single', 'x--', '#888888'
)

DEFENSES = [
    ('random',    'Random Defense'),
    ('baseline',  'Baseline Defense'),
    ('strategic', 'Strategic Defense'),
    ('agent',     'GBD (RL Defender)'),
]

DEF_COLORS = {
    'random':    ('#e41a1c', 'o-', 'Random Defense'),
    'baseline':  ('#ff7f00', 's-', 'Baseline Defense'),
    'strategic': ('#4C72B0', '^-', 'Strategic Defense'),
    'agent':     ('#2ca02c', 'D-', 'GBD (RL Defender)'),
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def load_csv(path):
    """{(N, defense, strategy, k): [rate_per_seed]} as percentages."""
    out = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = (int(row['N']), row['defense'], row['strategy'], int(row['k']))
            out[key].append(float(row['id_rate']) * 100)
    return out


def series(data, N, defense, strategy):
    means, stds = [], []
    for k in K_VALUES:
        rates = data.get((N, defense, strategy, k), [])
        means.append(np.mean(rates) if rates else np.nan)
        stds.append(np.std(rates) if rates else np.nan)
    return np.array(means), np.array(stds)


def draw(data, N, outdir, paper=False, prefix='fig_idrate'):
    # k is a category, evenly spaced, as in Fig. 2
    xf = np.arange(len(K_VALUES), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for ax, (defense, title) in zip(axes, DEFENSES):
        for sname, skey, marker, color in STRATEGIES:
            mean, std = series(data, N, defense, skey)
            ax.plot(xf, mean, marker, color=color, linewidth=2,
                    markersize=6, label=sname)
            ax.fill_between(xf, mean - std, mean + std, alpha=0.12, color=color)

        sname, skey, marker, color = SINGLE_ATTACKER
        mean, std = series(data, N, defense, skey)
        ax.plot(xf, mean, marker, color=color, linewidth=1.6,
                markersize=6, alpha=0.85, label=sname)
        ax.fill_between(xf, mean - std, mean + std, alpha=0.08, color=color)

        if not paper:
            ref = ax.axhline(AGHDAM, color='#888888', linewidth=1.2,
                             linestyle='--', zorder=1)
            ref.set_label(f'Poorghaffar Aghdam et al. (2025), N=40 ({AGHDAM}%)')

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(xf)
        ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=10)
        ax.set_xlabel('Number of Attackers (k)', fontsize=11, labelpad=6)
        ax.set_ylabel('Identification Rate (%)', fontsize=11)
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8.5, loc='lower right',
                  framealpha=0.95, edgecolor='#cccccc')

    if not paper:
        fig.suptitle(
            f'All Strategies vs All Defenses, Identification Rate vs k · '
            f'Beacon Size N = {N}\n'
            'Dashed Gray = Single Attacker Control (scaled queries) · '
            '100 Queries/Account · 400 Episodes · 3 Seeds · Shading = ±1 Std',
            fontsize=13, fontweight='bold', y=1.01
        )
    plt.tight_layout()
    out = os.path.join(outdir, f'{prefix}_N{N}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def draw_per_strategy(data, N, outdir, paper=False, prefix='fig_idrate'):
    # k is a category, evenly spaced, as in Fig. 2
    xf = np.arange(len(K_VALUES), dtype=float)
    # paper mode matches Fig. 2's layout
    tick_fs, lab_fs, leg_fs, title_fs = (16, 18, 17, 15) if paper else (10, 11, 8.5, 12)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5) if paper else (18, 11))
    axes = axes.flatten()

    panels = [(sname, skey) for sname, skey, _, _ in STRATEGIES]
    panels.append((SINGLE_ATTACKER[0], SINGLE_ATTACKER[1]))

    for idx, (ax, (sname, skey)) in enumerate(zip(axes, panels)):
        for defense, (color, marker, dlabel) in DEF_COLORS.items():
            mean, std = series(data, N, defense, skey)
            ax.plot(xf, mean, marker, color=color,
                    linewidth=2.6 if paper else 2,
                    markersize=8 if paper else 6, label=dlabel)
            ax.fill_between(xf, mean - std, mean + std, alpha=0.12, color=color)

        if not paper:
            ref = ax.axhline(AGHDAM, color='#888888', linewidth=1.2,
                             linestyle='--', zorder=1)
            ref.set_label(f'Poorghaffar Aghdam et al. (2025), N=40 ({AGHDAM}%)')

        ax.set_title(f'({chr(65 + idx)})  {sname}' if paper else sname,
                     fontsize=title_fs, fontweight='bold', pad=10)
        ax.set_xticks(xf)
        ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=tick_fs)
        ax.tick_params(axis='y', labelsize=tick_fs)
        ax.set_xlabel('Number of Attackers (k)', fontsize=lab_fs, labelpad=6)
        ax.set_ylabel('Identification Rate (%)', fontsize=lab_fs)
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.grid(axis='y', alpha=0.3)
        if not paper:
            ax.legend(fontsize=leg_fs, loc='lower right',
                      framealpha=0.95, edgecolor='#cccccc')

    if paper:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4,
                   fontsize=leg_fs, frameon=False, bbox_to_anchor=(0.5, -0.02))
    else:
        fig.suptitle(
            f'Per-Strategy View: Identification Rate vs k Across All Defenses · '
            f'Beacon Size N = {N}\n'
            '100 Queries/Account · 400 Episodes · 3 Seeds · Shading = ±1 Std',
            fontsize=13, fontweight='bold', y=1.01
        )
    plt.tight_layout(w_pad=2.0, h_pad=1.6) if paper else plt.tight_layout()
    if paper:
        fig.subplots_adjust(bottom=0.14)
    out = os.path.join(outdir, f'{prefix}_perstrategy_N{N}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rates', required=True, type=str,
                   help='per-seed identification rates CSV')
    p.add_argument('--outdir', default='./results/beacon_size_idrate', type=str)
    p.add_argument('--sizes', nargs='+', type=int, default=SIZES,
                   help='beacon sizes to draw; a size absent from the CSV is skipped')
    p.add_argument('--paper', action='store_true',
                   help='drop the in-image title and the N=40 reference line, so the '
                        'LaTeX caption carries the description')
    p.add_argument('--prefix', default='fig_idrate', type=str)
    a = p.parse_args()

    data = load_csv(a.rates)
    present = sorted({key[0] for key in data})
    sizes = [N for N in a.sizes if N in present]
    missing = [N for N in a.sizes if N not in present]
    if missing:
        print(f'no rows for N={missing}, skipping those panels')
    if not sizes:
        raise SystemExit(f'none of N={a.sizes} present in {a.rates}')

    os.makedirs(a.outdir, exist_ok=True)
    for N in sizes:
        draw(data, N, a.outdir, a.paper, a.prefix)
        draw_per_strategy(data, N, a.outdir, a.paper, a.prefix)

    print()
    for N in sizes:
        print(f'\n=== Beacon size N = {N}, identification rate (%) ===')
        names = [s[0].split(':')[0] for s in STRATEGIES] + ['Single']
        keys = [s[1] for s in STRATEGIES] + [SINGLE_ATTACKER[1]]
        for defense, title in DEFENSES:
            print(f'\n  {title}')
            print('     k' + ''.join(f'{n:>16}' for n in names))
            for i, k in enumerate(K_VALUES):
                row = f'{k:>6}'
                for key in keys:
                    mean, std = series(data, N, defense, key)
                    row += f'{mean[i]:>9.1f} ±{std[i]:>4.1f}'
                print(row)


if __name__ == '__main__':
    main()
