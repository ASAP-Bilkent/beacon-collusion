#!/usr/bin/env python3
"""Figure 2: identification rate against coalition size, one panel per strategy.
Reads the member arm of results/{gbd_fpr,heuristic_fpr,scaled_control,s2g_ref}.
"""

import argparse
import collections
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K_VALUES = [1, 2, 3, 5, 7, 10]
FLAGS = ['majority_voting', 'random_partitioning', 'k_partitioning',
         'maf_specialization', 'strided_partitioning']
STRAT_OF_FLAGS = {('majority_voting',): 'S1', ('random_partitioning',): 'S2a',
                  ('k_partitioning',): 'S2b', ('maf_specialization',): 'S2c',
                  ('strided_partitioning',): 'S2d', (): 'single'}
PANELS = [('S1', 'S1: Majority Voting'), ('S2a', 'S2a: Random Partition'),
          ('S2b', 'S2b: MAF Partition'), ('S2c', 'S2c: Category Partition'),
          ('S2d', 'S2d: Strided Partition'),
          ('single', 'Single Attacker (scaled queries)')]
DEFENSES = [('random', '#e41a1c', 'o-', 'Random Defense'),
            ('baseline', '#ff7f00', 's-', 'Baseline Defense'),
            ('strategic', '#4C72B0', '^-', 'Strategic Defense'),
            ('agent', '#2ca02c', 'D-', 'GBD (RL Defender)')]


def _arg(a, key, default=None):
    if a is None:
        return default
    return a.get(key, default) if isinstance(a, dict) else getattr(a, key, default)


def collect():
    """Return {(panel, defense, k): [id_rate per seed]} for the member arm."""
    acc = collections.defaultdict(list)

    for d in glob.glob('results/heuristic_fpr/*') + glob.glob('results/gbd_fpr/*'):
        path = os.path.join(d, 'eval_results.npy')
        if not os.path.exists(path):
            continue
        obj = np.load(path, allow_pickle=True).item()
        args = obj['args']
        if _arg(args, 'victim_prob') != 1.0:
            continue
        strat = STRAT_OF_FLAGS.get(tuple(f for f in FLAGS if _arg(args, f, False)))
        if strat is None:
            continue
        rate = obj['collusion_summary']['id_rate'] * 100
        defense = str(_arg(args, 'beacon_type'))
        k = _arg(args, 'collusion_k')
        if strat == 'single' and k == 1:
            # k=1 point of every panel
            for p, _ in PANELS:
                acc[(p, defense, 1)].append(rate)
        else:
            acc[(strat, defense, k)].append(rate)

    # panel F: one account, whole coalition budget
    for d in glob.glob('results/scaled_control/*') + glob.glob('results/s2g_ref/*'):
        path = os.path.join(d, 'eval_results.npy')
        if not os.path.exists(path):
            continue
        obj = np.load(path, allow_pickle=True).item()
        args = obj['args']
        if _arg(args, 'victim_prob') != 1.0 or _arg(args, 'collusion_k') != 1:
            continue
        q = _arg(args, 'queries_per_account')
        if q is None or q % 100 or q // 100 not in K_VALUES:
            continue
        acc[('single', str(_arg(args, 'beacon_type')), q // 100)].append(
            obj['collusion_summary']['id_rate'] * 100)

    return acc


def series(acc, panel, defense):
    mean, std = [], []
    for k in K_VALUES:
        vals = acc.get((panel, defense, k))
        mean.append(np.mean(vals) if vals else np.nan)
        std.append(np.std(vals) if vals and len(vals) > 1 else 0.0)
    return np.array(mean), np.array(std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--style', choices=['original', 'revised'], default='revised')
    p.add_argument('--baseline_rate', type=float, default=None,
                   help='Reference line. Defaults to our own single-account GBD rate.')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    acc = collect()
    rev = args.style == 'revised'

    ref = args.baseline_rate
    if ref is None:
        ref = (float(np.mean(acc[('single', 'agent', 1)]))
               if acc.get(('single', 'agent', 1)) else 3.3)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    xf = np.arange(len(K_VALUES))

    tick_fs, lab_fs, leg_fs, title_fs = (16, 18, 17, 15) if rev else (10, 11, 8.5, 12)

    for idx, (ax, (panel, title)) in enumerate(zip(axes.flatten(), PANELS)):
        for defense, color, marker, dlabel in DEFENSES:
            mean, std = series(acc, panel, defense)
            ax.plot(xf, mean, marker, color=color,
                    linewidth=2.6 if rev else 2, markersize=8 if rev else 6,
                    label=dlabel)
            ax.fill_between(xf, mean - std, mean + std, alpha=0.12, color=color)

        if rev:
            # named in the shared legend, not per panel
            ax.axhline(ref, color='#111111', linewidth=3.4, linestyle=(0, (5, 2.5)),
                       zorder=5)
            ax.set_title(f'({chr(65 + idx)})  {title}', fontsize=title_fs,
                         fontweight='bold', pad=10)
            ax.set_ylabel('Identification Rate (%)', fontsize=lab_fs)
        else:
            ax.axhline(3.3, color='#888888', linewidth=1.2, linestyle='--', zorder=1)
            ax.text(xf[-1] - 4.5, 3.3 + 2.5,
                    'Poorghaffar Aghdam et al. (2025): 3.3%',
                    color='#888888', fontsize=7.5, ha='right')
            ax.set_title(title, fontsize=title_fs, fontweight='bold', pad=10)
            ax.set_ylabel('Identification Rate (%)', fontsize=lab_fs)

        ax.set_xticks(xf)
        ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=tick_fs)
        ax.tick_params(axis='y', labelsize=tick_fs)
        ax.set_xlabel('Number of Attackers (k)', fontsize=lab_fs, labelpad=6)
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.grid(axis='y', alpha=0.3)
        if not rev:
            ax.legend(fontsize=leg_fs, loc='lower right', framealpha=0.95,
                      edgecolor='#cccccc')

    if rev:
        handles, labels = axes.flatten()[0].get_legend_handles_labels()
        handles.append(matplotlib.lines.Line2D(
            [], [], color='#111111', linewidth=3.4, linestyle=(0, (5, 2.5))))
        labels.append(f'single account vs GBD, this work: {ref:.1f}%')
        fig.legend(handles, labels, loc='lower center', ncol=3,
                   fontsize=leg_fs, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(w_pad=2.0, h_pad=1.6)
    if rev:
        fig.subplots_adjust(bottom=0.17)
    out = args.out or f'results/fig2_rebuilt_{args.style}.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f'saved {out}')

    missing = [(p_, d, k) for p_, _ in PANELS for d, _, _, _ in DEFENSES
               for k in K_VALUES if not acc.get((p_, d, k))]
    print(f'{len(missing)} empty cells' + (f': {missing[:6]}' if missing else ''))


if __name__ == '__main__':
    main()
