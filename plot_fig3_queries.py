#!/usr/bin/env python3
"""Figure S2: mean queries to identification over successful episodes.
Reads the same cells as plot_fig2.py.
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
STRATEGIES = [('S1', 'S1: Majority Voting', 'o-', '#e41a1c'),
              ('S2a', 'S2a: Random Partition', 's-', '#ff7f00'),
              ('S2b', 'S2b: MAF Partition', '^-', '#4C72B0'),
              ('S2c', 'S2c: Category Partition', 'D-', '#984ea3'),
              ('S2d', 'S2d: Strided Partition', 'P-', '#2ca02c')]
SINGLE = ('single', 'Single Attacker (scaled queries)', 'x--', '#888888')
DEFENSES = [('random', 'Random Defense'), ('baseline', 'Baseline Defense'),
            ('strategic', 'Strategic Defense'), ('agent', 'GBD (RL Defender)')]


def _arg(a, key, default=None):
    if a is None:
        return default
    return a.get(key, default) if isinstance(a, dict) else getattr(a, key, default)


def _queries(obj):
    q = obj['collusion_summary'].get('avg_queries_to_id')
    return None if q is None or (isinstance(q, float) and np.isnan(q)) else float(q)


def collect():
    """Return {(strategy, defense, k): [avg queries per seed]} for the member arm."""
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
        q = _queries(obj)
        if q is None:
            continue
        defense = str(_arg(args, 'beacon_type'))
        k = _arg(args, 'collusion_k')
        if strat == 'single' and k == 1:
            # k=1 point of every line
            for s, _, _, _ in STRATEGIES + [SINGLE[:1] + SINGLE[1:]]:
                acc[(s, defense, 1)].append(q)
        else:
            acc[(strat, defense, k)].append(q)

    # control: one account, whole coalition budget
    for d in glob.glob('results/scaled_control/*') + glob.glob('results/s2g_ref/*'):
        path = os.path.join(d, 'eval_results.npy')
        if not os.path.exists(path):
            continue
        obj = np.load(path, allow_pickle=True).item()
        args = obj['args']
        if _arg(args, 'victim_prob') != 1.0 or _arg(args, 'collusion_k') != 1:
            continue
        qpa = _arg(args, 'queries_per_account')
        if qpa is None or qpa % 100 or qpa // 100 not in K_VALUES:
            continue
        q = _queries(obj)
        if q is not None:
            acc[('single', str(_arg(args, 'beacon_type')), qpa // 100)].append(q)

    return acc


def series(acc, strat, defense):
    mean, std = [], []
    for k in K_VALUES:
        vals = acc.get((strat, defense, k))
        mean.append(np.mean(vals) if vals else np.nan)
        std.append(np.std(vals) if vals and len(vals) > 1 else 0.0)
    return np.array(mean), np.array(std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/fig3_avg_queries_4panel.png')
    args = p.parse_args()

    acc = collect()
    lines = [(s, lab, m, c) for s, lab, m, c in STRATEGIES] + [SINGLE]

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    tick_fs, lab_fs, leg_fs, title_fs = 16, 18, 17, 15
    fig, axes = plt.subplots(2, 2, figsize=(14.45, 9))
    xf = np.arange(len(K_VALUES), dtype=float)

    for idx, (ax, (defense, title)) in enumerate(zip(axes.flatten(), DEFENSES)):
        top = []
        for strat, label, marker, color in lines:
            mean, std = series(acc, strat, defense)
            ok = ~np.isnan(mean)
            if not ok.any():
                continue
            single = strat == 'single'
            ax.plot(xf[ok], mean[ok], marker, color=color,
                    linewidth=2.0 if single else 2.6,
                    markersize=8, alpha=0.85 if single else 1.0, label=label)
            ax.fill_between(xf[ok], (mean - std)[ok], (mean + std)[ok],
                            alpha=0.08 if single else 0.12, color=color)
            top.extend((mean + std)[ok].tolist())

        ax.set_title(f'({chr(65 + idx)})  {title}', fontsize=title_fs,
                     fontweight='bold', pad=10)
        ax.set_xticks(xf)
        ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=tick_fs)
        ax.tick_params(axis='y', labelsize=tick_fs)
        ax.set_xlabel('Number of Attackers (k)', fontsize=lab_fs, labelpad=6)
        ax.set_ylabel('Avg Queries to Identification', fontsize=lab_fs)
        ax.set_ylim(0, max(top) * 1.12 if top else 300)
        ax.grid(axis='y', alpha=0.3)

    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=leg_fs,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(w_pad=2.0, h_pad=1.6)
    fig.subplots_adjust(bottom=0.17)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    print(f'saved {args.out}')

    print('\navg queries to identification, GBD:')
    for strat, label, _, _ in lines:
        mean, _ = series(acc, strat, 'agent')
        print(f'  {label:<34} ' + '  '.join(
            f'k={k}:{m:7.1f}' if not np.isnan(m) else f'k={k}:    n/a'
            for k, m in zip(K_VALUES, mean)))


if __name__ == '__main__':
    main()
