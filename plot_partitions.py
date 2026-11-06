#!/usr/bin/env python3
"""Figure S1: how each strategy divides the MAF-ordered genome at k = 3.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np

K = 3
N = 60                                               # SNP cells drawn per row
COLORS = ['#4C72B0', '#DD8452', '#55A868']           # account 1, 2, 3
CAT_N = [43350, 11495, 12992, 11523, 10496, 10144]   # Table 1 SNP counts
CAT_SHARE = [n / sum(CAT_N) for n in CAT_N]
INK = '#333333'

W = 1.0 / N
GAP = 0.20 * W                                       # white space between SNPs

fig = plt.figure(figsize=(11, 8.4))
gs = GridSpec(6, 1, figure=fig, height_ratios=[0.34, 1, 1, 1, 1, 1], hspace=0.55)
axr = fig.add_subplot(gs[0])
axes = [fig.add_subplot(gs[i + 1]) for i in range(5)]
rng = np.random.default_rng(7)


def cells(ax, owners, y=0.0, h=1.0):
    """Draw one cell per SNP; owners[i] is the account index for SNP i."""
    for i, o in enumerate(owners):
        ax.add_patch(plt.Rectangle((i * W + GAP / 2, y), W - GAP, h,
                                   facecolor=COLORS[o], linewidth=0))


def label(ax, name, text):
    ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=13,
                  fontweight='bold', labelpad=12)
    ax.set_title(text, fontsize=10, loc='left', pad=5, color=INK)


# header
fig.text(0.5, 0.975, 'How the three accounts divide the beacon between them',
         ha='center', va='top', fontsize=14, fontweight='bold', color='#111111')
fig.text(0.5, 0.942,
         'Each narrow cell is one SNP, and its colour shows which account asks '
         'about it. All five rows hold the same SNPs in the same order.',
         ha='center', va='top', fontsize=11, color=INK)

# ruler
axr.annotate('', xy=(1, 0.30), xytext=(0, 0.30),
             arrowprops=dict(arrowstyle='<->', lw=1.1, color='#777777'))
axr.text(0.5, 0.44, 'sorted by minor allele frequency', ha='center', va='bottom',
         fontsize=10, color='#777777')
axr.set_xlim(0, 1); axr.set_ylim(0, 1)
axr.set_xticks([]); axr.set_yticks([]); axr.spines[:].set_visible(False)

# S1: every account holds the whole genome
ax = axes[0]
for i in range(K):
    cells(ax, [i] * N, y=i / K, h=1 / K)
label(ax, 'S1', 'S1  no split: all three accounts ask about every SNP, '
                'so each SNP carries all three colours')

# S2a: random subset per account
ax = axes[1]
cells(ax, rng.permutation(np.repeat(np.arange(K), N // K)))
label(ax, 'S2a', 'S2a  random split: each account takes a random third of the '
                 'SNPs, scattered across the whole range')

# S2b: contiguous blocks of equal SNP count
ax = axes[2]
cells(ax, np.repeat(np.arange(K), N // K))
label(ax, 'S2b', 'S2b  equal blocks: each account takes a third of the SNPs '
                 'in one unbroken stretch of the range')

# S2c: fixed MAF categories, unequal counts
ax = axes[3]
share = [sum(CAT_SHARE[2 * i:2 * i + 2]) for i in range(K)]
edge = [0] + [int(round(c * N)) for c in np.cumsum(share)]
owners = np.concatenate([[i] * (edge[i + 1] - edge[i]) for i in range(K)])
cells(ax, owners)
for i in range(K):
    ax.text((edge[i] + edge[i + 1]) / 2 * W, 0.5, f'{share[i] * 100:.1f}%',
            ha='center', va='center', color='white', fontsize=11,
            fontweight='bold',
            bbox=dict(facecolor=COLORS[i], edgecolor='none',
                      boxstyle='round,pad=0.3'))
label(ax, 'S2c', 'S2c  fixed frequency categories: the cuts sit at set MAF '
                 'values, so the accounts get unequal numbers of SNPs')

# S2d: strided
ax = axes[4]
cells(ax, np.arange(N) % K)
label(ax, 'S2d', 'S2d  interleaved split: the accounts take turns down the '
                 'list, so each one covers the whole range')

for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.spines[:].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.set_xticks([])

axes[-1].set_xticks([0, 1])
axes[-1].set_xticklabels(['rarest SNP\n(MAF 0)', 'most common SNP\n(MAF 0.5)'],
                         fontsize=11, color=INK)
for lab, side in zip(axes[-1].get_xticklabels(), ['left', 'right']):
    lab.set_horizontalalignment(side)

fig.legend(handles=[Patch(facecolor=COLORS[i], label=f'account {i + 1}')
                    for i in range(K)],
           loc='lower center', ncol=3, frameon=False, fontsize=12,
           bbox_to_anchor=(0.5, 0.005))
fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.095)
out = 'results/fig_partitions.png'
plt.savefig(out, dpi=200)
print(f'saved {out}')
