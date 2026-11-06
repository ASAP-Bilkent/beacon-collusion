# Coordinated multi-account membership inference against genomic beacons

Code for the paper. Five collusion strategies, each a rule for splitting the
genomic search space among k accounts plus a rule for combining their evidence,
evaluated against four beacon defenses.

Strategies: S1 (no partition, majority voting), S2a (random partition), S2b
(MAF-ranked equal-count blocks), S2c (fixed MAF categories), S2d (strided).
S2a to S2d declare identification as soon as any one account reaches p < 0.05.

Defenses: Random response flipping, Baseline MAF-threshold suppression,
Strategic discriminative-power suppression, and GBD, the learned per-query
policy.

Every account runs the released GBA attacker policy unchanged and GBD is the
released defender checkpoint, both from
[ciceklab/beacon-defense-strategies](https://github.com/ciceklab/beacon-defense-strategies).
Only the distribution of that fixed policy across accounts varies.

## Setup

```bash
conda env create -f environment.yml
conda activate beacon
```

Runs on CPU. `environment.yml` pins a CUDA build of PyTorch; without a GPU,
swap `pytorch-cuda=11.8` for `cpuonly` first.

## Data

The CEU HapMap data is the same as the base paper's and is not duplicated here.
It is on Git LFS in that repository:

```bash
git lfs install
git clone https://github.com/ciceklab/beacon-defense-strategies.git
cd beacon-defense-strategies && git lfs pull && cd ..
cp -r beacon-defense-strategies/data ./data
```

That gives `data/CEU/` with `Beacon_164.txt`, `MAF.txt` and `reference.pickle`.
Experiments use the first 100,000 SNPs in the file's original order, sorted by
ascending MAF. The first run caches the parsed genotype matrix to
`binary_cache_CEU.joblib`.

## Running the experiments

Each driver calls `eval_collusion.py` once per cell and writes one
`eval_results.npy` per cell. Re-running skips cells that are already done.
Drivers set `DEBUG_COLLUSION_CONTINUE=1` and `BEACON_BINARY=0` themselves, so
episodes run the full budget and responses are continuous. Set `BEACON_PY` to
pick the interpreter for the workers.

### Figure 2

N = 40, 400 episodes, seeds 1-3, 100 queries per account, k in {1,2,3,5,7,10},
member and non-member arms.

```bash
python3 run_gbd_fpr.py          # 156 cells -> results/gbd_fpr
python3 run_heuristic_fpr.py    # 468 cells -> results/heuristic_fpr
python3 run_scaled_control.py   #  30 cells -> results/scaled_control
python3 plot_fig2.py --style revised
```

Output is `results/fig2_rebuilt_revised.png`.

`run_scaled_control.py` gives panel F, the single account spending the whole kB
budget, for GBD. The three heuristic lines in that panel come from
`results/s2g_ref`, produced the same way with `--beacon_type` changed; without
those cells the panel draws the GBD line only.

`run_heuristic_fpr.py` reads `HF_EPISODES`, `HF_SEEDS`, `HF_DEFENSES` and
`HF_PARALLEL` from the environment for shorter runs.

### Supplementary Figure S1, partition geometry

```bash
python3 plot_partitions.py
```

Self-contained, needs neither the data nor any runs. Writes
`results/fig_partitions.png`.

### Supplementary Figure S2, query efficiency

```bash
python3 plot_fig3_queries.py
```

Mean queries to identification over successful episodes. Reads the Figure 2
cells, so run those drivers first. Writes
`results/fig3_avg_queries_4panel.png`.

### Supplementary Figures S3 to S5, beacon size

N = 63 is the largest the 164-person panel allows once the victim and both
50-person control sets are reserved.

```bash
python3 run_beacon_size.py \
    --sizes 10 20 63 \
    --defenses agent baseline strategic random \
    --strategies s1 s2a s2b s2c s2d single \
    --collusion_ks 2 3 5 7 10 --seeds 1 2 3 --arms mem \
    --n_episodes 400 --jobs 8 --out ./results/beacon_size_gated
```

1,080 cells. Each worker holds the genotype matrix, roughly 5 GB, so keep
`jobs x 5 GB` under available RAM. The N = 63 block alone is 360 cells and took
about 21 hours at `--jobs 36`.

`results/beacon_size_idrate_gated.csv` holds the summarised result of that
sweep, one row per cell: beacon size, defense, strategy, k, seed,
identification rate, episode count, majority-voting flag. The rate is each run's
own decision flag, so S1 keeps its majority rule and S2a to S2d keep the any-one
rule; nothing is re-scored. The figures come straight from it, so they can be
redrawn without repeating the sweep:

```bash
python3 plot_beacon_size_idrate.py \
    --rates ./results/beacon_size_idrate_gated.csv \
    --sizes 10 20 63 --paper --prefix fig_beacon_size \
    --outdir ./results/beacon_size_idrate
```

Writes `fig_beacon_size_perstrategy_N{10,20,63}.png`, which are Figures S3, S4
and S5.

## Single cell

`eval_collusion.py` runs one configuration directly. S2c at k = 3 against GBD:

```bash
DEBUG_COLLUSION_CONTINUE=1 BEACON_BINARY=0 python3 eval_collusion.py \
    --data ./data/CEU --gene_size 100000 \
    --gbd_checkpoint ./weights/GBD --gba_checkpoint ./weights/GBA.pth \
    --beacon_type agent --beacon_size 40 \
    --a_control_size 50 --b_control_size 50 \
    --collusion_mode round_robin --collusion_k 3 --maf_specialization \
    --queries_per_account 100 --n_episodes 400 --seed 1 \
    --presence_aware_response --private_account_evidence --individual_local \
    --log_scores --results_dir ./results/example_cell
```

Four flags change the numbers materially:

- `--presence_aware_response` applies the defense only where some member carries
  the queried allele. Without it the attacker gets the policy's honesty value
  even for alleles nobody carries, which inflates identification.
- `--private_account_evidence` keeps each account's evidence and control set to
  its own queries.
- `--individual_local` runs a local test per account instead of one pooled
  coalition statistic.
- `--calibrated_null` draws the target from the control pool, giving the
  non-member arm.

The beacon, both control sets and one spare slot must fit in the 164-person
panel, so `beacon_size <= 164 - 1 - a_control_size - b_control_size`, which is
63 at the defaults. A configuration that does not fit raises an error rather
than silently shrinking the control sets.

## Not included

Raw evaluation output is about 1.8 GB, so `results/` is ignored apart from the
CSV above. The data is about 2.2 GB and comes from the base repository.

## License

GPL-3.0, see `LICENSE`.
