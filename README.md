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

Every account runs the same pre-trained GBA attacker policy and GBD is the
pre-trained defender checkpoint, both in `weights/`. Only the distribution of
that fixed policy across accounts varies.

## Setup

```bash
conda env create -f environment.yml
conda activate beacon
```

Runs on CPU. `environment.yml` pins a CUDA build of PyTorch; without a GPU,
swap `pytorch-cuda=11.8` for `cpuonly` first.

## Data

`data/CEU/` holds the CEU HapMap cohort used in the paper: 164 individuals over
the 100,000 SNPs the experiments evaluate, taken in the dataset's original
ordering and then sorted by ascending MAF at load time. `Beacon_164.txt` is the
genotype table, `reference.pickle` the homozygous-major reference used to turn
it into a binary carrier matrix, and `MAF.txt` the allele frequencies.

The first run caches the parsed matrix to `binary_cache_CEU.joblib`. Delete that
file if you change the contents of `data/CEU`.

## Running the experiments

Each driver calls `eval_collusion.py` once per cell and writes one
`eval_results.npy` per cell under `results/`. Re-running skips cells that are
already done. Drivers set `DEBUG_COLLUSION_CONTINUE=1` and `BEACON_BINARY=0`
themselves, so episodes run the full budget and responses are continuous. Set
`BEACON_PY` to pick the interpreter for the workers.

### Main experiment

N = 40, 400 episodes, seeds 1-3, 100 queries per account, k in {1,2,3,5,7,10},
member and non-member arms.

```bash
python3 run_gbd_fpr.py          # 156 cells -> results/gbd_fpr
python3 run_heuristic_fpr.py    # 468 cells -> results/heuristic_fpr
python3 run_scaled_control.py   #  30 cells -> results/scaled_control
```

`run_scaled_control.py` is the budget-matched control: one account spending the
coalition's whole kB budget, at 200, 300, 500, 700 and 1000 queries. Budget 100,
the k = 1 point, comes from `results/gbd_fpr/single_k1_*`.

`run_heuristic_fpr.py` reads `HF_EPISODES`, `HF_SEEDS`, `HF_DEFENSES` and
`HF_PARALLEL` from the environment for shorter runs.

### Beacon size

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

1,080 cells. Each worker holds the genotype matrix, so keep `--jobs` within
available RAM. The N = 63 block alone is 360 cells and took about 21 hours at
`--jobs 36`.

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

Each cell stores per-episode identification flags, query counts and, with
`--log_scores`, the LRT trajectories.

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

The strategy flags are `--majority_voting` (S1), `--random_partitioning` (S2a),
`--k_partitioning` (S2b), `--maf_specialization` (S2c) and
`--strided_partitioning` (S2d). Passing none of them gives a single account.

The beacon, both control sets and one spare slot must fit in the 164-person
panel, so `beacon_size <= 164 - 1 - a_control_size - b_control_size`, which is
63 at the defaults. A configuration that does not fit raises an error rather
than silently shrinking the control sets.

## License

GPL-3.0, see `LICENSE`.
