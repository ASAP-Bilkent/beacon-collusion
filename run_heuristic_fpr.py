#!/usr/bin/env python3
"""Random, Baseline and Strategic defenses, same protocol as run_gbd_fpr.py.
468 cells -> results/heuristic_fpr.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.abspath(__file__))
PY   = os.environ.get("BEACON_PY", sys.executable)
OUT  = os.path.join(REPO, "results", "heuristic_fpr")

EPISODES = int(os.environ.get("HF_EPISODES", "400"))
SEEDS    = [int(s) for s in os.environ.get("HF_SEEDS", "1,2,3").split(",")]
KS       = [2, 3, 5, 7, 10]
DEFENSES = os.environ.get("HF_DEFENSES", "random,baseline,strategic").split(",")
PARALLEL = int(os.environ.get("HF_PARALLEL", "6"))

STRATS = {"S1":  ["--majority_voting"],
          "S2a": ["--random_partitioning"],
          "S2b": ["--k_partitioning"],
          "S2c": ["--maf_specialization"],
          "S2d": ["--strided_partitioning"]}


def cells():
    for defense in DEFENSES:
        for sd in SEEDS:
            for prob in (1.0, 0.0):
                yield (defense, "single", 1, (), sd, prob)
                for k in KS:
                    for name, fl in STRATS.items():
                        yield (defense, name, k, tuple(fl), sd, prob)


def run(job):
    defense, strat, k, flags, seed, prob = job
    tag = f"{defense}_{strat}_k{k}_seed{seed}_{'mem' if prob == 1.0 else 'non'}"
    out = os.path.join(OUT, tag)
    if os.path.exists(os.path.join(out, "eval_results.npy")):
        return f"SKIP {tag}"
    cmd = [PY, "eval_collusion.py",
           "--gbd_checkpoint", "./weights/GBD", "--gba_checkpoint", "./weights/GBA.pth",
           "--beacon_type", defense, "--collusion_k", str(k),
           "--queries_per_account", "100", "--n_episodes", str(EPISODES),
           "--seed", str(seed), "--collusion_mode", "round_robin",
           "--gene_size", "100000", "--log_scores",
           "--presence_aware_response", "--private_account_evidence",
           "--individual_local", *list(flags),
           "--victim_prob", str(prob), "--results_dir", out]
    env = dict(os.environ, DEBUG_COLLUSION_CONTINUE="1", BEACON_BINARY="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return (f"FAIL {tag}\n{r.stderr[-400:]}" if r.returncode else f"done {tag}")


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted(set(cells()), key=lambda j: (j[0], j[2]))
    total = len(jobs)
    print(f"heuristic defense sweep: {total} runs, defenses={DEFENSES}, "
          f"episodes={EPISODES}, seeds={SEEDS} -> {OUT}\n", flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        for fut in as_completed(ex.submit(run, j) for j in jobs):
            n += 1
            res = fut.result()
            if "FAIL" in res or n % 10 == 0:
                print(f"[{n}/{total}] {res}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
