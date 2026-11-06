#!/usr/bin/env python3
"""GBD defense, five strategies, both arms. 156 cells -> results/gbd_fpr.
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.abspath(__file__))
PY   = os.environ.get("BEACON_PY", sys.executable)
OUT  = os.path.join(REPO, "results", "gbd_fpr")
EPISODES = 400
SEEDS    = [1, 2, 3]
KS       = [2, 3, 5, 7, 10]
STRATS = {"S1":  ["--majority_voting"],
          "S2a": ["--random_partitioning"],
          "S2b": ["--k_partitioning"],
          "S2c": ["--maf_specialization"],
          "S2d": ["--strided_partitioning"]}

def cells():
    for sd in SEEDS:
        for prob in (1.0, 0.0):
            yield ("single", 1, (), sd, prob)
            for k in KS:
                for name, fl in STRATS.items():
                    yield (name, k, tuple(fl), sd, prob)

def run(job):
    strat, k, flags, seed, prob = job
    tag = f"{strat}_k{k}_seed{seed}_{'mem' if prob==1.0 else 'non'}"
    out = os.path.join(OUT, tag)
    if os.path.exists(os.path.join(out, "eval_results.npy")):
        return f"SKIP {tag}"
    cmd = [PY, "eval_collusion.py",
           "--gbd_checkpoint", "./weights/GBD", "--gba_checkpoint", "./weights/GBA.pth",
           "--beacon_type", "agent", "--collusion_k", str(k),
           "--queries_per_account", "100", "--n_episodes", str(EPISODES),
           "--seed", str(seed), "--collusion_mode", "round_robin",
           "--gene_size", "100000", "--log_scores",
           "--presence_aware_response", "--private_account_evidence",
           "--individual_local", *list(flags),
           "--victim_prob", str(prob), "--results_dir", out]
    env = dict(os.environ, DEBUG_COLLUSION_CONTINUE="1", BEACON_BINARY="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return (f"FAIL {tag}" if r.returncode else f"done {tag}")

def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted(set(cells()), key=lambda j: j[1])     # small k first
    print(f"GBD + H0 arm: {len(jobs)} runs, {EPISODES} episodes, seeds {SEEDS} -> {OUT}",
          flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed(ex.submit(run, j) for j in jobs):
            n += 1; res = f.result()
            if "FAIL" in res or n % 5 == 0:
                print(f"[{n}/{len(jobs)}] {res}", flush=True)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    sys.exit(main())
