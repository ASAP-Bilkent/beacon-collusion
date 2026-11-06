#!/usr/bin/env python3
"""One account spending the whole kB budget against GBD, budgets 200-1000.
30 cells -> results/scaled_control.
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.abspath(__file__))
PY   = os.environ.get("BEACON_PY", sys.executable)
OUT  = os.path.join(REPO, "results", "scaled_control")
BUDGETS = [200, 300, 500, 700, 1000]      # <-> k = 2,3,5,7,10
SEEDS   = [1, 2, 3]
EPISODES = 400

def run(job):
    q, seed, prob = job
    tag = f"q{q}_seed{seed}_{'mem' if prob==1.0 else 'non'}"
    out = os.path.join(OUT, tag)
    if os.path.exists(os.path.join(out, "eval_results.npy")):
        return f"SKIP {tag}"
    cmd = [PY, "eval_collusion.py",
           "--gbd_checkpoint", "./weights/GBD", "--gba_checkpoint", "./weights/GBA.pth",
           "--beacon_type", "agent", "--collusion_k", "1",
           "--queries_per_account", str(q), "--n_episodes", str(EPISODES),
           "--seed", str(seed), "--collusion_mode", "round_robin",
           "--gene_size", "100000", "--log_scores",
           "--presence_aware_response", "--private_account_evidence",
           "--individual_local",
           "--victim_prob", str(prob), "--results_dir", out]
    env = dict(os.environ, DEBUG_COLLUSION_CONTINUE="1", BEACON_BINARY="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return (f"FAIL {tag}" if r.returncode else f"done {tag}")

def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted([(q, s, p) for q in BUDGETS for s in SEEDS for p in (1.0, 0.0)],
                  key=lambda j: j[0])
    print(f"scaled single-attacker control + H0 arm: {len(jobs)} runs -> {OUT}", flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for f in as_completed(ex.submit(run, j) for j in jobs):
            n += 1; res = f.result()
            if "FAIL" in res or n % 4 == 0: print(f"[{n}/{len(jobs)}] {res}", flush=True)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    sys.exit(main())
