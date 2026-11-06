#!/usr/bin/env python3
"""Sweep beacon size for the coalition and the budget-matched single account.

Beacon plus both control sets plus one spare slot must fit the 164-person
panel, so N <= 63 at the default control sizes.
"""

import argparse
import itertools
import os
import subprocess
import sys
import time

PANEL = 164


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 40, 63])
    p.add_argument("--defenses", nargs="+", default=["baseline", "agent"],
                   choices=["agent", "baseline", "strategic", "random"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--arms", nargs="+", default=["mem", "non"],
                   choices=["mem", "non"])
    p.add_argument("--strategies", nargs="+",
                   default=["s1", "s2a", "s2b", "s2c", "s2d", "single"],
                   choices=["s1", "s2a", "s2b", "s2c", "s2d", "single"])
    p.add_argument("--collusion_ks", nargs="+", type=int, default=[2, 3, 5, 7, 10])
    p.add_argument("--queries_per_account", default=100, type=int)
    p.add_argument("--n_episodes", default=400, type=int)
    p.add_argument("--a_control_size", default=50, type=int)
    p.add_argument("--b_control_size", default=50, type=int)
    p.add_argument("--gene_size", default=100000, type=int)
    p.add_argument("--data", default="./data/CEU", type=str)
    p.add_argument("--gbd_checkpoint", default="./weights/GBD", type=str)
    p.add_argument("--gba_checkpoint", default="./weights/GBA.pth", type=str)
    p.add_argument("--out", default="./results/beacon_size", type=str)
    p.add_argument("--python", default=sys.executable, type=str)
    p.add_argument("--jobs", default=1, type=int,
                   help="Concurrent runs. Each holds the genotype matrix, "
                        "about 5 GB, so keep jobs * 5 GB under available RAM.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


# "single" is not a partition: one account spending the whole coalition budget.
STRATEGY_FLAG = {
    "s1":  ["--majority_voting"],
    "s2a": ["--random_partitioning"],
    "s2b": ["--k_partitioning"],
    "s2c": ["--maf_specialization"],
    "s2d": ["--strided_partitioning"],
}


def build(args, size, defense, seed, arm, strategy, collusion_k):
    """Return (tag, command) for one cell."""
    tag = f"N{size}_{defense}_{strategy}_k{collusion_k}_seed{seed}_{arm}"
    results_dir = os.path.join(args.out, tag)

    total_budget = collusion_k * args.queries_per_account
    if strategy == "single":
        # budget-matched to this k
        k, qpa, extra = 1, total_budget, []
    else:
        k, qpa, extra = collusion_k, args.queries_per_account, STRATEGY_FLAG[strategy]

    cmd = [
        args.python, "eval_collusion.py",
        "--data", args.data,
        "--gene_size", str(args.gene_size),
        "--gbd_checkpoint", args.gbd_checkpoint,
        "--gba_checkpoint", args.gba_checkpoint,
        "--beacon_type", defense,
        "--beacon_size", str(size),
        "--a_control_size", str(args.a_control_size),
        "--b_control_size", str(args.b_control_size),
        "--collusion_mode", "round_robin",
        "--collusion_k", str(k),
        "--queries_per_account", str(qpa),
        "--n_episodes", str(args.n_episodes),
        "--seed", str(seed),
        "--private_account_evidence",
        "--individual_local",
        "--log_scores",
        # without the gate, absent alleles get a perturbed answer too
        "--presence_aware_response",
        "--results_dir", results_dir,
    ] + extra

    if arm == "non":
        cmd.append("--calibrated_null")

    return tag, results_dir, cmd


def main():
    args = parse_args()

    ceiling = PANEL - 1 - args.a_control_size - args.b_control_size
    too_big = [n for n in args.sizes if n > ceiling]
    if too_big:
        sys.exit(
            f"Beacon sizes {too_big} do not fit: with a_control_size="
            f"{args.a_control_size} and b_control_size={args.b_control_size} the "
            f"largest beacon the {PANEL}-person panel supports is {ceiling}. "
            f"Lower the control sizes to go higher."
        )

    cells = list(itertools.product(args.sizes, args.defenses, args.strategies,
                                   args.collusion_ks, args.seeds, args.arms))
    planned = [build(args, s, d, sd, a, st, k) for s, d, st, k, sd, a in cells]
    pending = [c for c in planned if not os.path.exists(os.path.join(c[1], "eval_results.npy"))]

    print(f"{len(planned)} cells, {len(planned) - len(pending)} already done, "
          f"{len(pending)} to run")
    for n in args.sizes:
        print(f"  N={n}: coverage {args.n_episodes / n:.1f} passes over "
              f"{n} members, {args.n_episodes / args.b_control_size:.1f} over "
              f"{args.b_control_size} non-members")

    if args.dry_run:
        for tag, _, cmd in pending:
            print(f"\n# {tag}\n{' '.join(cmd)}")
        return

    os.makedirs(args.out, exist_ok=True)

    # longest cells first
    def cost(tag):
        k = int(tag.split("_k")[1].split("_")[0])
        return (0, -k) if "_single_" in tag else (1, k)

    pending.sort(key=lambda c: cost(c[0]))

    failed = []
    done = 0
    running = []          # (tag, log_path, popen, t0)
    queue = list(pending)

    while queue or running:
        while queue and len(running) < args.jobs:
            tag, results_dir, cmd = queue.pop(0)
            os.makedirs(results_dir, exist_ok=True)
            log_path = os.path.join(results_dir, "run.log")
            log = open(log_path, "w")
            # full budget, no early stop
            cell_env = dict(os.environ, DEBUG_COLLUSION_CONTINUE="1",
                            BEACON_BINARY="0")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=cell_env)
            proc._log = log
            running.append((tag, log_path, proc, time.time()))
            print(f"start  {tag}  ({len(running)} running, {len(queue)} queued)", flush=True)

        time.sleep(2)

        for entry in list(running):
            tag, log_path, proc, t0 = entry
            rc = proc.poll()
            if rc is None:
                continue
            running.remove(entry)
            proc._log.close()
            done += 1
            dt = (time.time() - t0) / 60
            if rc != 0:
                failed.append(tag)
                print(f"FAILED {tag}  rc={rc} after {dt:.1f} min, see {log_path}",
                      flush=True)
            else:
                print(f"done   {tag}  {dt:.1f} min  [{done}/{len(pending)}]", flush=True)

    print(f"\n{len(pending) - len(failed)}/{len(pending)} succeeded")
    if failed:
        print("failed cells:")
        for tag in failed:
            print(f"  {tag}")
        sys.exit(1)


if __name__ == "__main__":
    main()
