#!/usr/bin/env python3
"""Evaluate a coalition of beacon accounts against a frozen defense.

Each of k accounts runs the same pre-trained GBA policy over its own slice of
the MAF-ordered SNP space, keeps its own LRT accumulator and control set, and
queries in round-robin order within a per-account budget. The coalition
declares identification when any one account reaches p < 0.05; S1 instead
requires a majority.

The partitioning is chosen with one of --majority_voting (S1),
--random_partitioning (S2a), --k_partitioning (S2b), --maf_specialization
(S2c) or --strided_partitioning (S2d). With none of them set the run is a
single account.

Example:
    python eval_collusion.py \\
        --gbd_checkpoint ./weights/GBD --gba_checkpoint ./weights/GBA.pth \\
        --beacon_type agent --collusion_mode round_robin --collusion_k 10 \\
        --queries_per_account 100 --n_episodes 400 --seed 1 \\
        --gene_size 100000 --random_partitioning \\
        --private_account_evidence --individual_local \\
        --results_dir ./results/S2a_k10
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import os
import random
import torch
import joblib
import copy
from datetime import datetime
from collections import defaultdict

from env import Env
from td import TD3
from ppo import PPO


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(data_path: str, gene_size: int):
    cache_path = "./binary_cache.joblib"
    if os.path.exists(cache_path):
        print("Loading cached genomic data...")
        binary = joblib.load(cache_path)
    else:
        print("Loading genomic data from files...")
        beacon = pd.read_csv(
            os.path.join(data_path, "Beacon_164.txt"), index_col=0, sep=r"\s+"
        )
        reference = pickle.load(
            open(os.path.join(data_path, "reference.pickle"), "rb")
        )
        binary = np.logical_and(
            beacon.values != reference, beacon.values != "NN"
        ).astype(int)
        joblib.dump(binary, cache_path)

    maf = pd.read_csv(
        os.path.join(data_path, "MAF.txt"), index_col=0, sep=r"\s+"
    )
    maf.rename(
        columns={
            'referenceAllele': 'major',
            'referenceAlleleFrequency': 'major_freq',
            'otherAllele': 'minor',
            'otherAlleleFrequency': 'minor_freq',
        },
        inplace=True,
    )
    maf["maf"] = np.round(maf["maf"].values, 3)
    maf_values = maf["maf"].values

    if binary.shape[0] != 164 and binary.shape[1] == 164:
        binary = binary.T

    print(f"Data shape: {binary.shape[0]} individuals x {binary.shape[1]} SNPs")
    return maf_values, binary


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Colluding GBA vs Frozen GBD"
    )
    # Data
    p.add_argument("--data", default="./data/CEU", type=str)
    p.add_argument("--gene_size", default=100000, type=int)

    # Checkpoints
    p.add_argument(
        "--gbd_checkpoint", required=True, type=str,
        help="Path to frozen GBD (TD3) checkpoint directory"
    )
    p.add_argument(
        "--gba_checkpoint", required=True, type=str,
        help="Path to trained GBA (PPO) .pth file"
    )

    # Evaluation parameters
    p.add_argument(
        "--n_episodes", default=400, type=int,
        help="Total evaluation episodes (paper uses 400; >= beacon_size for full victim coverage)"
    )
    p.add_argument(
        "--queries_per_account", default=1000, type=int,
        help="Queries per account per episode (paper uses 1000). "
             "Total episode budget = queries_per_account * (collusion_k + num_benign). "
             "Overridden by --max_queries if provided explicitly."
    )
    p.add_argument(
        "--max_queries", default=None, type=int,
        help="Override total max queries per episode. If omitted, computed as "
             "queries_per_account * (collusion_k + num_benign)."
    )
    p.add_argument("--seed", default=42, type=int)

    # Environment
    p.add_argument("--beacon_size", default=40, type=int)
    p.add_argument("--a_control_size", default=50, type=int)
    p.add_argument("--b_control_size", default=50, type=int)
    p.add_argument("--collusion_k", default=3, type=int)
    p.add_argument(
        "--collusion_mode", default="round_robin",
        choices=["none", "round_robin"]
    )
    p.add_argument("--victim_prob", default=1.0, type=float)
    p.add_argument("--user_risk", default=0.2, type=float)
    # Score logging retains raw p-values plus continuous LRT margins for offline
    # calibration.
    p.add_argument("--log_scores", action="store_true")
    # Per-query LRT attribution: log (attacker_id, MAF, response, victim_genotype)
    # for every attacker query so the per-query evidence can be decomposed offline
    # (explainability: which SNPs/categories drive identification).
    p.add_argument("--log_attribution", action="store_true")
    # Ablation: restrict the attacker's action space to a single MAF category
    # (-1 = off). Used to test category-specialization vs account-splitting.
    p.add_argument("--restrict_category", default=-1, type=int)
    # Rarity targeting: only query SNPs with MAF <= this cap (<=0 disables).
    # Common SNPs are carried by some beacon member regardless of the target.
    p.add_argument("--max_maf", default=-1.0, type=float)
    # Informed attacker: always query loci the target carries, from the rare
    # band. Query selection only; partitioning, pooling and coalition
    # structure are unchanged.
    p.add_argument("--informed_attacker", action="store_true")
    p.add_argument("--executable_attacker", action="store_true",
                   help="Attacker sees only a sampled yes/no answer (one-directional "
                        "flipping) and scores it with an aggregate honesty rate. This is "
                        "the realizable statistic; Eq.4 by contrast consumes h*x_true.")
    p.add_argument("--qbudget_p", type=float, default=0.1,
                   help="Query Budget strength: per-account budget = -log(p). "
                        "Larger p = tighter budget = stronger defense (default 0.1).")
    p.add_argument("--exec_lrt_uses_response", action="store_true",
                   help="Both query-selection state and local scoring consume "
                        "the SAMPLED yes/no answer instead of h*x_true. Requires "
                        "--executable_attacker. Defender state/reward are unchanged.")
    p.add_argument("--exec_absent_yes", action="store_true",
                   help="Beacon answers YES to an ABSENT allele (manuscript's "
                        "Random-Defense semantics), so absence carries no signal. "
                        "Default: absent -> NO, which makes absence conclusive.")
    p.add_argument("--exec_h_bar", type=float, default=None,
                   help="FIXED aggregate honesty rate for the executable attacker's "
                        "no-term. Must be set: deriving it from the attacker's own "
                        "queries uses per-answer h against this victim, which is "
                        "privileged and leaks membership on its own (AUC 0.27).")
    p.add_argument("--repeat_query", type=int, default=0,
                   help="S3 collusion: spend this many queries on each chosen locus "
                        "before moving on. Defeats randomised response, since one YES "
                        "is conclusive. 0 disables (partition-style, no overlap).")
    # Calibrated H0: draw the non-member victim from the defense-control pool
    # (exchangeable with the attacker controls) so the empirical p-value has a
    # valid null. Use for a proper false-positive-rate measurement.
    p.add_argument("--calibrated_null", action="store_true")
    # Per-account evidence: every account gets a private copy of the victim
    # genome and only the querying account consumes the response. Without this
    # flag all accounts share one victim tensor, so the first account to query
    # a locus zeroes it for its peers.
    p.add_argument("--private_account_evidence", action="store_true")
    # Causal test: gate GBD's honesty value on whether the queried allele is
    # actually present in the beacon, so an absent allele yields a truthful "no".
    p.add_argument("--presence_aware_response", action="store_true")
    p.add_argument("--partition_state", action="store_true",
                   help="Defender state includes per-MAF-block risk features "
                        "(18 -> 31 dims). Required to load a checkpoint trained "
                        "with --partition_state.")
    p.add_argument("--min_block_queries", type=int, default=20,
                   help="Queries a block needs before its p-value is trusted.")
    p.add_argument("--per_account_defender", action="store_true",
                   help="Compute GBD observations from each physical account's own history.")
    p.add_argument("--single_physical_account", action="store_true",
                   help="Route all logical attacker slots through one physical account; "
                        "keep their policies, internal histories, schedule and pooling identical.")
    p.add_argument("--physical_accounts", type=int, default=None,
                   help="Route one shared GBA policy through this many physical IDs. "
                        "Requires collusion_k=1 and an explicit total max_queries.")
    p.add_argument("--account_scheduler", choices=["round_robin", "history"],
                   default="round_robin")

    # Adaptive (k-conditioned) checkpoint support
    p.add_argument(
        "--k_conditioned", action="store_true",
        help="Set when evaluating a checkpoint trained with train_adaptive_attacker.py. "
             "Adds normalised k to the attacker state (state_dim 10 → 11)."
    )

    # Mixed population (benign users)
    p.add_argument(
        "--num_benign", default=0, type=int,
        help="Number of benign (non-attacking) users in the simulation. "
             "0 = pure collusion, >0 = mixed population scenario."
    )
    p.add_argument(
        "--benign_query_mode", default="mixed",
        choices=["random", "common", "rare", "mixed"],
        help="Query pattern for benign users: random (uniform), common (high-MAF), "
             "rare (low-MAF), mixed (70%% common + 30%% rare, default)."
    )

    # Coordination mechanisms (evaluation-only; no effect on training)
    p.add_argument(
        "--maf_specialization", action="store_true",
        help="Assign each attacker exclusive MAF categories to eliminate duplicate queries "
             "(e.g. k=2: attacker 0 gets categories 0-2, attacker 1 gets 3-5)"
    )
    p.add_argument(
        "--maf_concentration", action="store_true",
        help="S2f: every attacker restricted to ONE MAF category (see "
             "--maf_concentration_category) for the whole episode -- unlike "
             "--maf_specialization (S2c), which spreads the coalition across all 6 "
             "categories and so dilutes coverage of whichever one actually matters as k "
             "grows. No per-account SNP partition: all accounts draw from the shared "
             "category pool, deduplicated via --private_account_evidence. Only routes "
             "GBA's own 12 native actions; no checkpoint change, no ground truth."
    )
    p.add_argument(
        "--strategic_rare_band", action="store_true",
        help="Strategic defense sensitivity arm: keep the same 5%% suppression "
             "budget but suppress the rarest alleles PRESENT in the beacon "
             "instead of Wan et al.'s delta-discriminative-power ranking, which "
             "at beacon_size=40 selects 70.5%% common variants (MAF>0.2). Tests "
             "whether results against Strategic depend on it guarding the wrong "
             "band. Default off = published behaviour, bit-identical to prior runs."
    )
    p.add_argument(
        "--baseline_target_snps", type=int, default=0,
        help="Baseline defense sensitivity knob: size the suppressed set to "
             "approximately this many SNPs (largest set of whole distinct MAF "
             "values, ascending, skipping the MAF==0 bin, that fits). Default 0 "
             "keeps the original behaviour (lowest 10%% of distinct MAF values = "
             "13,048 SNPs on CEU/100k). Use ~6500 for a '10%% of polymorphic "
             "SNVs' reading of the base paper's stated k_b=10 rule."
    )
    p.add_argument(
        "--maf_concentration_sharded", action="store_true",
        help="S2i: like --maf_concentration (category 0, rare band) but EVERY "
             "attacker additionally gets its own disjoint 1/k slice of category "
             "0's 43,350 SNPs (reuses --k_partitioning's block mechanism, scoped "
             "to category 0's index range) instead of a shared pool. Deliberately "
             "reintroduces mid-episode exhaustion so a single account can only "
             "extract a coverage-limited fraction of the rare-locus signal -- "
             "meant to produce gradual, coupon-collector-style scaling with k "
             "instead of --maf_concentration's near-immediate saturation."
    )
    p.add_argument(
        "--maf_concentration_category", type=int, default=0, choices=range(6),
        metavar="0-5",
        help="Which MAF category --maf_concentration targets (default 0, MAF<0.03). "
             "The right choice is DEFENSE-specific, not universal: verified against "
             "--beacon_type baseline, category 0 is a dead zone there (80.7%% of it is "
             "MAF==0 loci with zero true carriers in the panel; the rest falls entirely "
             "inside Baseline's own suppressed decile, which answers 'no' unconditionally "
             "and so cannot discriminate membership at all) -- category 2 (MAF 0.1-0.2) is "
             "100%% honest and 100%% usable against Baseline. Check the target defense's "
             "own suppression rule before assuming category 0 is the right pick."
    )
    p.add_argument(
        "--portfolio_targeting", action="store_true",
        help="S2g: no single fixed category is right for every defense -- GBD and "
             "Baseline benefit from concentrating on one band (see "
             "--maf_concentration), Random and Strategic are HURT by it (verified: "
             "single-account AUC 0.617->0.521/0.518 for Random, 0.692->0.575/0.443 "
             "for Strategic, restricting to category 1 or 2). Instead of picking "
             "one, split the coalition round-robin by attacker_id across three "
             "targeting policies: unrestricted (preserves Random/Strategic), "
             "category 0 (GBD's own concentrated dishonesty), category 2 (Baseline's "
             "clean band). Report the POOLED statistic, not any-one -- pooling is "
             "what lets a well-matched sub-group's signal survive alongside "
             "mismatched sub-groups without collapsing."
    )
    p.add_argument(
        "--snp_priority", action="store_true",
        help="Enable SNP priority queue: prefer SNPs that gave high LRT gain in past episodes. "
             "Priority accumulates across episodes via exponential moving average."
    )
    p.add_argument(
        "--virtual_sharding", action="store_true",
        help="Enable Virtual Sharding: when multiple attackers share a MAF category (k > 6), "
             "divide the ranked SNP list into disjoint strided subsets — attacker rank r takes "
             "every stride-th SNP starting at offset r. Only activates when ≥2 categories are "
             "shared (natural zeroing handles stride-1 cases). Combine with "
             "--maf_specialization and --snp_priority for best results."
    )
    p.add_argument(
        "--snp_reservation", action="store_true",
        help="Enable Shared SNP Reservation Board: before querying, each attacker checks which "
             "SNPs peers have reserved and skips them. After selecting a SNP, it posts a "
             "reservation with TTL=k (one full round-robin cycle). Prevents intra-category "
             "query collisions without requiring static assignment. Most effective at k=7 "
             "where only one category is shared and virtual sharding is inactive."
    )
    p.add_argument(
        "--k_partitioning", action="store_true",
        help="Enable k-adaptive SNP partitioning: divide all SNPs (sorted by MAF) into k "
             "equal contiguous blocks and assign one unique block per attacker. Replaces "
             "--maf_specialization — guarantees zero cross-attacker SNP overlap at any k "
             "without retraining. Attacker 0 gets the lowest-MAF block, attacker k-1 the "
             "highest. Combine with --snp_priority for best results."
    )
    p.add_argument(
        "--fisher_combined", action="store_true",
        help="Enable Fisher's combined p-value test: each attacker computes its own p-value "
             "independently, then the coalition combines them via χ²=-2·Σln(pᵢ) with 2k "
             "degrees of freedom. Identification fires when the combined test reaches α=0.05, "
             "even if no individual attacker crossed the threshold alone."
    )
    p.add_argument(
        "--fisher_local", action="store_true",
        help="Enable Fisher's combined test using per-partition LOCAL p-values. Each attacker "
             "independently tracks only its own queries' LRT contributions (different SNPs per "
             "partition → genuinely independent tests). Combined via χ²=-2·Σln(pᵢ) with 2k df. "
             "Identification fires when the coalition's combined evidence reaches α=0.05, even if "
             "no individual attacker crossed the threshold. Requires --k_partitioning. "
             "No private_controls needed — independence comes from partition disjointness. "
             "No retraining required."
    )
    p.add_argument(
        "--ebm_local", action="store_true",
        help="Empirical Brown's Method (Poole et al. 2016) using per-partition LOCAL "
             "p-values. Same setup as --fisher_local, but the attackers share the 50 "
             "control individuals, so their local p-values are weakly DEPENDENT and raw "
             "Fisher (χ² with 2k df) is anti-conservative. EBM rescales the combined "
             "statistic to c·χ²(2f), estimating the covariance of the -2·ln(ecdf) "
             "transforms empirically across the shared control panel. Reduces to plain "
             "Fisher when the estimated covariance is non-positive (never more significant "
             "than Fisher). Requires k-partitioning. No retraining required."
    )
    p.add_argument(
        "--private_controls", action="store_true",
        help="Give each attacker a non-overlapping subset of the 50 attack controls. "
             "Creates genuinely independent p-values across attackers so Fisher combination "
             "and the union of individual tests are statistically meaningful. "
             "p-value denominator stays at a_control_size=50; the smaller per-attacker pool "
             "makes the test more sensitive to marginal episodes. No retraining required."
    )
    p.add_argument(
        "--adaptive_schedule", action="store_true",
        help="Enable focus-fire scheduling: use round-robin during exploration, but once "
             "any attacker's p-value drops below 0.2, concentrate all subsequent queries "
             "on that attacker to push it over the 0.05 identification threshold. "
             "Falls back to round-robin if the leader stalls (p rises back above 0.2). "
             "No retraining required."
    )
    p.add_argument(
        "--talking", action="store_true",
        help="Enable explicit attacker-to-attacker coordination via a shared message board. "
             "After each query the active attacker broadcasts its current p-value, "
             "per-category avg LRT gain, per-category avg beacon honesty, and suspicion trend. "
             "Standalone (without --private_controls): peers suppress GBD-suppressed categories. "
             "Combined with --private_controls: each attacker has an independent p-value baseline; "
             "the scheduler reads communicated p-values from the board and concentrates queries on "
             "the attacker closest to the 0.05 threshold (talking-informed focus-fire). "
             "This is the recommended combined mode — independent evidence + explicit communication "
             "+ collective resource allocation. No retraining required."
    )
    p.add_argument(
        "--microburst", action="store_true",
        help="Enable Queue-Health Microburst Scheduling: give short bursts of --burst_b turns "
             "to the attacker whose partition has the highest auc_B (sum of top-B SNP priority "
             "scores). After each burst, enforce a cooldown of (k-1) allocation rounds to prevent "
             "monopolisation. Falls back to round-robin until the SNP priority queue has data. "
             "Requires --k_partitioning and --snp_priority. No retraining required."
    )
    p.add_argument(
        "--burst_b", default=4, type=int,
        help="Turns per microburst (default: 4). Larger values give more focused exploitation "
             "at the cost of less coverage. B=3-5 recommended."
    )
    p.add_argument(
        "--gain_sharing", action="store_true",
        help="Enable Gain-Sharing: each attacker broadcasts its recent average LRT gain to a "
             "shared message board. Peers read these gains and adjust their within-partition "
             "SNP ordering — biasing toward the MAF end that peers found most discriminative. "
             "Genuine runtime collusion: no turn redirection (stays round-robin), purely "
             "within-partition SNP preference shifts. Combine with --k_partitioning. "
             "No retraining required."
    )
    p.add_argument(
        "--random_partitioning", action="store_true",
        help="Randomly shuffle all SNPs and assign equal-sized non-overlapping subsets to "
             "each attacker. Each attacker tracks its own local LRT independently. "
             "Combine with --fisher_local for Fisher combination, or use alone for "
             "individual local p-value identification. No retraining required."
    )
    p.add_argument(
        "--individual_local", action="store_true",
        help="Each attacker tracks its own local LRT from only its own queries. "
             "Identification fires when ANY individual attacker's local p-value < 0.05. "
             "No Fisher combination. Works with --maf_specialization (S2c) or "
             "--random_partitioning (S2a without Fisher). No retraining required."
    )
    p.add_argument(
        "--strided_partitioning", action="store_true",
        help="Assign SNPs to attackers in round-robin order across the MAF-sorted list. "
             "Each attacker gets every k-th SNP, giving each a uniform spread of rare, "
             "medium, and common SNPs. Guarantees no overlap. Combine with --fisher_local. "
             "No retraining required."
    )
    p.add_argument(
        "--majority_voting", action="store_true",
        help="Change identification rule from any-one to majority: identification fires "
             "only when MORE than k/2 attackers individually have local p-value < 0.05. "
             "Use with --individual_local for Strategy 1 (same SNPs, majority vote). "
             "No retraining required."
    )
    # Baseline comparison
    p.add_argument(
        "--cold_relay", action="store_true",
        help="Enable Cold-Start Relay + Suspicion Flattening coordination. "
             "Attackers 0..k-2 act as setup agents: turns are routed to whichever "
             "has the lowest GBD-observed risk (suspicion flattening). "
             "Attacker k-1 is reserved as the cold finisher: once any setup attacker "
             "reaches p < 0.15, all remaining turns go to the cold attacker which has "
             "zero accumulated suspicion and receives near-1.0 honesty from GBD for "
             "the decisive final queries. Combine with --maf_specialization for best results."
    )
    p.add_argument(
        "--run_baseline", action="store_true",
        help="Also run single-attacker (k=1) baseline for comparison"
    )

    # Output
    p.add_argument("--results_dir", default="./results/eval_collusion", type=str)

    # Defense type (default: trained GBD agent; use heuristic defenses without retraining)
    p.add_argument(
        "--beacon_type", default="agent",
        choices=["agent", "truth", "honest", "random", "baseline", "strategic", "rtf", "qbudget", "OG-K"],
        help="Defense strategy to evaluate against. 'agent' = trained GBD (default). "
             "Others are heuristic defenses that require no checkpoint."
    )

    args = p.parse_args()
    if args.exec_lrt_uses_response and not args.executable_attacker:
        p.error("--exec_lrt_uses_response requires --executable_attacker")
    if args.physical_accounts is not None:
        if (args.physical_accounts < 1 or args.collusion_k != 1 or args.num_benign
                or args.single_physical_account or args.max_queries is None):
            p.error("Physical routing requires positive accounts, collusion_k=1, "
                    "no benign users/single_physical_account, and explicit max_queries")
        if not (args.executable_attacker and args.exec_lrt_uses_response
                and args.individual_local and args.per_account_defender):
            p.error("Physical routing requires executable_attacker, exec_lrt_uses_response, "
                    "individual_local and per_account_defender")
        if args.exec_h_bar is None or not 0 < args.exec_h_bar < 1:
            p.error("Physical routing requires a fixed exec_h_bar strictly between 0 and 1")
        if os.environ.get("BEACON_BINARY", "0") == "1":
            p.error("Physical routing samples responses; disable deterministic BEACON_BINARY")

    # Compute total max_queries per episode following paper protocol:
    # each of the (k + num_benign) accounts gets queries_per_account turns.
    total_users = args.collusion_k + args.num_benign
    if args.max_queries is None:
        # Pure collusion: k accounts × Q queries each
        # Mixed population: (k + b) accounts × Q queries each
        args.max_queries = args.queries_per_account * max(total_users, 1)

    # Fixed env flags required by Env
    args.attacker_type = "agent"
    # beacon_type comes from argparse; default "agent" preserves original behaviour
    args.beacon_agent = "td"
    args.train = "attacker"
    args.evaluate = True          # deterministic victim cycling + greedy PPO eval
    # The base repo trains with binary=False and evaluates with binary=True. A
    # real GA4GH beacon returns a bit, so binary is the realistic setting.
    # Overridable so both can be measured.
    args.binary = (os.environ.get("BEACON_BINARY", "0") == "1")
    args.collusion_features = False  # match Phase-1 GBD (18-dim state)
    args.print_freq = 9999        # suppress per-step prints
    # k_conditioned comes from argparse; default False = old 10-dim checkpoint
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)
    return args


# ---------------------------------------------------------------------------
# Load Agents
# ---------------------------------------------------------------------------
def load_gbd(checkpoint_dir: str, state_dim: int = 18) -> TD3:
    """Load frozen GBD. 18 dims by default; 31 with partition-aware state."""
    gbd = TD3(state_dim=state_dim, action_dim=1, max_action=1.0)
    gbd.load(checkpoint_dir)
    print(f"  GBD loaded from: {checkpoint_dir}  [FROZEN]")
    return gbd


def load_gba(checkpoint_path: str, k_conditioned: bool = False) -> PPO:
    """Load trained GBA (PPO, discrete action space, 12 actions).

    Args:
        k_conditioned: If True, loads an adaptive checkpoint trained with
                       train_adaptive_attacker.py (state_dim=11 instead of 10).
    """
    state_dim = 11 if k_conditioned else 10
    gba = PPO(
        state_dim=state_dim,
        action_dim=12,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=False,
        action_std_init=None,
    )
    gba.load(checkpoint_path)
    ckpt_type = "adaptive (k-conditioned, state_dim=11)" if k_conditioned else "standard (state_dim=10)"
    print(f"  GBA loaded from: {checkpoint_path}  [{ckpt_type}]")
    return gba


# ---------------------------------------------------------------------------
# Run One Evaluation Pass
# ---------------------------------------------------------------------------
def run_evaluation(args, maf_values, binary, gbd, gba, label="EVAL"):
    """
    Run n_episodes evaluation episodes.

    PPO operates in eval_mode (stochastic sampling, no buffer writes) via the
    args.evaluate=True flag that env.step() passes through to select_action.
    Stochastic sampling is correct here: the PPO policy is a learned distribution
    over 12 actions in a mixed-strategy adversarial game. Greedy argmax collapses
    this to a single pure strategy that underperforms the trained policy.

    Returns a list of per-episode result dicts.
    """
    env = Env(args, maf_values, binary)

    episode_results = []

    for ep in range(args.n_episodes):
        ep_identified = False
        ep_queries_to_id = None
        honesty_values = []
        raw_policy_honesty = []
        beacon_responses = []
        env._min_local_pvals = None   # fresh per-account min-p tracker per episode
        env._min_pooled_pval = None   # fresh coalition pooled min-p tracker per episode
        env._max_local_margin = None  # fresh per-account continuous LRT-margin tracker
        env._final_local_margin = None  # margin after the full budget (single test)
        env._max_pooled_margin = None   # pooled coalition margin (all accounts combined)
        env._attribution = []         # fresh per-query LRT-attribution log per episode
        env._resp_presence = []       # (response, allele-present-in-beacon) per query

        for t in range(args.max_queries):
            _, rewards, done, info = env.step(
                beacon_agent=gbd,
                attacker_agent=gba,
            )

            # info = [pvalue, beacon_action]
            if len(info) >= 2:
                raw_policy_honesty.append(env.last_raw_policy_honesty)
                honesty_val = float(info[1])
                honesty_values.append(honesty_val)
                beacon_responses.append(honesty_val)

            if len(env.identified_attackers) > 0 and not ep_identified:
                ep_identified = True
                ep_queries_to_id = env.current_step  # steps completed (1-indexed)

            if done:
                break

        # Coverage and overlap: of the coalition's queries, how many hit a locus
        # already queried this episode. S2a-S2d re-query only absent-allele loci,
        # while the uncoordinated S1 also collides on rare discriminative loci.
        queried = list(getattr(env, "_queried_loci", []))
        n_total_loci = len(queried)
        n_distinct_loci = len(set(queried))
        n_duplicate_loci = n_total_loci - n_distinct_loci

        episode_results.append({
            "episode": ep + 1,
            "victim_id": env.victim_id,
            "identified": ep_identified,
            "queries_to_id": ep_queries_to_id,
            "total_queries": env.current_step,
            "avg_honesty": float(np.mean(honesty_values)) if honesty_values else float("nan"),
            "avg_raw_policy_honesty": float(np.mean(raw_policy_honesty)),
            "physical_attacker_accounts": (args.physical_accounts if getattr(args, "physical_accounts", None)
                                          else 1 if getattr(args, "single_physical_account", False)
                                          else env.num_attackers),
            "routing_query_counts": (list(env.account_router.counts)
                                     if getattr(env, "account_router", None) else None),
            "defender_account_query_counts": {int(k): v.defender_query_count
                                              for k, v in env.beacon.defender_histories.items()},
            "final_local_victim_lrts": [float(a.local_victim_lrt) for a in env.attackers
                                        if hasattr(a, "local_victim_lrt")],
            "final_local_control_lrts": [a.local_control_lrts.tolist() for a in env.attackers
                                         if hasattr(a, "local_control_lrts")],
            "identifying_attacker": (
                min(env.identified_attackers) if ep_identified else None
            ),
            "min_local_pvals": list(getattr(env, "_min_local_pvals", None) or []),
            "min_pooled_pval": getattr(env, "_min_pooled_pval", None),
            "max_local_margin": list(getattr(env, "_max_local_margin", None) or []),
            # Margin after the full query budget (one test), against the running maximum
            # above (a test after every query).
            "final_local_margin": list(getattr(env, "_final_local_margin", None) or []),
            # Coalition-level statistic: all accounts' evidence combined into a
            # single test, which is what exploits a per-account defense.
            "max_pooled_margin": getattr(env, "_max_pooled_margin", None),
            "final_pooled_margin": getattr(env, "_final_pooled_margin", None),
            # Executable (sampled-response) attacker -- the realizable statistic.
            "final_exec_llr": getattr(env, "_final_exec_llr", None),
            "final_exec_null_mean": getattr(env, "_final_exec_null_mean", None),
            "final_exec_null_var": getattr(env, "_final_exec_null_var", None),
            "final_exec_margin": getattr(env, "_final_exec_margin", None),
            "final_exec_margin_per_account": list(getattr(env, "_final_exec_margin_per_account", None) or []),
            "final_exec_margin_anyone": getattr(env, "_final_exec_margin_anyone", None),
            "run_exec_margin_anyone": getattr(env, "_run_exec_margin_anyone", None),
            "run_exec_margin_per_account": list(getattr(env, "_run_exec_margin_per_account", None) or []),
            "run_exec_margin_pooled": getattr(env, "_run_exec_margin_pooled", None),
            "final_exec_n_loci": getattr(env, "_final_exec_n_loci", None),
            "final_exec_draws": getattr(env, "_final_exec_draws", None),
            "final_exec_h_bar": getattr(env, "_final_exec_h_bar", None),
            "final_pooled_victim_lrt": getattr(env, "_final_pooled_victim_lrt", None),
            "final_pooled_null_mean": getattr(env, "_final_pooled_null_mean", None),
            "final_pooled_null_var": getattr(env, "_final_pooled_null_var", None),
            "final_pooled_n_informative": getattr(env, "_final_pooled_n_informative", None),
            "final_llr_f": getattr(env, "_final_llr_f", None),
            "final_llr_f_mean": getattr(env, "_final_llr_f_mean", None),
            "final_llr_f_var": getattr(env, "_final_llr_f_var", None),
            "final_f_grid": getattr(env, "_final_f_grid", None),
            "budget_snapshots": dict(getattr(env, "_budget_snapshots", None) or {}),
            "attribution": list(getattr(env, "_attribution", None) or []),
            # Ground-truth check: (beacon response, whether any beacon member
            # actually carries the queried allele).
            "resp_presence": list(getattr(env, "_resp_presence", None) or []),
            "distinct_loci": n_distinct_loci,
            "duplicate_loci": n_duplicate_loci,
            "overlap_rate": (n_duplicate_loci / n_total_loci) if n_total_loci else 0.0,
        })

        env.reset()

    return episode_results


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------
def summarise(results, label: str, max_queries: int):
    n = len(results)
    n_identified = sum(r["identified"] for r in results)
    id_rate = n_identified / n

    queries_to_id = [
        r["queries_to_id"] for r in results
        if r["identified"] and r["queries_to_id"] is not None
    ]
    avg_queries = float(np.mean(queries_to_id)) if queries_to_id else float("nan")

    honesty_vals = [r["avg_honesty"] for r in results if not np.isnan(r["avg_honesty"])]
    avg_honesty = float(np.mean(honesty_vals)) if honesty_vals else float("nan")

    # Per-victim breakdown
    by_victim = defaultdict(list)
    for r in results:
        by_victim[r["victim_id"]].append(r["identified"])

    # Per-attacker breakdown (which attacker triggered identification)
    by_attacker = defaultdict(int)
    for r in results:
        if r["identified"] and r["identifying_attacker"] is not None:
            by_attacker[r["identifying_attacker"]] += 1

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Episodes evaluated:        {n}")
    print(f"  Victims identified:        {n_identified} / {n}  ({id_rate:.1%})")
    print(f"  Avg queries to ID:         {avg_queries:.1f}  (budget={max_queries})")
    print(f"  Avg beacon honesty (util): {avg_honesty:.3f}")

    # Which attacker triggered identification
    if by_attacker:
        print(f"\n  Identifications by attacker:")
        for att_id in sorted(by_attacker):
            print(f"    Attacker {att_id}: {by_attacker[att_id]} identifications")

    # Per-victim ID rates
    print(f"\n  Per-victim ID rate (victim_id: identified/total):")
    for vid in sorted(by_victim):
        trials = by_victim[vid]
        v_rate = sum(trials) / len(trials)
        bar = "#" * int(v_rate * 20)
        print(f"    Victim {vid:2d}: {sum(trials):2d}/{len(trials)}  {v_rate:.0%}  {bar}")

    # Queries-to-ID distribution
    if queries_to_id:
        q_arr = np.array(queries_to_id)
        print(f"\n  Queries-to-ID distribution:")
        print(f"    Min:    {q_arr.min():.0f}")
        print(f"    Median: {np.median(q_arr):.0f}")
        print(f"    Mean:   {q_arr.mean():.1f}")
        print(f"    Max:    {q_arr.max():.0f}")
        print(f"    Identified at query   1: {(q_arr == 1).sum()}/{len(q_arr)}")
        print(f"    Identified at query <=5: {(q_arr <= 5).sum()}/{len(q_arr)}")
        print(f"    Identified at query<=10: {(q_arr <= 10).sum()}/{len(q_arr)}")

    print(f"{'='*65}\n")

    return {
        "label": label,
        "n_episodes": n,
        "n_identified": n_identified,
        "id_rate": id_rate,
        "avg_queries_to_id": avg_queries,
        "utility": avg_honesty,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    population_str = f"k={args.collusion_k} attackers"
    if args.num_benign > 0:
        population_str += f" + {args.num_benign} benign ({args.benign_query_mode})"
    total_users = args.collusion_k + args.num_benign

    print("\n" + "=" * 65)
    print("  COLLUSION ATTACKER EVALUATION")
    print("=" * 65)
    print(f"  Defense type:    {args.beacon_type}")
    if args.beacon_type == "agent":
        print(f"  GBD checkpoint:  {args.gbd_checkpoint}  [FROZEN]")
    print(f"  GBA checkpoint:  {args.gba_checkpoint}")
    print(f"  Population:      {population_str}")
    print(f"  Queries/account: {args.queries_per_account}")
    print(f"  Max queries/ep:  {args.max_queries}  ({total_users} users × {args.queries_per_account})")
    print(f"  Episodes:        {args.n_episodes}")
    print(f"  Seed:            {args.seed}")
    print(f"  PPO eval mode:   stochastic sampling, no buffer writes")
    print("=" * 65 + "\n")

    maf_values, binary = load_data(args.data, args.gene_size)

    print("\nLoading agents...")
    gbd = load_gbd(args.gbd_checkpoint, 31 if args.partition_state else 18) if args.beacon_type == "agent" else None
    gba = load_gba(args.gba_checkpoint, k_conditioned=args.k_conditioned)

    start = datetime.now().replace(microsecond=0)

    # -----------------------------------------------------------------------
    # 1. Evaluate colluding GBA (k attackers)
    # -----------------------------------------------------------------------
    print(f"\n[1/2] Running colluding GBA evaluation (k={args.collusion_k})...")
    collusion_results = run_evaluation(
        args, maf_values, binary, gbd, gba,
        label=f"Colluding GBA (k={args.collusion_k}) vs Frozen GBD"
    )
    collusion_summary = summarise(
        collusion_results,
        label=f"COLLUDING GBA (k={args.collusion_k}) vs Frozen Single-Attacker GBD",
        max_queries=args.max_queries,
    )

    # -----------------------------------------------------------------------
    # 2. Baseline: single attacker (k=1), same GBA weights
    # -----------------------------------------------------------------------
    baseline_summary = None
    baseline_results = None

    if args.run_baseline:
        print(f"[2/2] Running single-attacker baseline (k=1, {args.queries_per_account} queries)...")
        baseline_args = copy.copy(args)
        baseline_args.collusion_k = 1
        baseline_args.collusion_mode = "none"
        baseline_args.num_benign = 0
        # k=1 baseline always uses exactly queries_per_account total queries (paper protocol)
        baseline_args.max_queries = args.queries_per_account

        baseline_results = run_evaluation(
            baseline_args, maf_values, binary, gbd, gba,
            label="Single GBA (k=1) vs Frozen GBD"
        )
        baseline_summary = summarise(
            baseline_results,
            label="SINGLE GBA (k=1) vs Frozen Single-Attacker GBD  [BASELINE]",
            max_queries=baseline_args.max_queries,
        )

        # Comparison table
        print("\n" + "=" * 65)
        print("  COMPARISON SUMMARY")
        print("=" * 65)
        print(f"  {'Setting':<35} {'ID Rate':>8}  {'Avg Queries':>12}  {'Utility':>8}")
        print(f"  {'-'*63}")
        print(
            f"  {'Single GBA (k=1)  [baseline]':<35} "
            f"{baseline_summary['id_rate']:>8.1%}  "
            f"{baseline_summary['avg_queries_to_id']:>12.1f}  "
            f"{baseline_summary['utility']:>8.3f}"
        )
        print(
            f"  {'Colluding GBA (k=' + str(args.collusion_k) + ')':<35} "
            f"{collusion_summary['id_rate']:>8.1%}  "
            f"{collusion_summary['avg_queries_to_id']:>12.1f}  "
            f"{collusion_summary['utility']:>8.3f}"
        )
        print("=" * 65)

    elapsed = datetime.now().replace(microsecond=0) - start
    print(f"\nEvaluation completed in {elapsed}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = os.path.join(args.results_dir, "eval_results.npy")
    save_dict = {
        "collusion_results": collusion_results,
        "collusion_summary": collusion_summary,
        "args": vars(args),
    }
    if baseline_results is not None:
        save_dict["baseline_results"] = baseline_results
        save_dict["baseline_summary"] = baseline_summary

    np.save(out_path, save_dict)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
