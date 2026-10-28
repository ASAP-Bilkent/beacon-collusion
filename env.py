import numpy as np
import copy
import os
import csv
import math
import random
import torch
import gym

# Query counts at which the intermediate-budget snapshot is taken (see the
# _budget_snapshots block in step()). Chosen to span the k=10 budget so that
# d(N) can be read with the individual and k held fixed.
_BUDGET_CHECKPOINTS = frozenset({1, 2, 5, 10, 15, 20, 30, 40,
                                 50, 100, 200, 300, 500, 700, 1000})
from gym import spaces

from utils import plot_lists, lrt, create_csv, log_env, log_victim, plot_two_lists, plot_three_lists
from beacon import Beacon
from attacker import Attacker
from helpers.maf_tools import MAFHelper

class MessageBoard:
    """Shared bulletin board for attacker-to-attacker communication (--talking mode).

    Each attacker posts a message dict after its query; peers read these on
    their next turn to adapt their category selection away from GBD-suppressed
    areas and toward high-yield regions.
    """
    def __init__(self, num_attackers: int):
        self._board = {i: None for i in range(num_attackers)}

    def post(self, attacker_id: int, message: dict):
        self._board[attacker_id] = message

    def read_others(self, attacker_id: int):
        return [msg for i, msg in self._board.items()
                if i != attacker_id and msg is not None]

    def read_all(self):
        """Return all posted messages (including own). Used by the scheduler."""
        return [msg for msg in self._board.values() if msg is not None]

    def reset(self, num_attackers: int):
        self._board = {i: None for i in range(num_attackers)}


class ReservationBoard:
    """Shared SNP reservation board for attacker-to-attacker deduplication.

    Before querying, an attacker checks which SNPs are reserved by peers and
    skips them. After selecting a SNP, it posts a reservation with a TTL so
    the reservation expires if the attacker stalls or the SNP is consumed.
    TTL is set to k (one full round-robin cycle) so the reservation lasts
    exactly as long as it takes for the reserving attacker to query again.
    """
    def __init__(self):
        self._reservations: dict[int, tuple[int, int]] = {}  # snp_idx -> (attacker_id, ttl)

    def reserve(self, attacker_id: int, snp_idx: int, ttl: int):
        self._reservations[snp_idx] = (attacker_id, ttl)

    def get_reserved(self, excluding_attacker_id: int) -> set:
        return {snp for snp, (aid, ttl) in self._reservations.items()
                if aid != excluding_attacker_id}

    def tick(self):
        to_remove = [snp for snp, (_, ttl) in self._reservations.items() if ttl <= 1]
        for snp in to_remove:
            del self._reservations[snp]
        for snp in list(self._reservations):
            if snp in self._reservations:
                aid, ttl = self._reservations[snp]
                self._reservations[snp] = (aid, ttl - 1)

    def reset(self):
        self._reservations = {}


class Env():
    def __init__(self, args, maf, binary):
        self.maf = maf[:args.gene_size]
        self.binary = binary[:, :args.gene_size]
        self.args = copy.copy(args)
        self.categorized_maf = None
        
        self.observation_space = None
        self.action_space = None

        if self.args.attacker_type == "agent":
            sorted_args = np.argsort(self.maf)
            self.maf = self.maf[sorted_args]
            self.binary = self.binary[:, sorted_args]

        self.log_beacon = create_csv(self.args.results_dir, "beacon_log")
        self.log_beacon_control = create_csv(self.args.results_dir, "beacon_control_log")
        self.log_attacker = create_csv(self.args.results_dir, "attacker_log")
        self.log_attacker_control = create_csv(self.args.results_dir, "attacker_control_log")

        self.episode = 0
        self._episode_result_printed = False
        print(f"==================================⬇️⬇️⬇️⬇️Episode: {self.episode+1}⬇️⬇️⬇️⬇️==============================")
        print("ATTACKER TYPE: ", self.args.attacker_type)
        print("BEACON TYPE: ", self.args.beacon_type)

        (beacon_case, beacon_control, attack_control, self.victim, self.victim_id,
         mafs) = self.get_populations()

        # Always initialize with num_groups to support switching between attacker types
        num_groups = 99
        self.maf_helper = MAFHelper(mafs, num_groups)

        # Shadow-beacon empirical null. Instead of the analytic
        # p0 = 1 - (1-maf)^(2*beacon_size), resample the beacon composition from
        # the panel excluding the target: for a locus carried by c of the
        # remaining n-1 individuals, the chance that none of the beacon_size
        # sampled carry it is the hypergeometric C(n-1-c, k) / C(n-1, k).
        _n_panel = int(self.binary.shape[0])
        _k_bea = int(self.args.beacon_size)
        _carriers = np.asarray(self.binary, dtype=np.int64).sum(axis=0)
        self._locus_carrier_count = _carriers
        # P(no carrier among k drawn from n-1 individuals) for each c
        _m = _n_panel - 1
        _tab = np.zeros(_m + 1, dtype=np.float64)
        from math import lgamma
        def _logC(a, b):
            if b < 0 or b > a:
                return -np.inf
            return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
        _den = _logC(_m, _k_bea)
        for _c in range(_m + 1):
            _tab[_c] = np.exp(_logC(_m - _c, _k_bea) - _den) if (_m - _c) >= _k_bea else 0.0
        self._shadow_no_carrier = _tab
        for _att in getattr(self, "attackers", []) or []:
            _att.shadow_no_carrier = _tab
            _att.locus_carrier_count = _carriers

        self.beacon = Beacon(self.args, beacon_case, beacon_control, mafs,
                             self.victim_id)
        self._init_spaces()
        
        # --- COLLUSION UPDATE: Init Multiple Attackers ---
        self.attackers = []
        num_attackers = self.args.collusion_k if self.args.collusion_mode != "none" else 1
        if self.args.collusion_k > 1 and self.args.collusion_mode == "none":
            print(f"WARNING: collusion_k={self.args.collusion_k} but collusion_mode='none'. "
                  f"Only 1 attacker will be used. Pass --collusion_mode round_robin to enable collusion.")
        print(f"Initializing {num_attackers} attacker(s) in mode {self.args.collusion_mode}")
        independent = self._uses_independent_consumption(num_attackers)
        self._independent_consumption = independent
        self._queried_loci = []
        self._budget_snapshots = {}
        self._attacker_resp_log = []
        for idx in range(num_attackers):
            attacker = Attacker(self.args, self.victim, attack_control, self.maf_helper)
            attacker.shadow_no_carrier = getattr(self, '_shadow_no_carrier', None)
            attacker.locus_carrier_count = getattr(self, '_locus_carrier_count', None)
            attacker.attacker_id = idx
            self.attackers.append(attacker)
        # Share maf_categories across all attackers (agent type only) so category
        # counts stay in sync when the active attacker zeroes a victim SNP.
        if self.args.attacker_type == "agent" and len(self.attackers) > 1 and not independent:
            shared_cats = self.attackers[0].maf_categories
            for att in self.attackers[1:]:
                att.maf_categories = shared_cats

        # Private control subsets: each attacker gets a non-overlapping slice of
        # the attack_control population, so their p-values are independent.
        self._apply_private_controls(self.attackers, attack_control)

        self.identified_attackers = set()
        self.identify_events = {idx: 0 for idx in range(num_attackers)}

        # --- MIXED POPULATION: Add Benign Users ---
        self.benign_users = []
        num_benign = getattr(self.args, 'num_benign', 0)
        if num_benign > 0:
            from benign_user import BenignUser
            for idx in range(num_benign):
                mode = getattr(self.args, 'benign_query_mode', 'mixed')
                benign = BenignUser(self.args, self.maf_helper, query_mode=mode)
                benign.user_id = num_attackers + idx  # Offset ID to avoid collision
                benign.is_attacker = False
                self.benign_users.append(benign)
            print(f"Initializing {num_benign} benign user(s) with query mode: {mode}")

        # Combined user list for round-robin cycling
        self.all_users = self.attackers + self.benign_users
        self.num_attackers = num_attackers

        # SNP priority queue: maps snp_index -> smoothed LRT gain.
        # Persists across episodes so high-gain SNPs are preferred in future turns.
        self.snp_priority = {}

        # Shared bulletin board, allocated for talking and/or cold_relay.
        self.message_board = (
            MessageBoard(num_attackers)
            if (getattr(self.args, 'talking', False)
                or getattr(self.args, 'cold_relay', False)
                or getattr(self.args, 'gain_sharing', False)) else None
        )

        # SNP reservation board, prevents two attackers from querying the same SNP
        # in the same round. Active when --snp_reservation is set.
        self.reservation_board = (
            ReservationBoard()
            if getattr(self.args, 'snp_reservation', False) else None
        )

        # k-adaptive SNP partitioning: one unique MAF-range block per attacker.
        self._attacker_snp_partitions = None
        if getattr(self.args, 'k_partitioning', False) and num_attackers > 1:
            self._attacker_snp_partitions = self.compute_snp_partitions(
                num_attackers, self.args.gene_size
            )
        # Sharded concentration (S2i): every account is confined to category 0 and
        # further restricted to its own disjoint 1/k slice of that band.
        elif getattr(self.args, 'maf_concentration_sharded', False):
            self._attacker_snp_partitions = self.compute_snp_partitions(
                max(num_attackers, 1), 43350  # category 0 (MAF<0.03) SNP count, Table 1
            )
        # maf_concentration does not partition category 0. Every account draws from
        # the full pool and cross-account duplicates are prevented by consumed_snps
        # and the reservation board.
        elif getattr(self.args, 'random_partitioning', False) and num_attackers > 1:
            self._attacker_snp_partitions = self.compute_random_snp_partitions(
                num_attackers, self.args.gene_size
            )
        elif getattr(self.args, 'strided_partitioning', False) and num_attackers > 1:
            self._attacker_snp_partitions = self.compute_strided_snp_partitions(
                num_attackers, self.args.gene_size
            )
        # Phase 2 flag for cold-start relay: latches True once triggered,
        # stays True for the rest of the episode.
        self._cold_relay_phase2 = False

        # Microburst scheduler state.
        # _burst_B:         turns per burst (configurable via --burst_b)
        # _burst_remaining: turns left in the current burst
        # _burst_owner:     attacker_id holding the current burst
        # _burst_cooldowns: attacker_id -> allocation rounds before eligible again
        self._burst_B         = getattr(self.args, 'burst_b', 4)
        self._burst_remaining = 0
        self._burst_owner     = None
        self._burst_cooldowns = {}

        self.current_step = 0
        self.max_steps = self.args.max_queries

        self.beacon_prewards = []
        self.beacon_urewards = []
        self.beacon_total = []
        self.attacker_prewards = []
        self.attacker_urewards = []
        self.attacker_total = []
        self.attacker_actions = []
        self.attacker_agent_actions = []
        self.beacon_agent_actions = []
        self.ever_identified = False
        self.max_victim_lrt = None

    def reset(self, args=None) -> torch.Tensor:
        if args is not None:
            self.args = copy.copy(args)
            # max_steps is otherwise fixed in __init__, so a per-episode change of
            # collusion_k (and hence max_queries) would be silently capped at the
            # value the env was constructed with.
            self.max_steps = self.args.max_queries

        self.episode += 1
        
        # Clear Logs
        self._run_exec_margin_per_account = None
        self._run_exec_margin_anyone = None
        self._run_exec_margin_pooled = None
        self.beacon_prewards = []
        self.beacon_urewards = []
        self.beacon_total = []
        self.attacker_urewards = []
        self.attacker_prewards = []
        self.attacker_total = []
        self.attacker_actions = []
        self.attacker_agent_actions = []
        self.beacon_agent_actions = []
        self.ever_identified = False
        self.identified_attackers = set()
        self.identify_events = {}
        self.max_victim_lrt = None
        self._episode_result_printed = False

        print(f"==================================⬇️⬇️⬇️⬇️Episode: {self.episode+1}⬇️⬇️⬇️⬇️==============================")
        print("ATTACKER TYPE: ", self.args.attacker_type)

        (beacon_case, beacon_control, attack_control, self.victim, self.victim_id,
         mafs) = self.get_populations()

        self.current_step = 0

        self.beacon = Beacon(self.args, beacon_case, beacon_control, mafs,
                             self.victim_id)
        
        # --- COLLUSION UPDATE: Reset All Attackers ---
        self.attackers = []
        num_attackers = self.args.collusion_k if self.args.collusion_mode != "none" else 1
        independent = self._uses_independent_consumption(num_attackers)
        self._independent_consumption = independent
        self._queried_loci = []
        self._budget_snapshots = {}
        self._attacker_resp_log = []
        for idx in range(num_attackers):
            attacker = Attacker(args=self.args, victim=self.victim, control=attack_control, maf_helper=self.maf_helper)
            attacker.shadow_no_carrier = getattr(self, '_shadow_no_carrier', None)
            attacker.locus_carrier_count = getattr(self, '_locus_carrier_count', None)
            attacker.attacker_id = idx
            self.attackers.append(attacker)
        # Share maf_categories so all attackers stay in sync when any one zeroes a
        # victim SNP. Skipped for independent (S1) collusion.
        if self.args.attacker_type == "agent" and len(self.attackers) > 1 and not independent:
            shared_cats = self.attackers[0].maf_categories
            for att in self.attackers[1:]:
                att.maf_categories = shared_cats

        self._apply_private_controls(self.attackers, attack_control)

        self.identified_attackers = set()
        self.identify_events = {idx: 0 for idx in range(num_attackers)}

        # --- MIXED POPULATION: Reset Benign Users ---
        self.benign_users = []
        num_benign = getattr(self.args, 'num_benign', 0)
        if num_benign > 0:
            from benign_user import BenignUser
            for idx in range(num_benign):
                mode = getattr(self.args, 'benign_query_mode', 'mixed')
                benign = BenignUser(self.args, self.maf_helper, query_mode=mode)
                benign.user_id = num_attackers + idx
                benign.is_attacker = False
                self.benign_users.append(benign)

        # Rebuild combined user list
        self.all_users = self.attackers + self.benign_users
        self.num_attackers = num_attackers

        if self.message_board is not None:
            self.message_board.reset(num_attackers)

        self._cold_relay_phase2 = False

        self._burst_remaining = 0
        self._burst_owner     = None
        self._burst_cooldowns = {}

        if self.reservation_board is not None:
            self.reservation_board.reset()

        if getattr(self.args, 'k_partitioning', False) and num_attackers > 1:
            self._attacker_snp_partitions = self.compute_snp_partitions(
                num_attackers, self.args.gene_size
            )
        # sharded concentration (S2i) -- see the matching comment at __init__
        elif getattr(self.args, 'maf_concentration_sharded', False):
            self._attacker_snp_partitions = self.compute_snp_partitions(
                max(num_attackers, 1), 43350
            )
        # maf_concentration: no per-account partition -- see the matching
        # comment at __init__ for why (narrow slices exhaust mid-episode).
        elif getattr(self.args, 'random_partitioning', False) and num_attackers > 1:
            self._attacker_snp_partitions = self.compute_random_snp_partitions(
                num_attackers, self.args.gene_size
            )
        elif getattr(self.args, 'strided_partitioning', False) and num_attackers > 1:
            self._attacker_snp_partitions = self.compute_strided_snp_partitions(
                num_attackers, self.args.gene_size
            )

        self._init_spaces()

        # Helper return to match expected gym signature if needed, though step handles the loop
        # Returning dummy state for compatibility
        return np.zeros(self.observation_space.shape[0])

    def _init_spaces(self):
        state_dim = self.beacon.get_state_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    @staticmethod
    def get_assigned_categories(attacker_id: int, k: int, n_categories: int = 6):
        """Return the MAF categories exclusively assigned to this attacker.

        For k <= n_categories: distribute categories as evenly as possible.
          e.g. k=2: attacker 0 → [0,1,2], attacker 1 → [3,4,5]
               k=4: attacker 0 → [0,1], attacker 1 → [2,3], attacker 2 → [4], attacker 3 → [5]
        For k > n_categories: wrap around (attacker_id % n_categories).
          Multiple attackers may share a category; SNP priority queue prevents
          duplicate queries within the shared category.
        """
        if k >= n_categories:
            return [attacker_id % n_categories]
        cats_per_attacker = n_categories // k
        remainder = n_categories % k
        start = attacker_id * cats_per_attacker + min(attacker_id, remainder)
        end = start + cats_per_attacker + (1 if attacker_id < remainder else 0)
        return list(range(start, end))

    @staticmethod
    def compute_snp_partitions(k: int, gene_size: int) -> dict:
        """Divide gene_size SNPs (sorted by MAF) into k equal contiguous blocks.

        Returns attacker_id -> (start, end) SNP index range.
        Attacker 0 gets the lowest-MAF block, attacker k-1 the highest.
        Guarantees no two attackers share the same SNP frontier at any k.
        """
        block = gene_size // k
        return {
            i: (i * block, (i + 1) * block if i < k - 1 else gene_size)
            for i in range(k)
        }

    @staticmethod
    def compute_strided_snp_partitions(k: int, gene_size: int) -> dict:
        """Assign SNPs in round-robin order across the MAF-sorted list.

        Attacker i gets SNPs at positions i, i+k, i+2k, ... giving each attacker
        a uniform spread of rare, medium, and common SNPs.
        Returns attacker_id -> frozenset of SNP indices.
        """
        return {
            i: frozenset(range(i, gene_size, k))
            for i in range(k)
        }

    @staticmethod
    def compute_random_snp_partitions(k: int, gene_size: int) -> dict:
        """Randomly shuffle all SNPs and assign equal-sized non-overlapping subsets.

        Returns attacker_id -> frozenset of SNP indices.
        Each attacker gets exactly gene_size // k SNPs (last attacker gets remainder).
        """
        indices = list(range(gene_size))
        random.shuffle(indices)
        block = gene_size // k
        partitions = {}
        for i in range(k):
            start = i * block
            end = (i + 1) * block if i < k - 1 else gene_size
            partitions[i] = frozenset(indices[start:end])
        return partitions

    def _apply_private_controls(self, attackers, attack_control):
        """Split attack_control into non-overlapping subsets, one per attacker.

        Each attacker gets a unique slice of the control population so its
        p-value is computed against a different baseline than its peers.
        The denominator in _calc_pvalue() stays at args.a_control_size (50),
        so a count of ≤2 still signals identification — but now the count is
        drawn from a smaller private pool, making the test more sensitive to
        marginal evidence that the full 50-control test would miss.

        No-op when --private_controls is not set or k == 1.
        """
        if not getattr(self.args, 'private_controls', False) or len(attackers) <= 1:
            return
        n = attack_control.shape[0]
        # Split indices as evenly as possible across attackers
        splits = torch.chunk(torch.arange(n), len(attackers))
        for att, idx in zip(attackers, splits):
            att.attacker_control = attack_control[idx]
            # Reinitialise control_lrts to match the new (smaller) control size
            att.control_lrts = torch.zeros(att.attacker_control.shape[0])

    def step(self, beacon_agent=None, attacker_agent=None):
        done = False
        if self.current_step == 0:
            self.account_router = None
            if getattr(self.args, 'physical_accounts', None) is not None:
                from account_routing import AccountRouter
                self.account_router = AccountRouter(
                    self.args.physical_accounts, self.args.account_scheduler)

        # 1. Determine Active User (attacker OR benign)
        total_users = len(self.all_users)
        if self.args.collusion_mode == "round_robin" and total_users > 1:
            if (getattr(self.args, 'cold_relay', False)
                    and self.message_board is not None
                    and self.num_attackers > 1
                    and len(self.benign_users) == 0):
                # Cold-start relay. Phase 1: attackers 0..k-2 take turns, routed to whichever
                # has the lowest risk_proxy. Phase 2: once any setup attacker's p-value
                # drops below 0.15, the cold attacker (index k-1) takes all remaining turns.
                cold_idx  = self.num_attackers - 1
                n_setup   = self.num_attackers - 1   # attackers 0..k-2

                if not self._cold_relay_phase2:
                    # Check Phase 2 trigger from message board
                    setup_msgs = [m for m in self.message_board.read_all()
                                  if m['attacker_id'] != cold_idx]
                    if setup_msgs:
                        best_p = min(m['p_value'] for m in setup_msgs)
                        if best_p < 0.15:
                            self._cold_relay_phase2 = True

                if self._cold_relay_phase2:
                    active_idx = cold_idx
                else:
                    setup_msgs = [m for m in self.message_board.read_all()
                                  if m['attacker_id'] != cold_idx]
                    if len(setup_msgs) == n_setup:
                        # Route to lowest-risk setup attacker (suspicion flattening)
                        risks = {m['attacker_id']: m['risk_proxy'] for m in setup_msgs}
                        active_idx = min(risks, key=risks.get)
                    else:
                        # Cold start: round-robin among setup attackers only
                        active_idx = self.current_step % n_setup

            elif (getattr(self.args, 'talking', False)
                    and (getattr(self.args, 'private_controls', False)
                         or getattr(self.args, 'k_partitioning', False))
                    and self.message_board is not None
                    and self.num_attackers > 1
                    and len(self.benign_users) == 0):
                # Talking-informed scheduling. Each attacker broadcasts its p-value after
                # every query and the scheduler uses those messages to pick who queries
                # next, concentrating on the leader once any attacker reaches p < 0.2.
                all_msgs = self.message_board.read_all()
                if len(all_msgs) == self.num_attackers:
                    # All attackers have posted at least once, use communicated p-values
                    comm_pvals = {msg['attacker_id']: msg['p_value'] for msg in all_msgs}
                    pvals = [comm_pvals.get(i, 1.0) for i in range(self.num_attackers)]
                    best_p = min(pvals)
                    if best_p < 0.2:
                        active_idx = pvals.index(best_p)
                    else:
                        active_idx = self.current_step % self.num_attackers
                else:
                    # Cold start: not all attackers have posted yet → round-robin
                    active_idx = self.current_step % self.num_attackers
            elif (getattr(self.args, 'microburst', False)
                    and self._attacker_snp_partitions is not None
                    and self.num_attackers > 1
                    and len(self.benign_users) == 0):
                # Queue-health microburst scheduling. Give bursts of B turns to the attacker
                # whose partition has the highest auc_B, with a cooldown of k-1 rounds.
                # Falls back to round-robin when the priority queue has no data.
                B = self._burst_B
                if self._burst_remaining > 0 and self._burst_owner is not None:
                    # Continue the current burst
                    active_idx = self._burst_owner
                    self._burst_remaining -= 1
                else:
                    # Burst ended, tick cooldowns and pick the next best attacker
                    for aid in list(self._burst_cooldowns.keys()):
                        self._burst_cooldowns[aid] -= 1
                        if self._burst_cooldowns[aid] <= 0:
                            del self._burst_cooldowns[aid]

                    best_auc = -1.0
                    best_aid = self.current_step % self.num_attackers  # round-robin fallback
                    for aid in range(self.num_attackers):
                        if aid in self._burst_cooldowns:
                            continue
                        p_start, p_end = self._attacker_snp_partitions[aid]
                        scores = sorted(
                            (s for idx, s in self.snp_priority.items()
                             if p_start <= idx < p_end),
                            reverse=True
                        )
                        auc_b = sum(scores[:B])
                        if auc_b > best_auc:
                            best_auc = auc_b
                            best_aid = aid

                    active_idx = best_aid
                    if best_auc > 0:
                        # Start a new burst and put winner on cooldown
                        self._burst_remaining = B - 1  # this turn counts as first
                        self._burst_owner = best_aid
                        self._burst_cooldowns[best_aid] = self.num_attackers - 1
                    else:
                        # No queue data yet, pure round-robin, no burst
                        self._burst_remaining = 0
                        self._burst_owner = None
            elif (getattr(self.args, 'adaptive_schedule', False)
                    and self.num_attackers > 1
                    and len(self.benign_users) == 0):
                # Focus-fire driven by local p-value state (no message board).
                pvals = [float(att._calc_pvalue()) for att in self.attackers]
                best_p = min(pvals)
                if best_p < 0.2:
                    active_idx = pvals.index(best_p)
                else:
                    active_idx = self.current_step % self.num_attackers
            else:
                active_idx = self.current_step % total_users
        else:
            active_idx = 0

        # Determine if this is an attacker or benign user
        is_attacker = active_idx < self.num_attackers
        active_user = self.all_users[active_idx]

        # Set beacon tracking info. Only the physical identity sent to the defender
        # is collapsed for the matched single-account control.
        self.beacon.current_agent_id = (
            0 if is_attacker and getattr(self.args, 'single_physical_account', False)
            else active_idx if not getattr(self.args, 'single_physical_account', False)
            else 1 + active_idx - self.num_attackers)
        self.beacon.current_user_type = "attacker" if is_attacker else "benign"
        if self.account_router is not None and is_attacker:
            self.beacon.current_agent_id = self.account_router.choose()

        # For backwards compatibility, also set active_attacker reference
        if is_attacker:
            active_attacker = active_user
        else:
            # Use first attacker for reward/state calculations when benign user is active
            active_attacker = self.attackers[0] if self.attackers else None

        # Store beacon agent's expected state dim (if provided) for state adaptation
        beacon_expected_dim = getattr(beacon_agent, 'state_dim', None) if beacon_agent else None

        if self.current_step % self.args.print_freq == 0:
            user_type_str = "Attacker" if is_attacker else "Benign"
            print(f"--------------------------------Query: {self.current_step+1} ({user_type_str} {active_idx})---------------------------------")

        # 2. Get Action from Active User (attacker OR benign)
        if is_attacker:
            # Attacker: use RL agent or optimal strategy
            if self.args.attacker_type == "agent":
                include_k = getattr(self.args, 'k_conditioned', False)
                attacker_state = active_attacker.get_state(
                    include_k=include_k,
                    k=self.num_attackers,
                )
                # --- MAF Specialization: restrict policy to this attacker's categories ---
                action_mask = None
                if getattr(self.args, 'maf_specialization', False) and self.num_attackers > 1:
                    assigned = self.get_assigned_categories(
                        active_attacker.attacker_id, self.num_attackers
                    )
                    action_mask = [0.0] * 12
                    for cat in assigned:
                        action_mask[cat] = 1.0       # presence=0, this category
                        action_mask[cat + 6] = 1.0   # presence=1, this category
                # MAF concentration (S2e): restrict every account to category 0. This routes
                # GBA's 12 native actions; the SNP itself is chosen by snp_partition.
                elif (getattr(self.args, 'maf_concentration', False)
                        or getattr(self.args, 'maf_concentration_sharded', False)):
                    # maf_concentration_sharded reuses this action_mask; its SNP-level sharding
                    # comes from _attacker_snp_partitions. Applied at k=1 as well. The category
                    # is configurable with --maf_concentration_category.
                    _cat = int(getattr(self.args, 'maf_concentration_category', 0))
                    action_mask = [0.0] * 12
                    action_mask[_cat] = 1.0       # presence=0, this category
                    action_mask[_cat + 6] = 1.0   # presence=1, this category
                # Portfolio targeting: split the coalition across three targeting policies by
                # attacker_id, combined by pooling rather than the any-one rule.
                elif getattr(self.args, 'portfolio_targeting', False):
                    _slot = active_attacker.attacker_id % 3
                    if _slot == 0:
                        action_mask = None   # unrestricted
                    else:
                        _pcat = 0 if _slot == 1 else 2
                        action_mask = [0.0] * 12
                        action_mask[_pcat] = 1.0
                        action_mask[_pcat + 6] = 1.0
                # Talking: peers broadcast per-category average honesty, so categories the
                # defender is suppressing are avoided here too. Skipped when k_partitioning
                # is active, since the partition filter already selects the SNP.
                if (getattr(self.args, 'talking', False)
                        and self.message_board is not None
                        and self.num_attackers > 1
                        and not getattr(self.args, 'k_partitioning', False)):
                    peer_msgs = self.message_board.read_others(active_attacker.attacker_id)
                    if peer_msgs:
                        n_p = len(peer_msgs)
                        avg_peer_honesty = [
                            sum(m['avg_honesty'][c] for m in peer_msgs) / n_p
                            for c in range(6)
                        ]
                        if action_mask is None:
                            action_mask = [1.0] * 12
                        for c in range(6):
                            if avg_peer_honesty[c] < 0.35:
                                # Only suppress this category if at least one other
                                # category remains unblocked (prevent deadlock).
                                other_ok = any(
                                    action_mask[cc] > 0
                                    for cc in range(6) if cc != c
                                )
                                if other_ok:
                                    action_mask[c] = 0.0
                                    action_mask[c + 6] = 0.0
                # Single-category restriction: force the attacker to one MAF category
                # regardless of coalition size.
                _rc = getattr(self.args, 'restrict_category', -1)
                if _rc is not None and _rc >= 0:
                    action_mask = [0.0] * 12
                    action_mask[_rc] = 1.0        # presence=0, category _rc
                    action_mask[_rc + 6] = 1.0    # presence=1, category _rc
                # -----------------------------------------------------------------------
                agent_action = attacker_agent.select_action(
                    attacker_state, eval_mode=self.args.evaluate, action_mask=action_mask
                )
                # --- SNP Priority Queue: pick highest-gain SNP within chosen category ---
                snp_pri = self.snp_priority if getattr(self.args, 'snp_priority', False) else None
                # Virtual sharding: per-category (offset, stride) for this attacker. Only
                # applied when at least two categories are shared.
                shard_info = None
                if getattr(self.args, 'virtual_sharding', False) and self.num_attackers > 1:
                    cat_to_attackers = {}
                    for aid in range(self.num_attackers):
                        for cat in self.get_assigned_categories(aid, self.num_attackers):
                            cat_to_attackers.setdefault(cat, []).append(aid)
                    n_shared = sum(1 for owners in cat_to_attackers.values() if len(owners) > 1)
                    if n_shared >= 2:
                        aid = active_attacker.attacker_id
                        assigned_cats = self.get_assigned_categories(aid, self.num_attackers)
                        shard_info = {}
                        for cat in assigned_cats:
                            owners = cat_to_attackers.get(cat, [aid])
                            stride = len(owners)
                            offset = owners.index(aid) if aid in owners else 0
                            shard_info[cat] = (offset, stride)
                # --- SNP Reservation Board: skip SNPs reserved by peers this round ---
                reserved_snps = (
                    self.reservation_board.get_reserved(active_attacker.attacker_id)
                    if self.reservation_board is not None else None
                )
                # Coalition-wide permanent exclusion: never re-ask a SNP any attacker has
                # queried this episode. reserved_snps only blocks a re-ask for the
                # reservation TTL, after which a peer can re-select a consumed SNP,
                # especially when accounts share a MAF category.
                consumed_snps = (
                    set(self._queried_loci)
                    if (getattr(self.args, 'private_account_evidence', False)
                        and self.num_attackers > 1)
                    else None
                )
                # --- k-Partitioning: restrict to this attacker's unique MAF block ---
                snp_partition = (
                    self._attacker_snp_partitions.get(active_attacker.attacker_id)
                    if self._attacker_snp_partitions is not None else None
                )
                # --- Gain-sharing: update within-partition SNP ordering bias ---
                if (getattr(self.args, 'gain_sharing', False)
                        and self.message_board is not None
                        and self.num_attackers > 1):
                    peer_msgs = self.message_board.read_others(active_attacker.attacker_id)
                    active_attacker.update_partition_bias(peer_msgs, active_attacker.attacker_id)
                # ---------------------------------------------------------------------------
                attacker_action = active_attacker.act(
                    self.current_step, agent_action,
                    snp_priority=snp_pri, shard_info=shard_info,
                    reserved_snps=reserved_snps, snp_partition=snp_partition,
                    consumed_snps=consumed_snps,
                )
                # Post reservation so peers skip this SNP for the next k turns
                if self.reservation_board is not None:
                    self.reservation_board.reserve(
                        active_attacker.attacker_id, attacker_action, ttl=self.num_attackers
                    )
                # ---------------------------------------------------------------------------
                self.attacker_agent_actions.append(agent_action)
                # One-time confirmation that PPO is driving attacker actions.
                if self.current_step == 0 and not getattr(self, "_printed_attacker_policy", False):
                    if getattr(self.args, 'log_level', 'info') == "debug":
                        print(f"Using RL attacker policy: PPO (agent_action={agent_action})")
                    self._printed_attacker_policy = True
            else:
                attacker_action = active_attacker.act(self.current_step)
                self.attacker_agent_actions.append(attacker_action)
        else:
            # Benign user: simple query selection (no RL agent needed)
            attacker_action = active_user.act(self.current_step)
            self.attacker_agent_actions.append(attacker_action)

        # Repeat-query mode. Randomised response flips one-directionally, so a single
        # YES is conclusive and asking the same locus more than once defeats it.
        # With --repeat_query R the coalition spends R queries on each locus.
        if getattr(self.args, 'repeat_query', 0) and is_attacker:
            _R = int(self.args.repeat_query)
            _buf = getattr(self, '_rq_buf', None)
            if _buf is not None and _buf['left'] > 0:
                attacker_action = _buf['locus']
                _buf['left'] -= 1
            else:
                self._rq_buf = {'locus': int(attacker_action), 'left': _R - 1}
        # ---------------------------------------------------------------------

        # 3. Get Beacon Response
        if self.args.beacon_type == "agent":
            beacon_state = self.beacon.get_state(attacker_action, self.current_step)

            # --- STATE DIMENSION ADAPTATION ---
            # If beacon agent expects different state dim (e.g., trained without collusion features),
            # adapt the state by truncating or padding
            if beacon_expected_dim is not None and len(beacon_state) != beacon_expected_dim:
                beacon_state_arr = np.array(beacon_state) if not isinstance(beacon_state, np.ndarray) else beacon_state
                if len(beacon_state_arr) > beacon_expected_dim:
                    # Truncate: checkpoint trained without collusion features
                    beacon_state = beacon_state_arr[:beacon_expected_dim].tolist()
                else:
                    # Pad: checkpoint trained with collusion features but running without
                    padding = np.zeros(beacon_expected_dim - len(beacon_state_arr))
                    beacon_state = np.concatenate([beacon_state_arr, padding]).tolist()
            # ----------------------------------------

            # --- AGENT SELECTION & PENALTY INJECTION ---
            if self.args.beacon_agent == "td":
                if self.args.evaluate:
                    beacon_action = beacon_agent.select_action(beacon_state, True)
                else :
                    beacon_action = beacon_agent.select_action(beacon_state, False)
                    beacon_action = torch.Tensor(beacon_action).squeeze()

            elif self.args.beacon_agent == "ppo":
                    beacon_action = beacon_agent.select_action(beacon_state)
                    beacon_action = torch.Tensor(beacon_action)
            else:
                raise NotImplemented
        else:
            beacon_state = self.beacon.get_state(attacker_action, self.current_step)
            max_pvalue_change_threshold = 0.3
            beacon_action = self.beacon.act(attacker_action, max_pvalue_change_threshold)

        self.last_raw_policy_honesty = float(torch.as_tensor(beacon_action).item())

        # Gate the honesty value on presence: the policy's honesty if some
        # beacon member carries the queried allele, 0 otherwise. Matches the
        # base paper's threat model, where a genuine absence is reported
        # truthfully. Off by default.
        if getattr(self.args, 'presence_aware_response', False):
            _present = float(torch.sum(self.beacon.beacon_case[:, int(attacker_action)]) > 0)
            if isinstance(beacon_action, torch.Tensor):
                beacon_action = beacon_action * _present
            else:
                beacon_action = type(beacon_action)(float(beacon_action) * _present) \
                    if not isinstance(beacon_action, float) else float(beacon_action) * _present

        # Executable attacker: emit an actual yes/no. NO whenever the allele is
        # absent, since flipping is one-directional, otherwise YES with probability
        # h. The defender's own state is left untouched.
        _exec_r = None
        _exec_h = None
        if getattr(self.args, 'executable_attacker', False) and is_attacker:
            _x = float(torch.sum(self.beacon.beacon_case[:, int(attacker_action)]) > 0)
            _hraw = float(torch.as_tensor(beacon_action, dtype=torch.float32).flatten()[-1])
            # under presence gating beacon_action is already h*x, so for x=1 it is h
            _exec_h = _hraw if _x > 0 else 1.0
            if getattr(self.args, 'exec_absent_yes', False):
                # With --exec_absent_yes an absent allele is answered YES, so a NO is
                # conclusive of presence and a YES is ambiguous.
                _exec_r = 0.0 if (_x > 0 and np.random.random() >= _exec_h) else 1.0
            else:
                # Default: an absent allele is answered NO truthfully
                # (one-directional flipping, base paper's "if x_j=0 then h_j=1").
                _exec_r = 1.0 if (_x > 0 and np.random.random() < _exec_h) else 0.0
        # ----------------------------------------------------------------------

        if self.args.binary:
            if isinstance(beacon_action, torch.Tensor):
                beacon_action = (beacon_action >= 0.5).float()
            elif isinstance(beacon_action, np.ndarray):
                beacon_action = (beacon_action >= 0.5).astype(float)
            elif isinstance(beacon_action, (np.float32, float)):
                beacon_action = np.array([float(beacon_action >= 0.5)], dtype=np.float32)[0]

        if self.current_step % self.args.print_freq == 0 and getattr(self.args, 'log_level', 'info') == "debug":
            print(f"Attacker Action: {attacker_action}")
            print(f"Beacon Action: {beacon_action}")

        # 4. Synchronized Update: Update ALL users with the shared result
        # Record pre-update LRT so we can compute the gain for SNP priority
        # tracking and the talking mechanism (both need lrt_gain).
        pre_victim_lrt = (
            active_attacker.victim_lrt.item()
            if is_attacker and (
                getattr(self.args, 'snp_priority', False)
                or getattr(self.args, 'talking', False)
                or getattr(self.args, 'cold_relay', False)
                or getattr(self.args, 'gain_sharing', False)
            )
            else None
        )
        # Capture victim's genome value before update() zeroes it.
        # Needed for any local LRT tracking (fisher_local, individual_local).
        _needs_local = (
            getattr(self.args, 'fisher_local', False)
            or getattr(self.args, 'individual_local', False)
            or getattr(self.args, 'ebm_local', False)
        )
        victim_snp_value = (
            float(active_attacker.victim[int(attacker_action)])
            if is_attacker and _needs_local
            else None
        )
        # Record the actual beacon query (one locus per step) for coverage/overlap.
        if is_attacker:
            self._queried_loci.append(int(attacker_action))
            # Running log of the defender's responses to attacker queries, so the
            # intermediate-budget snapshots can show whether its behaviour drifts.
            if getattr(self, '_attacker_resp_log', None) is None:
                self._attacker_resp_log = []
            try:
                self._attacker_resp_log.append(
                    float(torch.as_tensor(beacon_action, dtype=torch.float32).flatten()[-1]))
            except Exception:
                pass

        # Per-query LRT attribution (explainability): capture the raw inputs to the
        # LRT contribution BEFORE update() zeroes the consumed genotype. The exact
        # per-query delta is recomputed offline from (maf, response, genotype).
        if getattr(self.args, 'log_attribution', False) and is_attacker:
            if getattr(self, '_attribution', None) is None:
                self._attribution = []
            _snp = int(attacker_action)
            self._attribution.append((
                int(getattr(active_attacker, 'attacker_id', 0)),  # which account
                float(active_attacker.mafs[_snp]),                # MAF of queried SNP
                float(beacon_action),                             # response/honesty in LRT
                float(active_attacker.victim[_snp]),              # victim genotype (1=carries)
            ))
            # Record the ground-truth presence next to the response, so the dependence
            # of the response on actual presence can be checked directly.
            if getattr(self, '_resp_presence', None) is None:
                self._resp_presence = []
            _present = float(torch.sum(self.beacon.beacon_case[:, _snp]) > 0)
            self._resp_presence.append((float(beacon_action), _present))

        # Query selection and local scoring must see the SAME observable bit.
        # Keep the defender's own expectation-based state update unchanged.
        attacker_response = beacon_action
        if getattr(self.args, 'exec_lrt_uses_response', False) and is_attacker:
            if _exec_r is None:
                raise ValueError("Sampled attacker feedback requires executable_attacker")
            attacker_response = float(_exec_r)
        self.last_attacker_response = float(torch.as_tensor(attacker_response).item())
        if self.account_router is not None and is_attacker:
            self.account_router.observe(self.beacon.current_agent_id, _exec_r)

        if getattr(self, '_independent_consumption', False):
            # Uncoordinated S1: only the querying attacker incorporates this response,
            # and marks the locus consumed in its own victim copy.
            if is_attacker:
                active_attacker.update(beacon_action=attacker_response, attacker_action=attacker_action)
        else:
            # Coordinated strategies (S2a-d) and single-attacker: all attackers
            # share every response (pooled evidence) and mark loci consumed in
            # lock-step, so the coalition never re-queries.
            for att in self.attackers:
                att.update(beacon_action=attacker_response, attacker_action=attacker_action)

        # Per-partition local LRT update (Fisher's local test, active attacker only).
        if victim_snp_value is not None:
            active_attacker.update_local(beacon_action, attacker_action, victim_snp_value,
                                         exec_r=_exec_r, exec_h=_exec_h)

        # Update benign users too (they track their own state)
        for benign in self.benign_users:
            benign.update(beacon_action=beacon_action, attacker_action=attacker_action)

        self.beacon.update(beacon_action=beacon_action, attacker_action=attacker_action)

        # Update SNP priority queue: exponential moving average of LRT gain per SNP.
        # Persists across episodes, SNPs that consistently yield high gain rank higher.
        if pre_victim_lrt is not None:
            gain = active_attacker.victim_lrt.item() - pre_victim_lrt
            if getattr(self.args, 'snp_priority', False):
                snp_key = int(attacker_action)
                old = self.snp_priority.get(snp_key, 0.0)
                self.snp_priority[snp_key] = old * 0.7 + gain * 0.3
            # --- Talking: record per-category stats and broadcast message -----------
            if ((getattr(self.args, 'talking', False)
                    or getattr(self.args, 'cold_relay', False)
                    or getattr(self.args, 'gain_sharing', False))
                    and self.message_board is not None
                    and is_attacker):
                if isinstance(beacon_action, torch.Tensor):
                    ba_scalar = beacon_action.item()
                elif isinstance(beacon_action, np.ndarray):
                    ba_scalar = float(beacon_action.flat[0])
                else:
                    ba_scalar = float(beacon_action)
                cats = getattr(active_attacker, 'categorized_maf', None)
                active_attacker.record_query(attacker_action, ba_scalar, gain, cats)
                self.message_board.post(
                    active_attacker.attacker_id, active_attacker.build_message()
                )
            # -----------------------------------------------------------------------

        # 5. Calculate Rewards & Check Done (using active attacker state)
        beacon_reward, beacon_done, beacon_rewards = self.beacon.calc_reward(beacon_action=beacon_action)
        attacker_reward, attacker_done, attacker_rewards = active_attacker.calc_reward(beacon_action, self.current_step)
        # S1 (majority voting): a single attacker reaching significance must not end
        # the episode or count as an identification; only a majority may decide,
        # which is checked in the individual-local block below.
        _defer_to_majority = (
            getattr(self.args, 'individual_local', False)
            and getattr(self.args, 'majority_voting', False)
            and self.num_attackers > 1
        )
        if _defer_to_majority:
            attacker_done = False
        else:
            if getattr(active_attacker, "last_identify_event", False):
                attacker_id = getattr(active_attacker, "attacker_id", active_idx)
                self.identified_attackers.add(attacker_id)
                self.identify_events[attacker_id] = self.identify_events.get(attacker_id, 0) + 1
                self.ever_identified = len(self.identified_attackers) > 0
            if attacker_done and os.getenv("DEBUG_CONTINUE_AFTER_IDENTIFY") == "1":
                # Continue-after-identify mode: do not terminate the episode on identification.
                attacker_done = False

        # Fisher's combined p-value test. Each attacker runs an independent LRT, and
        # their p-values are combined as chi2 = -2 * sum(ln p_i) ~ chi2(2k).
        if (getattr(self.args, 'fisher_combined', False)
                and self.num_attackers > 1
                and not attacker_done):
            from scipy.stats import chi2 as _chi2
            pvals = [max(float(att._calc_pvalue()), 1e-6) for att in self.attackers]
            chi2_stat = -2.0 * sum(math.log(p) for p in pvals)
            combined_pvalue = _chi2.sf(chi2_stat, df=2 * self.num_attackers)
            if combined_pvalue < 0.05:
                for att in self.attackers:
                    self.identified_attackers.add(att.attacker_id)
                self.ever_identified = True
                attacker_done = True
                print(f"[FISHER] χ²={chi2_stat:.3f} df={2*self.num_attackers} "
                      f"combined_p={combined_pvalue:.4f} individual_p={[round(p,3) for p in pvals]}"
                      f" → IDENTIFIED")
        # ---------------------------------------

        # Fisher's combined test on per-partition local p-values, which are
        # independent because each attacker queries a different SNP set.
        if (getattr(self.args, 'fisher_local', False)
                and self.num_attackers > 1
                and not attacker_done):
            from scipy.stats import chi2 as _chi2_local
            if all(att._local_n_queries > 0 for att in self.attackers):
                local_pvals = [att.calc_local_pvalue() for att in self.attackers]
                chi2_stat = -2.0 * sum(math.log(p) for p in local_pvals)
                combined_pvalue = _chi2_local.sf(chi2_stat, df=2 * self.num_attackers)
                if combined_pvalue < 0.05:
                    for att in self.attackers:
                        self.identified_attackers.add(att.attacker_id)
                    self.ever_identified = True
                    attacker_done = True
                    print(f"[FISHER-LOCAL] χ²={chi2_stat:.3f} df={2*self.num_attackers} "
                          f"combined_p={combined_pvalue:.4f} "
                          f"local_p={[round(p, 3) for p in local_pvals]} → IDENTIFIED")
        # -----------------------------------------------------------------

        # Individual local p-value identification. Each attacker tracks only its own
        # queries. majority_voting=True fires when more than k/2 attackers have
        # local p < 0.05 (S1); False fires when any one does (S2a-S2d).
        if (getattr(self.args, 'individual_local', False)
                and self.num_attackers > 1
                and not attacker_done):
            local_pvals = [
                att.calc_local_pvalue() if att._local_n_queries > 0 else 1.0
                for att in self.attackers
            ]
            n_identified = sum(1 for p in local_pvals if p < 0.05)
            majority_voting = getattr(self.args, 'majority_voting', False)
            threshold = self.num_attackers / 2 if majority_voting else 0
            if n_identified > threshold:
                for att in self.attackers:
                    self.identified_attackers.add(att.attacker_id)
                self.ever_identified = True
                attacker_done = True
                rule = "MAJORITY" if majority_voting else "ANY-ONE"
                print(f"[INDIVIDUAL-LOCAL/{rule}] {n_identified}/{self.num_attackers} "
                      f"local_p={[round(p,3) for p in local_pvals]} → IDENTIFIED")
        # -----------------------------------------------------------------

        # Empirical Brown's Method on per-partition local p-values. The attackers
        # share a control panel, so their p-values are weakly dependent and raw
        # Fisher over-fires. EBM (Poole et al. 2016) rescales the statistic to
        # c * chi2(2f), falling back to Fisher when the covariance is degenerate.
        if (getattr(self.args, 'ebm_local', False)
                and self.num_attackers > 1
                and not attacker_done):
            from scipy.stats import chi2 as _chi2_ebm
            if all(att._local_n_queries > 0 for att in self.attackers):
                local_pvals = [att.calc_local_pvalue() for att in self.attackers]
                W_obs = -2.0 * sum(math.log(p) for p in local_pvals)
                c_scale, dof = self._ebm_brown_params(
                    [att.local_control_lrts for att in self.attackers])
                combined_pvalue = float(_chi2_ebm.sf(W_obs / c_scale, df=dof))
                if combined_pvalue < 0.05:
                    for att in self.attackers:
                        self.identified_attackers.add(att.attacker_id)
                    self.ever_identified = True
                    attacker_done = True
                    print(f"[EBM-LOCAL] W={W_obs:.3f} c={c_scale:.3f} dof={dof:.2f} "
                          f"combined_p={combined_pvalue:.4f} "
                          f"local_p={[round(p, 3) for p in local_pvals]} → IDENTIFIED")
        # -----------------------------------------------------------------

        # Score logging: per-account running minimum local p-value over the episode,
        # so TPR/FPR, ROC/AUC and threshold corrections can be derived offline
        # without changing the identification logic.
        if getattr(self.args, 'log_scores', False):
            # (a) per-account local min-p (own-query LRT), the LOCAL any-one rule
            if getattr(self, '_min_local_pvals', None) is None \
                    or len(self._min_local_pvals) != self.num_attackers:
                self._min_local_pvals = [1.0] * self.num_attackers
            for _j, _att in enumerate(self.attackers):
                if getattr(_att, '_local_n_queries', 0) > 0:
                    _p = _att.calc_local_pvalue()
                    if _p < self._min_local_pvals[_j]:
                        self._min_local_pvals[_j] = _p
            # (b) coalition min-p over all attackers' full LRTs
            if getattr(self, '_min_pooled_pval', None) is None:
                self._min_pooled_pval = 1.0
            _pp = min(float(_a._calc_pvalue()) for _a in self.attackers)
            if _pp < self._min_pooled_pval:
                self._min_pooled_pval = _pp
            # (c) Per-account local LRT margin. A lower LRT is more member-like,
            # so min(control_lrts) - victim_lrt is large-positive when the victim
            # sits below the best control. Unlike r/50 it does not floor at 0.
            # Running max per account; the any-one coalition score is the max
            # across accounts.
            if getattr(self, '_max_local_margin', None) is None \
                    or len(self._max_local_margin) != self.num_attackers:
                self._max_local_margin = [float('-inf')] * self.num_attackers
            if getattr(self, '_final_local_margin', None) is None \
                    or len(self._final_local_margin) != self.num_attackers:
                self._final_local_margin = [float('-inf')] * self.num_attackers
            for _j, _att in enumerate(self.attackers):
                if getattr(_att, '_local_n_queries', 0) > 0 \
                        and len(getattr(_att, 'local_control_lrts', [])) > 0:
                    _m = (float(torch.min(_att.local_control_lrts))
                          - float(_att.local_victim_lrt))
                    if _m > self._max_local_margin[_j]:
                        self._max_local_margin[_j] = _m
                    # Current margin, overwritten every step, so at the end of the
                    # episode it holds the value after the full query budget: one test
                    # at a fixed budget, as in Shringarpure & Bustamante (2015), rather
                    # than the running maximum above.
                    self._final_local_margin[_j] = _m

            # (d) Pooled coalition evidence. The per-account LRTs are built from
            # disjoint queries, so the coalition statistic is the sum of the
            # per-account victim LRTs against the element-wise sum of the
            # per-account control LRTs. Tracked as a running maximum.
            _pv, _pc = 0.0, None
            _pnm, _pnv, _pni = 0.0, 0.0, 0
            _pf = _pfm = _pfv = _pfm1 = _fgrid = None
            _psm = _psv = _psm1 = None
            _prare, _pmafsum = 0, 0.0
            for _att in self.attackers:
                if getattr(_att, '_local_n_queries', 0) > 0 \
                        and len(getattr(_att, 'local_control_lrts', [])) > 0:
                    _pv += float(_att.local_victim_lrt)
                    _c = _att.local_control_lrts.detach().clone()
                    _pc = _c if _pc is None else _pc + _c
                    # Accounts query DISJOINT loci, so the analytic H0 moments of
                    # the coalition's summed LRT are the sums of the per-account
                    # moments (variances add under independence across loci).
                    _pnm += float(getattr(_att, 'local_null_mean', 0.0))
                    _pnv += float(getattr(_att, 'local_null_var', 0.0))
                    _pni += int(getattr(_att, '_local_n_informative', 0))
                    # Flip-rate-aware LLR, pooled over the same f grid.
                    _fg = getattr(_att, 'f_grid', None)
                    if _fg:
                        if _pf is None:
                            _pf = [0.0] * len(_fg)
                            _pfm = [0.0] * len(_fg)
                            _pfv = [0.0] * len(_fg)
                            _pfm1 = [0.0] * len(_fg)
                            _psm = [0.0] * len(_fg)
                            _psv = [0.0] * len(_fg)
                            _psm1 = [0.0] * len(_fg)
                            _fgrid = list(_fg)
                        for _i in range(len(_fg)):
                            _pf[_i] += float(_att.llr_f[_i])
                            _pfm[_i] += float(_att.llr_f_mean[_i])
                            _pfv[_i] += float(_att.llr_f_var[_i])
                            _pfm1[_i] += float(_att.llr_f_mean_h1[_i])
                            _psm[_i] += float(_att.shadow_mean[_i])
                            _psv[_i] += float(_att.shadow_var[_i])
                            _psm1[_i] += float(_att.shadow_mean_h1[_i])
                    _prare += int(getattr(_att, '_n_informative_rare', 0))
                    _pmafsum += float(getattr(_att, '_maf_sum_informative', 0.0))
            # Pooled victim LRT and its analytic H0 moments, stored raw so the offline
            # analysis can form Z = (LRT - E0) / sqrt(Var0). For the executable
            # attacker, per-locus tallies are merged across accounts first.
            if getattr(self.args, 'executable_attacker', False):
                _merged = {}
                _hs, _hn = 0.0, 0
                for _att in self.attackers:
                    _hs += float(getattr(_att, 'exec_h_sum', 0.0))
                    _hn += int(getattr(_att, 'exec_h_n', 0))
                for _att in self.attackers:
                    for _loc, _d in getattr(_att, 'exec_locus', {}).items():
                        if _loc in _merged:
                            _merged[_loc]['n_yes'] += _d['n_yes']
                            _merged[_loc]['n'] += _d['n']
                        else:
                            _merged[_loc] = dict(_d)
                _proxy = self.attackers[0]
                _saved = getattr(_proxy, 'exec_locus', {})
                _proxy.exec_locus = _merged
                # h_bar must be exogenous: estimating it from this coalition's own
                # queries against this victim would leak membership on its own.
                _hbar_fixed = getattr(self.args, 'exec_h_bar', None)
                _absent_yes = bool(getattr(self.args, 'exec_absent_yes', False))
                _el, _ee0, _ev0, _en, _ehb = _proxy.finalize_exec(absent_yes=_absent_yes,
                    h_bar=_hbar_fixed if _hbar_fixed is not None
                    else ((_hs / _hn) if _hn else None))
                self._final_exec_h_observed = (_hs / _hn) if _hn else None
                _proxy.exec_locus = _saved
                self._final_exec_llr = _el
                self._final_exec_null_mean = _ee0
                self._final_exec_null_var = _ev0
                self._final_exec_n_loci = _en
                self._final_exec_h_bar = _ehb
                # member-high convention, matching the other margins
                self._final_exec_margin = (
                    -(_el - _ee0) / (_ev0 ** 0.5) if _ev0 > 0 else float('nan'))
                self._final_exec_draws = int(sum(v['n'] for v in _merged.values()))
                # Per-account exec statistic, so the coalition can be scored by the any-one
                # rule. The pooled statistic above is partition-invariant.
                _pa = []
                for _att in self.attackers:
                    if not getattr(_att, 'exec_locus', None):
                        continue
                    _l, _e0, _v0, _n, _ = _att.finalize_exec(h_bar=_ehb, absent_yes=_absent_yes)
                    if _v0 > 0:
                        _pa.append(-(_l - _e0) / (_v0 ** 0.5))   # member-high
                self._final_exec_margin_per_account = _pa
                # any-one rule: the coalition takes its strongest account
                self._final_exec_margin_anyone = max(_pa) if _pa else float('nan')
                # Running maxima. The two fields above are overwritten each step, so
                # only the end-of-episode value survives, while the any-one rule
                # fires at the first crossing from any account. Keep the running
                # max per account and across accounts.
                if getattr(self, '_run_exec_margin_per_account', None) is None \
                        or len(self._run_exec_margin_per_account) != len(_pa):
                    self._run_exec_margin_per_account = list(_pa)
                else:
                    self._run_exec_margin_per_account = [
                        max(a, b) for a, b in zip(self._run_exec_margin_per_account, _pa)]
                _ra = self._run_exec_margin_per_account
                self._run_exec_margin_anyone = max(_ra) if _ra else float('nan')
                _pm = self._final_exec_margin
                if _pm == _pm:   # not NaN
                    _prev = getattr(self, '_run_exec_margin_pooled', None)
                    self._run_exec_margin_pooled = _pm if _prev is None else max(_prev, _pm)
            self._final_pooled_victim_lrt = _pv
            self._final_pooled_null_mean = _pnm
            self._final_pooled_null_var = _pnv
            self._final_pooled_n_informative = _pni
            self._final_llr_f = _pf
            self._final_llr_f_mean = _pfm
            self._final_llr_f_var = _pfv
            self._final_f_grid = _fgrid

            # Intermediate-budget readouts: read the statistic at fixed intermediate
            # budgets with the individual and k held fixed, which separates locus
            # depletion from within-episode adaptation and decouples N from k.
            _n_q = len(self._queried_loci)
            if _n_q in _BUDGET_CHECKPOINTS:
                if getattr(self, '_budget_snapshots', None) is None:
                    self._budget_snapshots = {}
                self._budget_snapshots[_n_q] = {
                    'victim_lrt': _pv,
                    'null_mean': _pnm,
                    'null_var': _pnv,
                    'n_informative': _pni,
                    'llr_f': list(_pf) if _pf else None,
                    'llr_f_mean': list(_pfm) if _pfm else None,
                    'llr_f_var': list(_pfv) if _pfv else None,
                    'llr_f_mean_h1': list(_pfm1) if _pfm1 else None,
                    'shadow_mean': list(_psm) if _psm else None,
                    'shadow_var': list(_psv) if _psv else None,
                    'shadow_mean_h1': list(_psm1) if _psm1 else None,
                    'n_informative_rare': _prare,
                    'maf_sum_informative': _pmafsum,
                    'resp_mean_so_far': (float(np.mean(self._attacker_resp_log))
                                         if getattr(self, '_attacker_resp_log', None) else None),
                    'resp_mean_recent': (float(np.mean(self._attacker_resp_log[-100:]))
                                         if getattr(self, '_attacker_resp_log', None) else None),
                }
            if _pc is not None and len(_pc) > 0:
                _pm = float(torch.min(_pc)) - _pv
                if getattr(self, '_max_pooled_margin', None) is None:
                    self._max_pooled_margin = float('-inf')
                if _pm > self._max_pooled_margin:
                    self._max_pooled_margin = _pm
                # Pooled margin after the full query budget, mirroring _final_local_margin
                # but for the coalition-combined statistic. max_pooled_margin is the
                # running maximum; this one is overwritten every step.
                self._final_pooled_margin = _pm
        # ---------------------------------------------------------------------

        self.attacker_actions.append(attacker_action)
        self.current_step += 1
        if self.reservation_board is not None:
            self.reservation_board.tick()

        # Under --calibrated_null the victim is a non-member from the
        # defence-control pool, so it is tracked in control_lrts, not
        # beacon_lrts (which holds the N members only).
        _n_members = self.beacon.beacon_lrts.shape[0]
        victim_lrt = (self.beacon.beacon_lrts[self.victim_id].item()
                      if int(self.victim_id) < _n_members
                      else self.beacon.control_lrts[int(self.victim_id) - _n_members].item())
        if self.max_victim_lrt is None or victim_lrt > self.max_victim_lrt:
            self.max_victim_lrt = victim_lrt
        
        if self.current_step >= self.max_steps:
            done = True

        done = done or attacker_done

        # Print episode result whenever episode ends (either max_steps or early identification)
        if done and not getattr(self, '_episode_result_printed', False):
            self._episode_result_printed = True
            if not self.ever_identified:
                print("✅✅✅ Attacker Could NOT Indentify the VICTIM ✅✅✅")
            else:
                print("⚠️ Episode had IDENTIFICATION events")
        if os.getenv("DEBUG_COLLUSION_CONTINUE") == "1":
            done = self.current_step >= self.max_steps

        if done and self.current_step % self.args.print_freq == 0:
             print("Victim LRT (Final): ", active_attacker.victim_lrt)

        return [np.array(beacon_state), beacon_action, beacon_reward, done], \
               [beacon_reward, attacker_reward], \
               done, \
               [active_attacker._calc_pvalue(), beacon_action]

    def _uses_independent_consumption(self, num_attackers):
        """True only for the UNCOORDINATED S1 (Majority Voting) strategy: >1
        attacker, individual local p-values, majority decision rule, and no
        partitioning. Such attackers each keep a private record of consumed loci
        and see only their own responses, so overlapping (wasted) re-queries are
        possible — matching the paper's S1 definition. Every other configuration
        (S2a-d, single attacker, and all experimental schedulers) returns False
        and keeps the synchronized/pooled behavior, so their results are
        unchanged.
        """
        if num_attackers <= 1:
            return False
        # Every collusion strategy keeps per-account evidence: each attacker
        # maintains a private LRT accumulator over its own assigned
        # partition. With the private victim copy in Attacker.__init__, an
        # account's statistic depends only on its own queries.
        if getattr(self.args, 'private_account_evidence', False):
            return True
        if not (getattr(self.args, 'individual_local', False)
                and getattr(self.args, 'majority_voting', False)):
            return False
        partition_flags = ('k_partitioning', 'random_partitioning',
                           'strided_partitioning', 'maf_specialization',
                           'maf_concentration', 'portfolio_targeting',
                           'maf_concentration_sharded')
        return not any(getattr(self.args, f, False) for f in partition_flags)

    def _ebm_brown_params(self, control_lrt_vectors):
        """Empirical Brown's Method parameters (Poole et al. 2016, Bioinformatics).

        control_lrt_vectors: list of k tensors, each length m = shared control
        panel size. Entry (c, i) is attacker i's accumulated local LRT for the
        c-th shared control individual. Because the m controls are shared across
        all k attackers, correlation between columns is exactly the dependence
        Brown/EBM corrects for.

        Returns (c_scale, dof) for the rescaled null  W ~ c_scale · χ²(dof).
        Under independence this reduces to c_scale=1, dof=2k (plain Fisher). The
        method also caps dof at 2k with c_scale=1, so the combined test is never
        MORE significant than Fisher (Poole et al.'s conservative safeguard).
        """
        import numpy as _np
        k = len(control_lrt_vectors)
        if k < 2:
            return 1.0, 2.0 * max(k, 1)
        cols = []
        for v in control_lrt_vectors:
            arr = v.detach().cpu().numpy() if hasattr(v, 'detach') else _np.asarray(v)
            cols.append(_np.asarray(arr, dtype=float).ravel())
        m = min(len(c) for c in cols)
        if m < 3:
            return 1.0, 2.0 * k
        X = _np.column_stack([c[:m] for c in cols])          # m × k control stats

        # Right-tail empirical transform per attacker, matching calc_local_pvalue:
        # p(x) = (#controls with LRT <= x) / m,  w(x) = -2 ln p(x).
        W = _np.empty_like(X)
        for i in range(k):
            col = X[:, i]
            counts = (col[None, :] <= col[:, None]).sum(axis=1)   # #{lrt <= col[a]}
            p = _np.clip(counts / m, 1.0 / m, 1.0)
            W[:, i] = -2.0 * _np.log(p)

        cov = _np.cov(W, rowvar=False)                       # k × k
        if not _np.all(_np.isfinite(cov)):
            return 1.0, 2.0 * k

        expected = 2.0 * k                                   # E[-2 ln p] = 2 per test
        off_diag_sum = float(cov.sum() - _np.trace(cov))     # = 2·Σ_{i<j} cov_ij
        var_w = 4.0 * k + off_diag_sum                       # Brown: theoretical diag
        if var_w <= 0:
            return 1.0, 2.0 * k
        c_scale = var_w / (2.0 * expected)
        dof = 2.0 * expected * expected / var_w
        if dof > 2.0 * k:                                    # negative net covariance
            return 1.0, 2.0 * k                              # → fall back to Fisher
        return float(c_scale), float(dof)

    def step_classifier(self, defender, attacker_agent=None):
        return self.step(attacker_agent=attacker_agent)

    def get_populations(self):
        # print("-------------------------")
        if 1 + self.args.a_control_size + self.args.b_control_size + self.args.beacon_size > 164:
            # raise Exception("Size of the population is too low!")
            pass # Suppress for synthetic data scenarios
        
        beacon_size = self.args.beacon_size + 1
        all_people = self.binary

        beacon_case = all_people[:beacon_size]
        beacon_control = all_people[beacon_size:beacon_size + self.args.b_control_size]
        attack_control = all_people[beacon_size + self.args.b_control_size:beacon_size + self.args.b_control_size + self.args.a_control_size]

        if self.args.evaluate:
            # --- FIX: Modulo arithmetic to prevent IndexError if episodes > population ---
            victim_ind = (self.episode % (self.args.beacon_size)) + 1
        else:
            victim_ind = np.random.randint(1, beacon_size)
            
        victim = beacon_case[victim_ind]
        victim_id = victim_ind - 1

        if getattr(self.args, 'calibrated_null', False):
            # Draw the non-member victim from the defense-control pool, which is
            # not in the beacon and is disjoint from the attacker-control set
            # the empirical p-value is computed against. The beacon keeps its
            # full N members; the spare slot is dropped instead of the victim.
            null_ind = self.episode % self.args.b_control_size
            victim = beacon_control[null_ind]
            # Offset by the member count, not the local beacon_size, which is one
            # larger because of the spare slot deleted below.
            victim_id = self.args.beacon_size + null_ind
            beacon_case = np.delete(beacon_case, 0, axis=0)
        elif np.random.random() < self.args.victim_prob:
            beacon_case = np.delete(beacon_case, 0, axis=0)
        else:
            beacon_case = np.delete(beacon_case, victim_ind, axis=0)

        if not self.args.evaluate:
            beacon_control = attack_control
        
        mafs = torch.Tensor(self.maf)
        nsnp_msk = (mafs == 0)
        mafs[nsnp_msk] = torch.as_tensor(0.001)
        return (torch.Tensor(beacon_case), torch.Tensor(beacon_control),
                torch.Tensor(attack_control), torch.Tensor(victim), victim_id,
                mafs)
