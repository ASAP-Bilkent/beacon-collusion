import time
import numpy as np
import copy
import os
import csv
import math
import random
import bisect

import torch

from utils import calculate_ind_lrt, lrt
from helpers.maf_tools import MAFHelper


class Attacker():
    def __init__(self, args, victim, control, maf_helper: MAFHelper):
        self.args = args
        self.mafs = maf_helper.get_mafs()
        # Private per-account victim copy. update() zeroes the queried locus in
        # this tensor to mark it consumed, so without a copy the first account to
        # query a locus would consume it for every account.
        self.victim = (victim.clone()
                       if getattr(args, 'private_account_evidence', False)
                       else victim)
        self.attacker_control = control
        self.maf_helper = maf_helper
        # Positions this account has queried in EITHER presence direction, so
        # act() never re-asks one, see the exclusion check in act() for why the
        # zeroing side effect alone isn't sufficient.
        self._asked_indices = set()
        # print("Initializing {} Attacker".format(self.args.attacker_type))

        self.beacon_actions = torch.tensor([], dtype=torch.float32)
        self.attacker_actions = torch.tensor([], dtype=torch.long)
        #########################

        if self.args.attacker_type == "optimal":
            self.optimal_queries = self._init_optimal_attacker()

        if self.args.attacker_type == "SF":
            self.diff_discriminative_queries = self._init_diff_discriminative_attacker()

        if self.args.attacker_type == "agent":
            self.categorized_maf = maf_helper.get_mafs_categories()
            self.maf_categories = self._init_agent_attacker()

        # print("Optimal Attacker", self.optimal_queries)
        # print("diff_discriminative_queries Attacker", self.diff_discriminative_queries)

        if self.args.attacker_type == "random":
            self.random_queries = self._init_random(self.args.user_risk)

        if self.args.attacker_type == "regular":
            self.regular_queries = self._init_regular(self.args.query_binary_path)

        # Initializing the control LRTS
        self.control_lrts = torch.zeros(size=(self.args.a_control_size,))
        self.victim_lrt = torch.as_tensor(0)
        self.identified = False
        self.identify_events = 0
        self.last_identify_event = False

        # Per-category stats for talking / cold_relay / gain_sharing mechanisms.
        if (getattr(self.args, 'talking', False)
                or getattr(self.args, 'cold_relay', False)
                or getattr(self.args, 'gain_sharing', False)):
            self._cat_lrt_gains  = [[] for _ in range(6)]
            self._cat_honesty    = [[] for _ in range(6)]
            self._recent_honesty = []

        # Gain-sharing: rolling window of recent LRT gains + within-partition bias.
        # Bias > 0 → prefer low-MAF end of partition; bias < 0 → prefer high-MAF end.
        if getattr(self.args, 'gain_sharing', False):
            self._recent_gains       = []
            self._partition_low_bias = 0.0

        # Fisher's local test: per-partition LRT accumulators (active attacker's queries only).
        if (getattr(self.args, 'fisher_local', False)
                or getattr(self.args, 'individual_local', False)
                or getattr(self.args, 'ebm_local', False)):
            self.local_victim_lrt   = torch.as_tensor(0.0)
            self.local_control_lrts = torch.zeros(self.args.a_control_size)
            self._local_n_queries   = 0
            # Analytic H0 moments of the victim LRT, accumulated online. Under H0
            # (target absent) with an honest beacon the response at queried locus j is
            # Bernoulli(p0_j), with p0_j = 1 - (1 - maf_j)^(2 * beacon_size), the
            # chance some other member carries the allele. The increment is
            # (log1 + log2*r) * g, so
            #     E0   = sum_j g_j (log1_j + log2_j p0_j)
            #     Var0 = sum_j g_j^2 log2_j^2 p0_j (1 - p0_j)
            # Both follow from the target's genome, the queried loci and public MAFs.
            # --- Executable (sampled-response) attacker ---
            # Per-locus tally of actual yes/no answers, so repeated queries of the same
            # locus by different accounts combine exactly rather than by summing
            # single-draw LLRs.
            self.exec_locus = {}
            self.exec_h_sum = 0.0
            self.exec_h_n   = 0
            self.local_null_mean    = 0.0
            self.local_null_var     = 0.0
            self._local_n_informative = 0
            # --- flip-rate-aware LLR (replaces the sequencing-error epsilon) ---
            # The Shringarpure-Bustamante increment assumes a sequencing error
            # eps=0.001, while an adversarial defender flips a much larger share of
            # responses. With flip probability f and p0 = P(some other member carries
            # the allele):
            #     P(yes | H1) = 1 - f
            #     P(yes | H0) = (1-f) p0 + f (1-p0)
            # Higher is more member-like. f is unknown to the attacker, so accumulate
            # over a grid and report sensitivity.
            self.f_grid = [0.02, 0.05, 0.10, 0.15, 0.20]
            self.llr_f      = [0.0] * len(self.f_grid)
            self.llr_f_mean = [0.0] * len(self.f_grid)
            self.llr_f_var  = [0.0] * len(self.f_grid)
            # H1 moment, giving the Jeffreys divergence J = E1 - E0 and the theoretical
            # discriminability d_theory = J / sqrt(Var0), which depends only on the
            # query set and public MAFs.
            self.llr_f_mean_h1 = [0.0] * len(self.f_grid)
            # Physical supply curve: loci the target carries in the informative
            # window maf < 1/(2*beacon_size), plus the MAF of what is actually
            # being consumed, to test the depletion signature directly.
            self._n_informative_rare = 0
            self._maf_sum_informative = 0.0
            # Shadow-beacon null: the same LLR machinery, but p0 comes from resampling
            # the beacon composition out of the panel (hypergeometric on the panel's
            # own carrier counts) rather than from population MAF.
            self.shadow_mean = [0.0] * len(self.f_grid)
            self.shadow_var  = [0.0] * len(self.f_grid)
            self.shadow_mean_h1 = [0.0] * len(self.f_grid)
            self.shadow_no_carrier = None      # set by Env
            self.locus_carrier_count = None    # set by Env

    #################################################################################################
    # Get Attacker state

    def get_state(self, include_k: bool = False, k: int = 1) -> float:
        """Return attacker state vector.

        Args:
            include_k: If True, appends normalized coalition size as an
                       extra feature (state_dim 10 → 11). Required when
                       evaluating a k-conditioned (adaptive) checkpoint.
            k:         Current number of colluding attackers. Only used
                       when include_k=True.
        """
        min_control_lrt  = torch.min(self.control_lrts)
        mean_control_lrt = torch.mean(self.control_lrts)

        parts = [
            self.maf_categories.flatten(),        # 6 dims
            self.victim_lrt.unsqueeze(0),         # 1 dim
            min_control_lrt.unsqueeze(0),          # 1 dim
            mean_control_lrt.unsqueeze(0),         # 1 dim
            self._calc_pvalue().unsqueeze(0),      # 1 dim
        ]

        if include_k:
            # Normalize by a reasonable max (20 covers all paper conditions)
            k_norm = torch.tensor([k / 20.0], dtype=torch.float32)
            parts.append(k_norm)                   # 1 dim → total 11

        return torch.cat(parts, dim=0)

    #################################################################################################
    # Update Attacker
    def update(self, beacon_action, attacker_action):
        beacon_action_tensor = torch.as_tensor(
            beacon_action, dtype=torch.float32)
        attacker_action_tensor = torch.as_tensor(
            [attacker_action], dtype=torch.long)

        self.attacker_action = attacker_action_tensor
        self.beacon_actions = torch.cat(
            (self.beacon_actions, beacon_action_tensor.unsqueeze(0)))
        self.attacker_actions = torch.cat(
            (self.attacker_actions, attacker_action_tensor))

        maf = self.mafs[self.attacker_actions]
        self.victim_lrt = self._calc_group_lrts(
            self.victim[self.attacker_actions], maf, self.beacon_actions, self.victim_lrt, True)
        self.control_lrts = self._calc_group_lrts(
            self.attacker_control[:, self.attacker_actions], maf, self.beacon_actions, self.control_lrts)

        # Zero out the queried SNP in this account's private victim tensor.  In
        # synchronized modes Env calls update() for every account, keeping these
        # copies in lock-step; in S1 only the querying account consumes it.
        if self.args.attacker_type == "agent":
            self.victim[int(attacker_action)] = 0

    #################################################################################################
    # Reward Calculation
    def calc_reward(self, beacon_action, step) -> float:
        done = False
        self.last_identify_event = False
        pvalue = self._calc_pvalue()

        # preward1 = -pvalue
        preward2 = beacon_action
        preward1 = -1

        total = preward1 + preward2

        if pvalue < 0.05:
            total += 40
            done = True
            self.identified = True
            self.identify_events += 1
            self.last_identify_event = True
            attacker_id = getattr(self, "attacker_id", "unknown")
            if os.getenv("DEBUG_CONTINUE_AFTER_IDENTIFY") == "1":
                print(f"⚠️ Attacker {attacker_id} would have identified the VICTIM (continue mode enabled)")
            else:
                print(f"⛔⛔⛔ Attacker {attacker_id} Identified the VICTIM (attacker_pvalue={float(pvalue):.6f}) ⛔⛔⛔")

        # print("Attacker Rewards: \n\t1-pvalue: {}\n\tutility: {}\n\ttotal: {}".format(preward1, preward1, total))
        # print("-----------------------------------------------------------------")

        return total, done, [preward1, preward2]

    #################################################################################################
    # Attacker Action

    def act(self, current_step, agent_action=None, snp_priority=None, shard_info=None,
            reserved_snps=None, snp_partition=None, consumed_snps=None):
        if self.args.attacker_type == "optimal":
            return self.optimal_queries[current_step]

        if self.args.attacker_type == "agent":
            presence = agent_action // 6
            category = agent_action % 6

            # Informed attacker. The trained policy spends much of its budget on loci
            # the target does not carry, which contribute nothing to the LRT. This
            # flag always asks about carried loci and prefers the rare band;
            # partitioning, pooling and coalition structure are unchanged.
            if getattr(self.args, 'informed_attacker', False):
                presence = 1
                category = 0

            if category < 0 or category >= len(self.maf_categories):
                raise ValueError(f"Invalid group number: {category}. Must be between 0 and {len(self.maf_categories) - 1}.")

            # Try the requested category first, then fall back to others
            attempts = 0
            while attempts < 6:
                if self.maf_categories[category] > 0:
                    start_ind = self.maf_helper.cat_start_ind[category]
                    mask = (1 - presence - self.victim[start_ind:]) * self.mafs[start_ind:]
                    indices = torch.nonzero(mask, as_tuple=True)[0] + start_ind
                    valid_indices = indices[self.categorized_maf[indices] == category]

                    # Rarity targeting: keep only SNPs at or below a MAF cap. Membership
                    # information lives in the rare tail, since a common allele is carried by
                    # some beacon member regardless of the target. Category 0 spans [0, 0.03),
                    # so this narrows within it. Falls back to the unfiltered set when no SNP
                    # under the cap remains.
                    _max_maf = getattr(self.args, 'max_maf', None)
                    if _max_maf is not None and _max_maf > 0 and len(valid_indices) > 0:
                        _rare = self.mafs[valid_indices] <= _max_maf
                        if torch.any(_rare):
                            valid_indices = valid_indices[_rare]

                    # Partitioning: restrict to this attacker's unique SNP block.
                    # Supports contiguous (start, end) for k-partitioning and
                    # frozenset of indices for random partitioning.
                    if snp_partition is not None and len(valid_indices) > 0:
                        if isinstance(snp_partition, (frozenset, set)):
                            in_part = torch.tensor(
                                [idx.item() in snp_partition for idx in valid_indices],
                                dtype=torch.bool
                            )
                        else:
                            p_start, p_end = snp_partition
                            in_part = (valid_indices >= p_start) & (valid_indices < p_end)
                        if torch.any(in_part):
                            valid_indices = valid_indices[in_part]
                            # Gain-sharing bias: shift within-partition ordering toward the
                            # MAF end where peers are finding higher LRT gains.
                            # SNPs are index-sorted by MAF, so index rank == MAF rank.
                            # bias > 0 → prefer low-MAF end (ascending index order).
                            # bias < 0 → prefer high-MAF end (descending index order).
                            if (hasattr(self, '_partition_low_bias')
                                    and abs(self._partition_low_bias) > 0.2
                                    and len(valid_indices) > 1):
                                descending = (self._partition_low_bias < 0)
                                order = torch.argsort(valid_indices, descending=descending)
                                valid_indices = valid_indices[order]

                    # Virtual sharding: restrict to this attacker's strided subset.
                    # Sort by snp_priority score first so high-value SNPs spread evenly.
                    if (shard_info is not None
                            and category in shard_info
                            and len(valid_indices) > 0):
                        offset, stride = shard_info[category]
                        if stride > 1:
                            if snp_priority:
                                scores = torch.tensor([
                                    snp_priority.get(idx.item(), 0.0)
                                    for idx in valid_indices
                                ])
                                order = torch.argsort(scores, descending=True)
                                valid_indices = valid_indices[order]
                            valid_indices = valid_indices[offset::stride]

                    # SNP reservation board: skip SNPs reserved by peers this round. Vectorised
                    # with torch.isin, since shared-category candidate lists can be large.
                    if reserved_snps and len(valid_indices) > 0:
                        reserved_t = torch.tensor(list(reserved_snps), dtype=valid_indices.dtype)
                        unreserved = valid_indices[~torch.isin(valid_indices, reserved_t)]
                        if len(unreserved) > 0:
                            valid_indices = unreserved
                        # else: all candidates reserved, fall through to reserved ones anyway

                    # Coalition-wide permanent exclusion. Unlike reserved_snps this never falls
                    # through: if every candidate has already been asked by someone in the
                    # coalition, the category fallback below tries a different category.
                    if consumed_snps and len(valid_indices) > 0:
                        consumed_t = torch.tensor(list(consumed_snps), dtype=valid_indices.dtype)
                        unconsumed = valid_indices[~torch.isin(valid_indices, consumed_t)]
                        valid_indices = unconsumed

                    # Never re-ask a position this account has already queried, in either
                    # presence direction. Zeroing self.victim[idx] only blocks a repeat under
                    # the same presence branch, so the position stays selectable under the
                    # other one.
                    if len(valid_indices) > 0 and getattr(self, '_asked_indices', None):
                        asked_t = torch.tensor(list(self._asked_indices), dtype=valid_indices.dtype)
                        fresh = valid_indices[~torch.isin(valid_indices, asked_t)]
                        if len(fresh) > 0:
                            valid_indices = fresh
                        # else: every remaining candidate already asked, fall
                        # through and allow a repeat rather than raise, since the
                        # category-fallback loop below has nowhere better to go.

                    if len(valid_indices) > 0:
                        # SNP priority queue: prefer SNPs with highest observed LRT gain
                        if snp_priority is not None and len(snp_priority) > 0:
                            valid_set = set(valid_indices.tolist())
                            best_snp, best_score = None, float('-inf')
                            for snp_idx, score in snp_priority.items():
                                if snp_idx in valid_set and score > best_score:
                                    best_score = score
                                    best_snp = snp_idx
                            if best_snp is not None:
                                if not hasattr(self, '_asked_indices'):
                                    self._asked_indices = set()
                                self._asked_indices.add(best_snp)
                                return best_snp
                        chosen = valid_indices[0].item()
                        if not hasattr(self, '_asked_indices'):
                            self._asked_indices = set()
                        self._asked_indices.add(chosen)
                        return chosen

                # Try next category
                category = (category + 1) % 6
                attempts += 1

            raise RuntimeError(f"No valid index found for category {category}")

        if self.args.attacker_type == "random":
            return self.random_queries[current_step]

        if self.args.attacker_type == "regular":
            return self.regular_queries[current_step]

        if self.args.attacker_type == "SF":
            return self.diff_discriminative_queries[current_step]

    #################################################################################################

    def _calc_group_lrts_all_snps(self, genome, maf, response) -> torch.Tensor:
        error = 0.001

        one_minus_maf = (1 - maf)
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)

        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - \
            torch.log(DN_i * (1 - error * DN_i_1))

        lrt = (log1 + log2 * response).unsqueeze(0) * genome
        return lrt
    # LRT PVALUES

    def _calc_group_lrts(self, genome, maf, response, prev_beacon_lrts, one_dim=False) -> torch.Tensor:
        error = 0.001

        one_minus_maf = (1 - maf[-1])  # Last MAF value
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)

        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - \
            torch.log(DN_i * (1 - error * DN_i_1))

        if one_dim:
            last_genome = genome[-1]
        else:
            last_genome = genome[:, -1]

        lrt = (log1 + log2 * response[-1]).mul(last_genome)
        updated_beacon_lrts = prev_beacon_lrts + lrt

        return updated_beacon_lrts

    def _calc_pvalue(self):
        victim_lrt   = copy.deepcopy(self.victim_lrt)
        control_lrts = copy.deepcopy(self.control_lrts)
        # Use the number of controls this attacker actually holds as the
        # denominator, since private_controls gives each attacker a smaller subset.
        # Falls back to args.a_control_size when control_lrts is empty.
        n = len(control_lrts) if len(control_lrts) > 0 else self.args.a_control_size
        pvalue = torch.sum(victim_lrt >= control_lrts) / n
        return torch.Tensor(pvalue)

    #################################################################################################
    # Talking mechanism (--talking flag)

    def record_query(self, attacker_action, beacon_action: float, lrt_gain: float, categorized_maf=None):
        """Record per-category stats for the talking mechanism. Call after update()."""
        if categorized_maf is not None:
            snp_idx = int(attacker_action)
            if snp_idx < len(categorized_maf):
                cat = int(categorized_maf[snp_idx].item())
                if 0 <= cat < 6:
                    self._cat_lrt_gains[cat].append(lrt_gain)
                    self._cat_honesty[cat].append(beacon_action)
        self._recent_honesty.append(beacon_action)
        if len(self._recent_honesty) > 20:
            self._recent_honesty.pop(0)
        if hasattr(self, '_recent_gains'):
            self._recent_gains.append(lrt_gain)
            if len(self._recent_gains) > 10:
                self._recent_gains.pop(0)

    def build_message(self) -> dict:
        """Return the broadcast message dict consumed by peer attackers."""
        cat_gains = [
            float(sum(g) / len(g)) if g else 0.0
            for g in self._cat_lrt_gains
        ]
        cat_honesty = [
            float(sum(h) / len(h)) if h else 1.0
            for h in self._cat_honesty
        ]
        rec = self._recent_honesty
        if len(rec) >= 10:
            suspicion_trend = sum(rec[-5:]) / 5 - sum(rec[-10:-5]) / 5
        else:
            suspicion_trend = 0.0
        # risk_proxy: 0 = GBD fully trusts this attacker, 1 = fully suspicious.
        risk_proxy = 1.0 - (sum(rec[-10:]) / len(rec[-10:])) if len(rec) >= 1 else 0.0
        recent_gains = getattr(self, '_recent_gains', [])
        recent_avg_gain = float(sum(recent_gains) / len(recent_gains)) if recent_gains else 0.0

        return {
            "attacker_id":      getattr(self, "attacker_id", -1),
            "p_value":          float(self._calc_pvalue()),
            "category_gains":   cat_gains,
            "avg_honesty":      cat_honesty,
            "suspicion_trend":  suspicion_trend,
            "risk_proxy":       float(risk_proxy),
            "recent_avg_gain":  recent_avg_gain,
        }

    def update_partition_bias(self, peer_msgs: list, my_partition_id: int):
        """Shift within-partition SNP preference based on where peers are finding gains.

        If a peer in a lower-MAF partition is getting higher gains than peers in
        higher-MAF partitions, this victim's discriminative SNPs tend to be in the
        low-MAF range — so bias toward the low end of my own partition, and vice versa.
        The bias only kicks in when the signal is meaningful (gain > 0.005).
        """
        if not peer_msgs:
            return
        best = max(peer_msgs, key=lambda m: m.get('recent_avg_gain', 0.0))
        best_gain = best.get('recent_avg_gain', 0.0)
        if best_gain <= 0.005:
            return
        best_pid = best.get('attacker_id', my_partition_id)
        if best_pid < my_partition_id:
            # High gain in lower-MAF partition → prefer low end of mine
            self._partition_low_bias = min(1.0, self._partition_low_bias + 0.15)
        elif best_pid > my_partition_id:
            # High gain in higher-MAF partition → prefer high end of mine
            self._partition_low_bias = max(-1.0, self._partition_low_bias - 0.15)

    #################################################################################################
    # Fisher's Local P-value (per-partition independent test)

    def finalize_exec(self, h_bar=None, absent_yes=False):
        """Exact LLR for the executable attacker, from observed yes/no answers only.

        Flipping is one-directional (the beacon never invents a YES), so for a
        locus queried k times with any YES observed, presence is certain and the
        evidence is log(p0/p1) -- the honesty rate cancels exactly.  With no YES
        in k draws the evidence is attenuated toward zero as h falls, which is
        what a coalition defeats by re-querying.

        Uses an AGGREGATE honesty rate: a per-answer h would itself disclose
        presence (h<1 is only ever set on a locus the beacon holds).

        Returns (llr, E0, Var0, n_loci, h_bar); lower llr = more member-like.
        """
        err = 0.001
        N = self.args.beacon_size
        if h_bar is None:
            h_bar = (self.exec_h_sum / self.exec_h_n) if self.exec_h_n else 1.0
        h_bar = min(max(float(h_bar), 1e-6), 1.0 - 1e-9)
        _tiny = 1e-300
        llr = 0.0; e0 = 0.0; v0 = 0.0; n = 0
        for _loc, d in self.exec_locus.items():
            om = 1.0 - d['maf']
            Dn = om ** (2 * N); Dn1 = om ** (2 * N - 2)
            p0 = 1.0 - Dn                 # P(locus present | victim NOT a member)
            p1 = 1.0 - err * Dn1          # P(locus present | victim IS a member)
            k = d['n']
            q = (1.0 - h_bar) ** k        # P(all k answers are NO | present)
            if absent_yes:
                # Absent -> YES, so a NO can only come from a lie about a present allele:
                # NO is conclusive of presence and h cancels from it, while a YES is
                # ambiguous.
                py0 = p0 * h_bar + (1.0 - p0)      # P(all YES | H0), k=1 form
                py1 = p1 * h_bar + (1.0 - p1)
                if k > 1:
                    py0 = p0 * (h_bar ** k) + (1.0 - p0)
                    py1 = p1 * (h_bar ** k) + (1.0 - p1)
                l_no = math.log(max(p0, _tiny) / max(p1, _tiny))
                l_yes = math.log(max(py0, _tiny)) - math.log(max(py1, _tiny))
                pyes0 = py0                        # P(no NO observed | H0)
                # note: "yes" here means NO NO was seen in k draws
                l_a, l_b, pa = l_yes, l_no, pyes0
                obs_a = (d['n_yes'] == d['n'])     # every draw was YES
            else:
                l_yes = math.log(max(p0, _tiny) / max(p1, _tiny))
                l_no = (math.log(max(p0 * q + (1.0 - p0), _tiny))
                        - math.log(max(p1 * q + (1.0 - p1), _tiny)))
                l_a, l_b, pa = l_yes, l_no, p0 * (1.0 - q)
                obs_a = (d['n_yes'] > 0)
            llr += d['g'] * (l_a if obs_a else l_b)
            e0 += d['g'] * (pa * l_a + (1.0 - pa) * l_b)
            v0 += (d['g'] ** 2) * pa * (1.0 - pa) * (l_a - l_b) ** 2
            n += 1
        return llr, e0, v0, n, h_bar

    def update_local(self, beacon_action, attacker_action: int, victim_snp_value: float,
                     exec_r=None, exec_h=None):
        """Accumulate LRT for only this attacker's own queries (Fisher's local test).

        Must be called with the victim's genome value BEFORE update() zeroes it.
        Uses the same LRT formula as _calc_group_lrts but operates on a per-attacker
        accumulator rather than the shared coalition victim_lrt.
        """
        error = 0.001
        maf_val = self.mafs[attacker_action]
        one_minus_maf = 1.0 - maf_val
        DN_i   = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)
        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = (torch.log((error * DN_i_1) * (1.0 - DN_i))
                - torch.log(DN_i * (1.0 - error * DN_i_1)))

        # The value the attacker's own likelihood ratio consumes. By default this is
        # beacon_action, the expected quantity h * x_true; with
        # --exec_lrt_uses_response it is the sampled yes/no the beacon emitted.
        # Only the attacker's view changes.
        if getattr(self.args, 'exec_lrt_uses_response', False) and exec_r is not None:
            resp = torch.as_tensor(float(exec_r), dtype=torch.float32)
        else:
            resp = torch.as_tensor(beacon_action, dtype=torch.float32).flatten()[-1]

        self.local_victim_lrt = (
            self.local_victim_lrt + (log1 + log2 * resp) * victim_snp_value
        )
        ctrl_genome = self.attacker_control[:, attacker_action]
        self.local_control_lrts = self.local_control_lrts + (log1 + log2 * resp) * ctrl_genome
        self._local_n_queries += 1

        # Executable attacker: bank the observed yes/no for this locus.
        if exec_r is not None and victim_snp_value:
            d = self.exec_locus.setdefault(
                int(attacker_action),
                {'n_yes': 0, 'n': 0, 'maf': float(maf_val), 'g': float(victim_snp_value)})
            d['n_yes'] += int(float(exec_r) >= 0.5)
            d['n']     += 1
            if exec_h is not None:
                self.exec_h_sum += float(exec_h)
                self.exec_h_n   += 1

        # Analytic H0 moments for this locus (see __init__ for the derivation).
        # Loci the target does not carry contribute nothing to the LRT and nothing
        # here either, since every term carries the factor victim_snp_value.
        if victim_snp_value:
            p0 = float(1.0 - (1.0 - maf_val).pow(2 * self.args.beacon_size))
            g = float(victim_snp_value)
            self.local_null_mean += g * (float(log1) + float(log2) * p0)
            self.local_null_var  += (g ** 2) * (float(log2) ** 2) * p0 * (1.0 - p0)
            self._local_n_informative += 1
            _m = float(maf_val)
            self._maf_sum_informative += _m
            # The informative window: above maf ~ 1/(2*beacon_size) some other
            # member almost surely carries the allele, so the answer is YES
            # regardless of membership and the query says nothing.
            if _m < 1.0 / (2.0 * self.args.beacon_size):
                self._n_informative_rare += 1

            # Flip-rate-aware LLR over the f grid (see __init__).
            r = float(resp)
            r_bin = 1.0 if r >= 0.5 else 0.0   # responses are continuous here
            for _i, _f in enumerate(self.f_grid):
                p_yes_h1 = 1.0 - _f
                p_yes_h0 = (1.0 - _f) * p0 + _f * (1.0 - p0)
                # guard the logs; p0 hits 1.0 for common alleles, which would make
                # P(no | H0) zero and the "no" LLR infinite.
                _e = 1e-12
                llr_yes = math.log(max(p_yes_h1, _e) / max(p_yes_h0, _e))
                llr_no = math.log(max(_f, _e) / max(1.0 - p_yes_h0, _e))
                self.llr_f[_i] += g * (llr_yes if r_bin else llr_no)
                # H1 moment: under H1 the response is Bernoulli(1-f).
                self.llr_f_mean_h1[_i] += g * ((1.0 - _f) * llr_yes + _f * llr_no)
                # Analytic H0 moments of THIS statistic: under H0 the response is
                # Bernoulli(p_yes_h0), so the increment is a two-point variable.
                q = p_yes_h0
                self.llr_f_mean[_i] += g * (q * llr_yes + (1.0 - q) * llr_no)
                self.llr_f_var[_i] += (g ** 2) * q * (1.0 - q) * (llr_yes - llr_no) ** 2

            # Same moments under the shadow-beacon null (see __init__). Only p0
            # changes: the chance that some OTHER panel member in a resampled
            # beacon carries this locus, given the target carries it.
            if self.shadow_no_carrier is not None and self.locus_carrier_count is not None:
                _c = int(self.locus_carrier_count[attacker_action]) - int(round(g))
                _c = max(0, min(_c, len(self.shadow_no_carrier) - 1))
                p0s = 1.0 - float(self.shadow_no_carrier[_c])
                for _i, _f in enumerate(self.f_grid):
                    qs = (1.0 - _f) * p0s + _f * (1.0 - p0s)
                    _e = 1e-12
                    ly = math.log(max(1.0 - _f, _e) / max(qs, _e))
                    ln_ = math.log(max(_f, _e) / max(1.0 - qs, _e))
                    self.shadow_mean[_i] += g * (qs * ly + (1.0 - qs) * ln_)
                    self.shadow_var[_i] += (g ** 2) * qs * (1.0 - qs) * (ly - ln_) ** 2
                    self.shadow_mean_h1[_i] += g * ((1.0 - _f) * ly + _f * ln_)

    def calc_local_pvalue(self) -> float:
        """P-value derived exclusively from this attacker's own partition queries."""
        if self._local_n_queries == 0:
            return 1.0
        n = len(self.local_control_lrts)
        if n == 0:
            return 1.0
        count = float(torch.sum(self.local_victim_lrt >= self.local_control_lrts))
        return max(count / n, 1e-6)

    #################################################################################################
    # Optimal Attacker

    def _init_optimal_attacker(self):
        maf_values = self.mafs

        maf_mask = torch.logical_or(maf_values >= 0.05, maf_values == 0.001)
        filtered_indices = torch.argsort(maf_values[maf_mask])
        sorted_gene_indices = torch.nonzero(maf_mask).squeeze()[
            filtered_indices]

        mask_one = (self.victim[sorted_gene_indices]
                    == 1)  # Get the SNPs with value 1
        mask_zero = (self.victim[sorted_gene_indices]
                     == 0)  # Get the SNPs with value 0

        # Use torch.cat to concatenate the indices
        # [:self.args.max_queries]
        queries = torch.cat(
            (sorted_gene_indices[mask_one], sorted_gene_indices[mask_zero]))

        return queries

    def _init_diff_discriminative_attacker(self, k=5):
        victim_lrts = self._calc_group_lrts_all_snps(self.victim, self.mafs, 1)
        control_lrts = self._calc_group_lrts_all_snps(
            self.attacker_control, self.mafs, 1)

        discriminative_powers = victim_lrts.mean(
            dim=0) - control_lrts.mean(dim=0)

        flipped_victim_lrts = self._calc_group_lrts_all_snps(
            self.victim, self.mafs, 0)
        flipped_control_lrts = self._calc_group_lrts_all_snps(
            self.attacker_control, self.mafs, 0)
        flipped_discriminative_powers = flipped_victim_lrts.mean(
            dim=0) - flipped_control_lrts.mean(dim=0)

        delta_discriminative_powers = discriminative_powers - flipped_discriminative_powers

        sorted_gene_indices = torch.argsort(
            delta_discriminative_powers, descending=False)
        mask_one = (self.victim[sorted_gene_indices]
                    == 1)  # Get the SNPs with value 1
        mask_zero = (self.victim[sorted_gene_indices]
                     == 0)  # Get the SNPs with value 0
        # [:self.args.max_queries]
        queries = torch.cat(
            (sorted_gene_indices[mask_one], sorted_gene_indices[mask_zero]))
        print("Attacker Queries: ", sorted_gene_indices)
        return queries

    def _get_maf_category(self, maf_value):
        for i in range(len(self.maf_thresholds) - 1):
            if self.maf_thresholds[i] <= maf_value < self.maf_thresholds[i + 1]:
                return i
        return len(self.maf_thresholds) - 1

    def _init_agent_attacker(self):
        # _, category_coutns = torch.unique_consecutive(
        #     self.categorized_maf[self.victim.to(torch.bool)], return_counts=True)
        category_coutns = torch.bincount(self.categorized_maf[self.victim.to(torch.bool)], minlength=6)

        return category_coutns

    def _divide_mafs(self, numbers, num_groups):
        sorted_numbers = torch.sort(numbers).values

        n = len(sorted_numbers)
        split_indices = [int(n * i / num_groups) for i in range(1, num_groups)]
        group_boundaries = [sorted_numbers[0].item()] + [sorted_numbers[idx].item()
                                                         for idx in split_indices] + [sorted_numbers[-1].item()]
        group_boundaries = [round(bound, 3) if bound >
                            1e-6 else 0.0 for bound in group_boundaries]
        group_counts = []

        for i in range(num_groups):
            if i == 0:
                count = torch.sum(numbers < group_boundaries[i+1]).item()
            elif i == num_groups - 1:
                count = torch.sum(numbers >= group_boundaries[i]).item()
            else:
                count = torch.sum((numbers >= group_boundaries[i]) & (
                    numbers < group_boundaries[i+1])).item()

            group_counts.append(count)

        return group_boundaries, group_counts


    def _init_random(self, risk_level):
        sorted_mafs, indices = torch.sort(self.mafs)
        weights = torch.linspace(0, 1, len(self.mafs.clone()))
        weights = weights ** (1 - risk_level)
        weights = weights / weights.sum()
        samples = torch.multinomial(weights, self.args.max_queries, replacement=True)
        queries = indices[samples]
        return queries

    def _init_regular(self, query_binary_path):
        if not os.path.exists(query_binary_path):
            raise FileNotFoundError(f"Query binary file not found at {query_binary_path}")

        queries = np.load(query_binary_path, allow_pickle=True)
        np.random.shuffle(queries)

        if len(queries) < self.args.max_queries:
            raise ValueError(f"Insufficient queries in the binary file. Expected at least {self.args.max_queries}, got {len(queries)}.")

        return torch.from_numpy(queries[:self.args.max_queries]).long()

        
