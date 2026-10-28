import numpy as np
import copy
import os
import csv
import math
import random
import torch
from utils import calculate_ind_lrt, calculate_pvalues

class Beacon():
    error = 0.001
    
    def __init__(self, args, case, control, mafs, victim_id):
        self.args = args
        self.mafs = mafs
        self.beacon_case = case
        self.beacon_control = control
        self.victim_id = victim_id

        self.attacker_actions = torch.tensor([], dtype=torch.long)
        self.responses = torch.tensor([], dtype=torch.float32)

        self.sum_probs = 1
        self.beacon_lrts = torch.zeros(size=(self.args.beacon_size,))
        # --- partition-robust risk ---
        # The coalition decides on min_i p_i over separately tested accumulators,
        # not the pooled p-value the defender is scored by. Mirror the global
        # accumulator once per MAF block so the defender can be penalised on its
        # worst block. The blocks are a monitored family, not an exhaustive one,
        # so worst-block risk lower-bounds the true worst case.
        self.n_blocks = 6
        # same cut points as helpers.maf_tools.MAFHelper.get_category
        self.snp_block = torch.zeros(len(self.mafs), dtype=torch.long)
        for _b, _hi in enumerate([0.03, 0.10, 0.20, 0.30, 0.40]):
            self.snp_block[self.mafs >= _hi] = _b + 1
        self.beacon_lrts_blk = torch.zeros(self.n_blocks, self.args.beacon_size)
        self.control_lrts_blk = torch.zeros(self.n_blocks, self.args.b_control_size)
        self.blk_counts = torch.zeros(self.n_blocks)
        self.control_lrts = torch.zeros(size=(self.args.b_control_size,))
        self.pvalues = self._calc_pvalues()
        # Physical-account histories affect GBD observations only. Global
        # accumulators remain available for legacy logging and rewards.
        self.defender_histories = {}
        self.defender_query_count = 0

        if self.args.beacon_type == "baseline":
            self.baseline_mafs = self._init_baseline_beacon()

        if self.args.beacon_type == "strategic":
            self.strategy_positions = self._init_strategic_beacon()

        if self.args.beacon_type == "qbudget":
            # p is the defender's strength knob: budget = -log(p), so a larger p is a
            # tighter budget and a stronger defense.
            p = float(getattr(self.args, 'qbudget_p', 0.1) or 0.1)
            self.initial_budget_val = -torch.log(torch.tensor(p))
            # Per-user budget, keyed by account id.
            self.user_budgets = {}
            
        if self.args.beacon_type == "rtf":
            self.old_pvalues = self.pvalues.clone()
            self.no_change_count = torch.zeros(size=(self.args.beacon_size,), dtype=torch.long)
            self.rare_threshold = 0.1
            
        if self.args.beacon_type == "OG-K":
            self.K = 1
            self._init_OG_beacon()

    def get_state(self, attacker_action, current_step)->float:
        if self._uses_per_account_defender():
            history = self._defender_history()
            return history.get_state(attacker_action, history.defender_query_count)
        # 1. Calculate Standard Features (~18 features)
        b_lrts_afterN, c_lrts_afterN = self.update(1, attacker_action, calculation_mode=True)
        pvalues_afterN = calculate_pvalues(b_lrts_afterN, c_lrts_afterN, self.args.b_control_size)

        b_lrts_after, c_lrts_after = self.update(0, attacker_action, calculation_mode=True)
        pvalues_after = calculate_pvalues(b_lrts_after, c_lrts_after, self.args.b_control_size)

        min_lrt_case = torch.min(self.beacon_lrts)
        mean_lrt_case = torch.mean(self.beacon_lrts)
        min_lrt_case_afterN = torch.min(b_lrts_afterN)
        mean_lrt_case_afterN = torch.mean(b_lrts_afterN)
        min_lrt_case_after = torch.min(b_lrts_after)
        mean_lrt_case_after = torch.mean(b_lrts_after)
        min_pvalue = torch.min(self.pvalues)
        min_pvalue_after = torch.min(pvalues_after)
        min_pvalue_afterN = torch.min(pvalues_afterN)

        min_lrt_control = torch.min(self.control_lrts)
        mean_lrt_control = torch.mean(self.control_lrts)
        thresh_lrt = torch.sort(self.control_lrts)[0][int(0.05*self.args.b_control_size)]
        min_lrt_control_afterN = torch.min(c_lrts_afterN)
        mean_lrt_control_afterN = torch.mean(c_lrts_afterN)
        min_lrt_control_after = torch.min(c_lrts_after)
        mean_lrt_control_after = torch.mean(c_lrts_after)
        
        base_state = [self.mafs[attacker_action], min_lrt_case, mean_lrt_case, min_lrt_case_afterN,
                      mean_lrt_case_afterN, min_lrt_case_after, mean_lrt_case_after, min_pvalue,
                      min_pvalue_after, min_pvalue_afterN, min_lrt_control, mean_lrt_control,
                      thresh_lrt, min_lrt_control_afterN, mean_lrt_control_afterN, min_lrt_control_after,
                      mean_lrt_control_after, self.sum_probs/(current_step+1)]

        if getattr(self.args, 'partition_state', False):
            # The 18 base features summarise members and controls in aggregate, so none
            # of them can express that one account is sweeping the rare band. Add, per
            # MAF block, the minimum p-value over members and the share of the budget
            # spent there. Members rather than the victim, since a real beacon knows
            # its own members but not the victim's identity.
            MINQ = getattr(self.args, 'min_block_queries', 20)
            blk_p, blk_cnt = [], []
            for b in range(self.n_blocks):
                if self.blk_counts[b] < MINQ:
                    blk_p.append(1.0)          # not yet testable -> no risk signal
                else:
                    pv = calculate_pvalues(self.beacon_lrts_blk[b],
                                           self.control_lrts_blk[b],
                                           self.args.b_control_size)
                    blk_p.append(float(torch.min(pv)))
                # Absolute count on a fixed scale rather than a share of the episode, so
                # the signal does not depend on episode length.
                blk_cnt.append(min(float(self.blk_counts[b].item()) / 100.0, 2.0))
            blk_share = blk_cnt
            base_state = base_state + blk_p + blk_share + [min(blk_p)]

        return torch.tensor(base_state, dtype=torch.float32).numpy().tolist()

    def _uses_per_account_defender(self):
        return (self.args.beacon_type == "agent"
                and getattr(self.args, "per_account_defender", False))

    def _defender_history(self):
        account = getattr(self, "current_agent_id", 0)
        if account not in self.defender_histories:
            local_args = copy.copy(self.args)
            local_args.per_account_defender = False
            # Genomes/MAFs are shared read-only; all history tensors are new.
            self.defender_histories[account] = Beacon(
                local_args, self.beacon_case, self.beacon_control,
                self.mafs, self.victim_id)
        return self.defender_histories[account]

    def update(self, beacon_action, attacker_action, calculation_mode: bool = False):
        beacon_action_tensor = torch.as_tensor(beacon_action, dtype=torch.float32)
        attacker_action_tensor = torch.as_tensor([attacker_action], dtype=torch.long)

        if calculation_mode:
            attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))
            res = torch.cat((self.responses, beacon_action_tensor.unsqueeze(0)))
            b_lrts = self._calc_group_lrts(self.beacon_case[:, attacker_actions], self.mafs[attacker_actions], res, self.beacon_lrts.clone())
            c_lrts = self._calc_group_lrts(self.beacon_control[:, attacker_actions], self.mafs[attacker_actions], res, self.control_lrts.clone())
            return b_lrts, c_lrts

        self.attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))
        self.responses = torch.cat((self.responses, beacon_action_tensor.unsqueeze(0)))

        maf = self.mafs[self.attacker_actions]
        self.beacon_lrts = self._calc_group_lrts(self.beacon_case[:, self.attacker_actions], maf, self.responses, self.beacon_lrts, update_qb=True)
        self.control_lrts = self._calc_group_lrts(self.beacon_control[:, self.attacker_actions], maf, self.responses, self.control_lrts)

        _blk = int(self.snp_block[int(attacker_action)])
        self.beacon_lrts_blk[_blk] = self._calc_group_lrts(
            self.beacon_case[:, self.attacker_actions], maf, self.responses,
            self.beacon_lrts_blk[_blk])
        self.control_lrts_blk[_blk] = self._calc_group_lrts(
            self.beacon_control[:, self.attacker_actions], maf, self.responses,
            self.control_lrts_blk[_blk])
        self.blk_counts[_blk] += 1

        if isinstance(beacon_action, list):
            self.sum_probs += beacon_action[0]
        else:
            self.sum_probs += beacon_action

        self.defender_query_count += 1
        if self._uses_per_account_defender():
            history = self._defender_history()
            history.update(beacon_action, attacker_action)
            history.pvalues = history._calc_pvalues()

    def _victim_p(self, b_lrts, c_lrts):
        """Victim's lower-tail empirical p-value, matching utils.calculate_pvalues.
        Handles --calibrated_null, where the victim is a non-member drawn from the
        control pool and therefore lives in c_lrts rather than b_lrts."""
        n_members = b_lrts.shape[0]
        vid = int(self.victim_id)
        vl = b_lrts[vid] if vid < n_members else c_lrts[vid - n_members]
        return torch.sum(vl >= c_lrts).float() / self.args.b_control_size

    def partition_risk(self, min_block_queries=20):
        """Smallest victim p-value over the monitored blocks, and the pooled one.

        This is the quantity the any-one coalition actually acts on. A block with
        too few queries has a degenerate empirical null (the controls have
        accumulated nothing), so it is gated out: without the gate the defender
        panics from the first query and suppresses everything.
        """
        worst = self._victim_p(self.beacon_lrts, self.control_lrts)
        for b in range(self.n_blocks):
            if self.blk_counts[b] < min_block_queries:
                continue
            pb = self._victim_p(self.beacon_lrts_blk[b], self.control_lrts_blk[b])
            if pb < worst:
                worst = pb
        return worst

    def calc_reward(self, beacon_action)->float:
        done = False
        if self.args.beacon_type == "rtf":
            self.old_pvalues = self.pvalues.clone()
        self.pvalues = self._calc_pvalues()
        # Under --calibrated_null the victim is drawn from the defense-control pool
        # and is not a beacon member, so victim_id indexes past self.pvalues. That
        # individual's LRT is tracked in control_lrts, so score it there. It feeds
        # only the defender's reward and the early-stop flag.
        if getattr(self.args, 'partition_risk_reward', False):
            # Penalise the worst separately-tested block, not the pooled stream.
            preward = self.partition_risk(
                getattr(self.args, 'min_block_queries', 20))
        elif int(self.victim_id) < self.pvalues.shape[0]:
            preward = self.pvalues[self.victim_id]
        else:
            null_ind = int(self.victim_id) - self.pvalues.shape[0]
            victim_lrt = self.control_lrts[null_ind]
            preward = torch.sum(victim_lrt >= self.control_lrts) / self.args.b_control_size
        ureward = beacon_action
        total = ureward + preward
    
        if preward < 0.05:
            total -= 15
            done=True
        return total, done, [preward, ureward]

    def _calc_victim_lrt_increment(self, attacker_action, beacon_action):
        maf = self.mafs[attacker_action]
        genome_val = self.beacon_case[self.victim_id, attacker_action]

        if isinstance(beacon_action, (list, tuple, np.ndarray)):
            response = float(beacon_action[0])
        elif isinstance(beacon_action, torch.Tensor):
            response = float(beacon_action.item())
        else:
            response = float(beacon_action)

        DN_i = (1 - maf).pow(2 * self.args.beacon_size)
        DN_i_1 = (1 - maf).pow(2 * self.args.beacon_size - 2)

        log1 = torch.log(DN_i) - torch.log(0.001 * DN_i_1)
        log2 = torch.log((0.001 * DN_i_1 * (1 - DN_i))) - torch.log(DN_i * (1 - 0.001 * DN_i_1))

        lrt_inc = (log1 + log2 * response) * genome_val
        return max(0.0, float(lrt_inc))

    def get_state_dim(self):
        # 18 base + (6 per-block victim p-values + 6 per-block query shares + 1
        # worst-block risk) when partition-aware state is enabled.
        return 31 if getattr(self.args, 'partition_state', False) else 18

    def act(self, attacker_action, max_pvalue_change_threshold):
        if self.args.beacon_type == "random":
            has_snp = torch.sum(self.beacon_case[:, attacker_action]) > 0
            if not has_snp: return 1
            else:
                return 1 if random.random() < 0.75 else 0

        if self.args.beacon_type == "truth": return 1

        # Honest beacon (validation control): answer the true presence of the
        # queried allele. This is the standard Shringarpure-Bustamante response,
        # used as the no-defense reference.
        if self.args.beacon_type == "honest":
            return 1 if torch.sum(self.beacon_case[:, attacker_action]) > 0 else 0
        
        if self.args.beacon_type == "beacon_strategy":
            return self.beacon_strategy_pvalue_change(self.beacon_case, attacker_action, max_pvalue_change_threshold)
        
        if self.args.beacon_type == "baseline":
            if self.mafs[attacker_action] in self.baseline_mafs:
                return 0
            else:
                return 1
        
        if self.args.beacon_type == "qbudget":
            # Per-User Budget Logic
            agent_id = getattr(self, 'current_agent_id', 0)
            
            if agent_id not in self.user_budgets:
                 self.user_budgets[agent_id] = torch.full(size=(self.args.beacon_size,), fill_value=self.initial_budget_val)

            self._update_budget(attacker_action, self.mafs[attacker_action], agent_id)
            
            current_budgets = self.user_budgets[agent_id]
            valid_individuals = current_budgets > 0
            remaining_genomes = self.beacon_case[valid_individuals, attacker_action]
            
            has_snp_after_budget_check = torch.sum(remaining_genomes) > 0
            if not has_snp_after_budget_check: return 0
            return 1
        
        if self.args.beacon_type == "strategic":
            if attacker_action in self.strategy_positions:
                return 0
            else:
                return 1

        if self.args.beacon_type == "rtf":
            return self._real_time_flipping(attacker_action)
        
        if self.args.beacon_type == "OG-K":
            # (OG-K implementation omitted for brevity, assumed unchanged from previous correct version)
            # Re-inserting the previous logic here to ensure file completeness
            self.eta_b += self.eta_in_beac[:, attacker_action]
            self.eta_nb += self.eta_not_in_beac[:, attacker_action]
            k_low_eta_nb, _ = torch.topk(self.eta_nb, k=self.K, largest=False)
            k_low_delt_nb, _ = torch.topk(self.delt_nb, k=self.K, largest=False)
            self.eta_n_mean = torch.mean(k_low_eta_nb)
            self.delt_n_mean = torch.mean(k_low_delt_nb)
            if ((self.delt_b - self.delt_n_mean) >= (self.eta_n_mean - self.eta_b)).all() == False and attacker_action in self.selected_snps:
                self.delt_b += self.Delt_in_Beac[:, attacker_action]
                self.delt_nb += self.Delt_Not_in_Beac[:, attacker_action]
                return 0
            return 1
        
        return 1 # Default fallthrough

    def _init_strategic_beacon(self, k=0.05):
        # Optional arm (--strategic_rare_band). The delta-discriminative-power
        # ranking below follows Wan et al. (2017). This arm keeps the same 5%
        # budget but suppresses the K rarest alleles present in the beacon.
        if getattr(self.args, 'strategic_rare_band', False):
            K = int(self.args.gene_size * k)
            carriers = self.beacon_case.sum(dim=0)
            present = torch.nonzero(carriers > 0, as_tuple=True)[0]
            mafs_t = torch.as_tensor(self.mafs)
            order = torch.argsort(mafs_t[present])
            return present[order[:K]]

        beacon_lrts = self._calc_group_lrts_all_snps(self.beacon_case, self.mafs, 1)
        control_lrts = self._calc_group_lrts_all_snps(self.beacon_control, self.mafs, 1)
        discriminative_powers = beacon_lrts.mean(dim=0) - control_lrts.mean(dim=0)
        flipped_beacon_lrts = self._calc_group_lrts_all_snps(self.beacon_case, self.mafs, 0)
        flipped_control_lrts = self._calc_group_lrts_all_snps(self.beacon_control, self.mafs, 0)
        flipped_discriminative_powers = flipped_beacon_lrts.mean(dim=0) - flipped_control_lrts.mean(dim=0)
        delta_discriminative_powers = discriminative_powers - flipped_discriminative_powers
        sorted_gene_indices = torch.argsort(delta_discriminative_powers, descending=False)
        K = int(self.args.gene_size * k)
        return sorted_gene_indices[:K]

    def _init_baseline_beacon(self, k=10):
        un_mafs = torch.unique(torch.as_tensor(self.mafs))
        # Optional sizing knob (--baseline_target_snps N). Default 0 keeps the
        # lowest 10% of distinct MAF values, skipping the monomorphic MAF==0 bin,
        # which on CEU/100k covers 13,048 SNPs (13.05%).
        target = int(getattr(self.args, 'baseline_target_snps', 0) or 0)
        if target <= 0:
            return un_mafs[1:int(k / 100 * un_mafs.numel())]

        mafs_t = torch.as_tensor(self.mafs)
        chosen, total = [], 0
        for v in un_mafs[1:]:                    # skip the MAF==0 bin, as above
            cnt = int((mafs_t == v).sum())
            if total + cnt > target:
                break
            chosen.append(v)
            total += cnt
        return torch.stack(chosen) if chosen else un_mafs[1:2]
    
    def _init_OG_beacon(self):
        beacon_size = self.args.beacon_size
        control_size = self.args.b_control_size
        x_beacon = torch.any(self.beacon_case, 0).to(torch.int)
        self.eta_in_beac = (self.beacon_case * self.A(self.mafs, beacon_size) * x_beacon) + (self.B(self.mafs, beacon_size) * (1 - x_beacon) * self.beacon_case)
        self.eta_not_in_beac = (self.beacon_control * self.A(self.mafs, control_size) * x_beacon) + (self.B(self.mafs, control_size) * (1 - x_beacon) * self.beacon_control)
        self.Delt_in_Beac = (self.beacon_case * self.B(self.mafs, beacon_size)) - (self.beacon_case * self.A(self.mafs, beacon_size) * x_beacon)
        self.Delt_Not_in_Beac = (self.beacon_control * self.B(self.mafs, control_size)) - (self.beacon_control * self.A(self.mafs, control_size) * x_beacon)
        self.x_beacon = x_beacon
        self.delt_b = torch.zeros(beacon_size)
        self.eta_b = torch.zeros(beacon_size)
        self.delt_nb = torch.zeros(control_size)
        self.eta_nb = torch.zeros(control_size)
        k_low_delt_nb, _ = torch.topk(self.delt_nb, k=self.K, largest=False)
        delt_n_mean = torch.mean(k_low_delt_nb)
        self.positives = torch.sum((self.Delt_in_Beac - delt_n_mean) >= 0, dim=0) 
        self.selected_snps = torch.where((x_beacon == 1) & (self.positives == beacon_size))[0]

    def D(self, f, n):
        return torch.clip(torch.pow((1 - f), (2 * n)), 5e-307, 0.999999)

    def A(self, f, n):
        delta = self.error
        return torch.log(1 - self.D(f, n)) - torch.log(1 - (delta * self.D(f, n - 1)))

    def B(self, f, n):
        delta = self.error
        return torch.log(self.D(f, n)) - torch.log(delta * self.D(f, n - 1))
        
    def _calc_group_lrts_all_snps(self, genome, maf, response) -> torch.Tensor:
        error = self.error
        one_minus_maf = (1 - maf)
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)
        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - torch.log(DN_i * (1 - error * DN_i_1))
        lrt = (log1 + log2 * response).unsqueeze(0) * genome 
        return lrt

    def _calc_group_lrts(self, genome, maf, response, prev_beacon_lrts, update_qb=False) -> torch.Tensor:
        error = self.error
        one_minus_maf = (1 - maf[-1])
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)
        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - torch.log(DN_i * (1 - error * DN_i_1))
        lrt = (log1 + log2 * response[-1]).mul(genome[:, -1])
        updated_beacon_lrts = prev_beacon_lrts + lrt
        return updated_beacon_lrts

    def _update_budget(self, attacker_action, maf, agent_id):
        budgets = self.user_budgets[agent_id]
        people_with_snp = self.beacon_case[:, attacker_action] > 0
        valid_individuals = people_with_snp & (budgets > 0)
        if torch.sum(valid_individuals) > 0:
            budget_costs = self._calculate_budget_cost(maf)
            budgets[valid_individuals] -= budget_costs
        self.user_budgets[agent_id] = budgets

    def _calc_pvalues(self):
        beacon_lrts = copy.deepcopy(self.beacon_lrts)
        control_lrts = copy.deepcopy(self.control_lrts)
        pvalues=[]
        for blrt in beacon_lrts:
            pvalue=torch.sum(blrt >= control_lrts) / self.args.b_control_size
            pvalues.append(pvalue)
        if torch.any(torch.Tensor(pvalues) < 0): assert "Wrong p value"
        return torch.Tensor(pvalues)

    def _calc_percentage_of_snps_at_position(self, people, position):
        if position < 0 or position >= people.size(1): raise ValueError("Index n is out of bounds.")
        count_ones = torch.sum(people[:, position])
        total_people = people.size(0)         
        percentage = count_ones / total_people
        return percentage 

    def _calculate_budget_cost(self, maf_value):
        D_qi = (1 - maf_value).pow(self.args.beacon_size - 1)
        return -torch.log(1 - D_qi)

    def _real_time_flipping(self, attacker_action):
        # (This remains unchanged from your original logic)
        maf_value = self.mafs[attacker_action]
        if maf_value >= self.rare_threshold: return 1
        has_variant = (self.beacon_case[:, attacker_action] == 1)
        if not torch.any(has_variant): return 0
        carriers = torch.where(has_variant)[0]
        b_lrts_after, c_lrts_after=self.update(1, attacker_action, calculation_mode=True)
        pvalues_after = calculate_pvalues(b_lrts_after, c_lrts_after, self.args.b_control_size)
        vulnerable_carriers = []
        for idx in carriers:
            current_p = pvalues_after[idx].item()
            old_p = self.pvalues[idx].item()
            diff_p = abs(current_p - old_p)
            if diff_p < 0.001: self.no_change_count[idx] += 1
            else: self.no_change_count[idx] = 0
            if (current_p < 0.05) and (self.no_change_count[idx] < 50):
                vulnerable_carriers.append(idx)
        if len(vulnerable_carriers) == 0: return 1
        min_pvalue = min([pvalues_after[i].item() for i in vulnerable_carriers])
        flip_prob = round((1 - min_pvalue), 1)
        if random.random() < flip_prob: return 0
        else: return 1
