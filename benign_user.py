"""
Benign User Class for Mixed-Population Collusion Defense Testing

A benign user represents a legitimate researcher who:
- Queries the beacon for valid research purposes
- Does NOT target any specific individual
- Queries SNPs based on research interest (random or MAF-based)
- Should NOT be penalized by the defense mechanism

This allows testing the defense in realistic scenarios where
n total users query the beacon, but only k of them are colluding attackers.
"""

import numpy as np
import torch
import random
from helpers.maf_tools import MAFHelper


class BenignUser:
    """
    Simulates a benign (non-attacking) user querying the beacon.

    Unlike attackers who target a specific victim, benign users:
    - Query SNPs for legitimate research
    - Don't have a specific victim
    - Query patterns are random or research-motivated
    """

    def __init__(self, args, maf_helper: MAFHelper, query_mode="random"):
        """
        Initialize a benign user.

        Args:
            args: Configuration arguments
            maf_helper: MAF helper for SNP information
            query_mode: How benign user selects queries
                - "random": Uniform random SNP selection
                - "common": Prefers common SNPs (high MAF) - typical research
                - "rare": Prefers rare SNPs (low MAF) - rare variant research
                - "mixed": Mix of common and rare (realistic research)
        """
        self.args = args
        self.maf_helper = maf_helper
        self.mafs = maf_helper.get_mafs()
        self.query_mode = query_mode

        # User identifier (set by env.py)
        self.user_id = None
        self.is_attacker = False  # Flag to identify user type

        # Query tracking
        self.queries_made = []
        self.query_count = 0

        # For state compatibility with attacker (minimal state)
        self.beacon_actions = torch.tensor([], dtype=torch.float32)
        self.attacker_actions = torch.tensor([], dtype=torch.long)

        # Benign users don't track victim LRT (they don't have a victim)
        # But we need these for interface compatibility
        self.victim_lrt = torch.as_tensor(0)
        self.control_lrts = torch.zeros(size=(self.args.a_control_size,))
        self.identified = False  # Benign users don't "identify" anyone
        self.identify_events = 0
        self.last_identify_event = False

        # Pre-compute query distribution based on mode
        self._init_query_distribution()

    def _init_query_distribution(self):
        """Initialize query probability distribution based on mode."""
        num_snps = len(self.mafs)

        if self.query_mode == "random":
            # Uniform random
            self.query_probs = np.ones(num_snps) / num_snps

        elif self.query_mode == "common":
            # Prefer common SNPs (high MAF)
            # Weight by MAF (higher MAF = more likely to query)
            weights = np.array(self.mafs) + 0.01  # Avoid zero
            self.query_probs = weights / weights.sum()

        elif self.query_mode == "rare":
            # Prefer rare SNPs (low MAF)
            # Weight by inverse MAF
            weights = 1.0 / (np.array(self.mafs) + 0.01)
            self.query_probs = weights / weights.sum()

        elif self.query_mode == "mixed":
            # 70% common, 30% rare (realistic research)
            maf_array = np.array(self.mafs)
            common_mask = maf_array > 0.1
            rare_mask = maf_array <= 0.1

            weights = np.zeros(num_snps)
            weights[common_mask] = 0.7 / common_mask.sum() if common_mask.sum() > 0 else 0
            weights[rare_mask] = 0.3 / rare_mask.sum() if rare_mask.sum() > 0 else 0

            # Normalize
            if weights.sum() > 0:
                self.query_probs = weights / weights.sum()
            else:
                self.query_probs = np.ones(num_snps) / num_snps
        else:
            # Default to random
            self.query_probs = np.ones(num_snps) / num_snps

    def act(self, step, agent_action=None):
        """
        Select a SNP to query (benign behavior).

        Args:
            step: Current query step
            agent_action: Ignored (benign users don't use RL agents)

        Returns:
            snp_position: Position of SNP to query
        """
        # Select SNP based on query mode distribution
        snp_position = np.random.choice(
            len(self.mafs),
            p=self.query_probs
        )

        self.queries_made.append(snp_position)
        self.query_count += 1

        return snp_position

    def update(self, beacon_action, attacker_action):
        """
        Update internal state after receiving beacon response.

        Benign users don't track LRT against any victim,
        but we update for interface compatibility.
        """
        beacon_action_tensor = torch.as_tensor(beacon_action, dtype=torch.float32)
        attacker_action_tensor = torch.as_tensor([attacker_action], dtype=torch.long)

        self.beacon_actions = torch.cat((self.beacon_actions, beacon_action_tensor.unsqueeze(0)))
        self.attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))

    def get_state(self):
        """
        Get state for RL agent (if needed).

        Benign users typically don't use RL, but this provides
        interface compatibility.
        """
        # Return minimal state
        return torch.zeros(10)  # Placeholder

    def reset(self):
        """Reset user state for new episode."""
        self.queries_made = []
        self.query_count = 0
        self.beacon_actions = torch.tensor([], dtype=torch.float32)
        self.attacker_actions = torch.tensor([], dtype=torch.long)
        self.victim_lrt = torch.as_tensor(0)
        self.control_lrts = torch.zeros(size=(self.args.a_control_size,))
        self.identified = False
        self.identify_events = 0
        self.last_identify_event = False


class BenignUserPool:
    """
    Manages a pool of benign users with diverse query behaviors.

    Creates benign users with different research motivations:
    - Some study common variants
    - Some study rare variants
    - Some have random interests
    """

    def __init__(self, args, maf_helper: MAFHelper, num_benign: int):
        """
        Create a pool of benign users.

        Args:
            args: Configuration arguments
            maf_helper: MAF helper for SNP information
            num_benign: Number of benign users to create
        """
        self.users = []

        # Distribute query modes among benign users
        modes = ["random", "common", "rare", "mixed"]

        for i in range(num_benign):
            # Cycle through modes for diversity
            mode = modes[i % len(modes)]
            user = BenignUser(args, maf_helper, query_mode=mode)
            user.user_id = i  # Will be overwritten by env.py
            self.users.append(user)

    def get_users(self):
        """Return list of benign users."""
        return self.users

    def reset_all(self):
        """Reset all benign users for new episode."""
        for user in self.users:
            user.reset()
