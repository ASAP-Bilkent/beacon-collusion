"""Physical-account routing using observed bits only; no defender access."""
import math
from collections import deque


class AccountRouter:
    def __init__(self, accounts, mode="round_robin", window=20):
        if accounts < 1 or window < 1 or mode not in ("round_robin", "history"):
            raise ValueError("Invalid account router configuration")
        self.mode = mode
        self.counts = [0] * accounts
        self.recent = [deque(maxlen=window) for _ in range(accounts)]

    def choose(self):
        total = sum(self.counts)
        if self.mode == "round_robin":
            return total % len(self.counts)
        for account, count in enumerate(self.counts):
            if count == 0:
                return account
        # A heuristic for observed YES yield, not an estimate of honesty or
        # membership information. Exploration prevents permanent starvation.
        def value(i):
            return (sum(self.recent[i]) / len(self.recent[i])
                    + math.sqrt(2 * math.log(total + 1) / self.counts[i]))
        return max(range(len(self.counts)), key=value)

    def observe(self, account, response):
        if response not in (0, 1):
            raise ValueError("Router accepts only observed binary responses")
        self.counts[account] += 1
        self.recent[account].append(int(response))
