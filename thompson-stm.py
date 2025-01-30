import numpy as np
import logging
from collections import deque
from functools import lru_cache
from typing import Tuple, Dict, Any
import warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ContextualThompsonSampler:
    def __init__(self, min_bid: int = 80, max_bid: int = 100, tau: float = 2.0, history_window: int = 10):
        """
        Contextual Thompson Sampling with Softmax Scaling and Short-Term Memory
        
        Args:
            min_bid: Minimum bid value
            max_bid: Maximum bid value
            tau: Temperature parameter for softmax scaling (higher = more exploration)
            history_window: Maximum number of historical data points to maintain per context
        """
        self.validate_init_params(min_bid, max_bid, tau, history_window)
        self.min_bid = min_bid
        self.max_bid = max_bid
        self.tau = tau
        self.history_window = history_window
        self.history: Dict[Tuple[Any, Any], deque] = {}
        self.beta_stats: Dict[Tuple[Any, Any], Dict[int, Tuple[int, int]]] = {}
        
    @staticmethod
    def validate_init_params(min_bid: int, max_bid: int, tau: float, history_window: int) -> None:
        if min_bid >= max_bid:
            raise ValueError(f"min_bid ({min_bid}) must be less than max_bid ({max_bid})")
        if tau <= 0:
            raise ValueError(f"tau ({tau}) must be positive")
        if history_window < 1:
            raise ValueError(f"history_window ({history_window}) must be at least 1")

    def stable_softmax(self, values: np.ndarray) -> np.ndarray:
        values = values / self.tau
        values = values - np.max(values)
        exp_values = np.exp(values)
        return exp_values / np.sum(exp_values)

    def sample_bid(self, campaign_id: Any, revenue_bucket: Any) -> int:
        key = (campaign_id, revenue_bucket)

        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_window)
            self.beta_stats[key] = {bid: [1, 1] for bid in range(self.min_bid, self.max_bid + 1)}
        
        bid_samples = {bid: np.random.beta(a, b) for bid, (a, b) in self.beta_stats[key].items()}
        bid_values = np.array(list(bid_samples.values()))
        bid_keys = np.array(list(bid_samples.keys()))
        bid_probs = self.stable_softmax(bid_values)
        chosen_bid = np.random.choice(bid_keys, p=bid_probs)
        
        return int(chosen_bid)

    def update(self, campaign_id: Any, revenue_bucket: Any, bid: int, win: int) -> None:
        key = (campaign_id, revenue_bucket)
        
        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_window)
            self.beta_stats[key] = {bid: [1, 1] for bid in range(self.min_bid, self.max_bid + 1)}
        
        self.history[key].append((bid, win))
        self._recompute_beta_from_deque(key)
        
    def _recompute_beta_from_deque(self, key: Tuple[Any, Any]) -> None:
        bid_counts = {bid: [1, 1] for bid in range(self.min_bid, self.max_bid + 1)}
        
        for bid, win in self.history[key]:
            bid_counts[bid][0] += win  # Increment wins (alpha)
            bid_counts[bid][1] += 1 - win  # Increment losses (beta)
        
        self.beta_stats[key] = bid_counts

    def get_bid_distribution(self, campaign_id: Any, revenue_bucket: Any) -> Dict[int, list]:
        key = (campaign_id, revenue_bucket)
        if key not in self.history or len(self.history[key]) == 0:
            return {}
        
        return self.beta_stats[key]
