"""Multi-armed bandit algorithms for layout exploration."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..layout.spec import LayoutSpec
from ..storage.metadata import MetadataStore


@dataclass
class Arm:
    """Represents a bandit arm (layout candidate)."""

    layout_id: str
    layout_spec: LayoutSpec
    pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0

    def update(self, reward: float) -> None:
        """Update arm statistics with a new reward."""
        self.pulls += 1
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.pulls


class MultiArmedBandit:
    """Multi-armed bandit for layout exploration."""

    def __init__(
        self,
        metadata_store: MetadataStore,
        exploration_constant: float = 2.0,
    ):
        """
        Initialize multi-armed bandit.

        Args:
            metadata_store: Metadata store for retrieving evaluations
            exploration_constant: UCB1 exploration constant (c)
        """
        self.metadata_store = metadata_store
        self.exploration_constant = exploration_constant
        self.arms: Dict[str, Arm] = {}

    def add_arm(self, layout_id: str, layout_spec: LayoutSpec) -> None:
        """Add a new arm (layout) to the bandit."""
        if layout_id not in self.arms:
            self.arms[layout_id] = Arm(
                layout_id=layout_id,
                layout_spec=layout_spec,
            )

    def update_arm(self, layout_id: str, reward: Optional[float]) -> None:
        """Update an arm with observed reward (ignored if reward is None)."""
        if reward is None:
            return
        if layout_id in self.arms:
            self.arms[layout_id].update(float(reward))

    def select_arm_ucb1(self) -> Optional[str]:
        """
        Select an arm using UCB1 algorithm.

        UCB1 formula: argmax(mean_reward + c * sqrt(ln(total_pulls) / arm_pulls))

        Returns:
            Layout ID of selected arm, or None if no arms available
        """
        if not self.arms:
            return None

        total_pulls = sum(arm.pulls for arm in self.arms.values())

        if total_pulls == 0:
            # First pull: select randomly
            return np.random.choice(list(self.arms.keys()))

        # Calculate UCB1 value for each arm
        best_arm_id = None
        best_ucb_value = float("-inf")

        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                # Unpulled arm gets infinite UCB value
                return arm_id

            ucb_value = arm.mean_reward + self.exploration_constant * np.sqrt(
                np.log(total_pulls) / arm.pulls
            )

            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_arm_id = arm_id

        return best_arm_id

    def select_arm_thompson(self) -> Optional[str]:
        """
        Select an arm using Thompson Sampling.

        Assumes rewards are normally distributed.

        Returns:
            Layout ID of selected arm, or None if no arms available
        """
        if not self.arms:
            return None

        # For Thompson Sampling, we need to sample from posterior
        # Simplified version: sample from Beta distribution
        # (assuming binary rewards normalized to [0, 1])

        best_arm_id = None
        best_sample = float("-inf")

        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                # Unpulled arm: use uniform prior
                sample = np.random.beta(1, 1)
            else:
                # Sample from posterior (Beta distribution)
                # Assuming rewards are in [0, 1], we can use successes/failures
                # For simplicity, use mean_reward as success rate
                successes = max(1, int(arm.mean_reward * arm.pulls))
                failures = max(1, arm.pulls - successes)
                sample = np.random.beta(successes + 1, failures + 1)

            if sample > best_sample:
                best_sample = sample
                best_arm_id = arm_id

        return best_arm_id

    def select_arm(self, method: str = "ucb1") -> Optional[str]:
        """
        Select an arm using specified method.

        Args:
            method: "ucb1" or "thompson"

        Returns:
            Layout ID of selected arm
        """
        if method == "ucb1":
            return self.select_arm_ucb1()
        elif method == "thompson":
            return self.select_arm_thompson()
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_best_arm(self) -> Optional[str]:
        """Get the arm with highest mean reward."""
        if not self.arms:
            return None

        best_arm_id = None
        best_mean = float("-inf")

        for arm_id, arm in self.arms.items():
            if arm.pulls > 0 and arm.mean_reward > best_mean:
                best_mean = arm.mean_reward
                best_arm_id = arm_id

        return best_arm_id

    def get_arm_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all arms."""
        return {
            arm_id: {
                "pulls": arm.pulls,
                "mean_reward": arm.mean_reward,
                "total_reward": arm.total_reward,
            }
            for arm_id, arm in self.arms.items()
        }

    def load_evaluations(self, table_name: str) -> None:
        """Load historical evaluations for a table and update arm statistics."""
        import json

        from ..layout.spec import LayoutSpec

        layouts = self.metadata_store.get_all_layouts(table_name)
        stats = self.metadata_store.get_reward_stats_for_table(
            table_name=table_name
        )

        for layout in layouts:
            layout_id = layout["layout_id"]

            # Create layout spec from stored data
            partition_cols = json.loads(layout.get("partition_cols") or "null")
            sort_cols = json.loads(layout.get("sort_cols") or "null")
            layout_spec = LayoutSpec(
                partition_cols=partition_cols,
                sort_cols=sort_cols,
            )

            # Add arm if it doesn't exist
            if layout_id not in self.arms:
                self.add_arm(layout_id, layout_spec)

            s = stats.get(layout_id, {})
            n = int(s.get("n", 0) or 0)
            mean = float(s.get("mean_reward", 0.0) or 0.0)
            self.arms[layout_id].pulls = n
            self.arms[layout_id].mean_reward = mean
            self.arms[layout_id].total_reward = mean * n
