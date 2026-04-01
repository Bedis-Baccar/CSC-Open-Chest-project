"""
Custom Stable Baselines 3 callbacks for training the KUKA reach task.

This module provides:
    - EpisodeMetricsCallback: records per-episode success rate, mean reward,
      and mean distance-to-target so that training curves can be plotted.
    - CheckpointWithMetricsCallback: saves model weights at regular intervals
      and appends a snapshot of the running metrics to a JSON log file.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class EpisodeMetricsCallback(BaseCallback):
    """
    Records episode-level metrics during PPO training.

    At the end of every episode (across all parallel environments) this
    callback collects:
        - ``episode_reward``    : undiscounted sum of rewards for the episode
        - ``is_success``        : 1 if the final ``info["is_success"]`` is True
        - ``distance_to_target``: Euclidean distance at episode termination

    All metrics are appended to internal lists so they can be retrieved after
    training for plotting.

    Parameters
    ----------
    verbose : int
        Verbosity level (0 = silent, 1 = print episode summaries).
    window : int
        Size of the rolling window used to compute smoothed statistics.
    """

    def __init__(self, verbose: int = 0, window: int = 20) -> None:
        super().__init__(verbose)
        self.window = window

        # Raw per-episode data
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_flags: List[int] = []
        self.final_distances: List[float] = []

        # Timestep at the end of each episode (for x-axis of plots)
        self.timesteps_at_episode_end: List[int] = []

        # Running statistics updated after each episode
        self._ep_reward_buf: List[float] = []
        self._ep_len_buf: List[int] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        """
        Called once per environment step by the SB3 training loop.

        When an episode terminates (``done = True``) we extract the episode
        summary from ``infos[i]`` (the SB3 ``"episode"`` key injected by
        ``Monitor`` wrappers) and any task-specific keys we added ourselves.
        """
        # ``self.locals["dones"]`` is a bool array of length n_envs
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue

            info = self.locals["infos"][i]

            # Episode reward and length come from the Monitor wrapper
            ep_info = info.get("episode", {})
            ep_reward = ep_info.get("r", 0.0)
            ep_len = ep_info.get("l", 0)

            # Task-specific metrics are stored directly in info
            success = int(info.get("is_success", False))
            distance = float(info.get("distance_to_target", -1.0))

            self.episode_rewards.append(float(ep_reward))
            self.episode_lengths.append(int(ep_len))
            self.success_flags.append(success)
            self.final_distances.append(distance)
            self.timesteps_at_episode_end.append(self.num_timesteps)

            if self.verbose >= 1:
                n = len(self.episode_rewards)
                win = self.window
                mean_r = np.mean(self.episode_rewards[-win:])
                mean_s = np.mean(self.success_flags[-win:])
                print(
                    f"[Ep {n:5d}] "
                    f"reward={ep_reward:7.2f}  "
                    f"success={success}  "
                    f"dist={distance:.3f}  "
                    f"mean_r({win})={mean_r:.2f}  "
                    f"sr({win})={mean_s:.1%}  "
                    f"steps={self.num_timesteps}"
                )
        return True

    def _on_training_end(self) -> None:
        """Print a final summary when training completes."""
        if not self.episode_rewards:
            return
        win = self.window
        n_eps = len(self.episode_rewards)
        print(
            f"\n=== Training finished after {self.num_timesteps} steps / "
            f"{n_eps} episodes ==="
        )
        print(
            f"  Last {win}-ep mean reward  : {np.mean(self.episode_rewards[-win:]):.2f}"
        )
        print(
            f"  Last {win}-ep success rate : {np.mean(self.success_flags[-win:]):.1%}"
        )
        print(
            f"  Last {win}-ep mean distance: {np.mean(self.final_distances[-win:]):.4f} m"
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def rolling_mean_reward(self) -> np.ndarray:
        """Rolling mean reward over a window of ``self.window`` episodes."""
        return _rolling_mean(np.array(self.episode_rewards, dtype=float), self.window)

    @property
    def rolling_success_rate(self) -> np.ndarray:
        """Rolling success rate over a window of ``self.window`` episodes."""
        return _rolling_mean(np.array(self.success_flags, dtype=float), self.window)

    @property
    def rolling_mean_distance(self) -> np.ndarray:
        """Rolling mean final distance over a window of ``self.window`` episodes."""
        return _rolling_mean(np.array(self.final_distances, dtype=float), self.window)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise all collected metrics to a plain dictionary."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "success_flags": self.success_flags,
            "final_distances": self.final_distances,
            "timesteps_at_episode_end": self.timesteps_at_episode_end,
        }


class CheckpointWithMetricsCallback(BaseCallback):
    """
    Saves model weights and appends metrics to a JSON log at a fixed interval.

    Parameters
    ----------
    save_freq : int
        Save every ``save_freq`` environment steps.
    save_path : str
        Directory where model zip files are written.
    metrics_callback : EpisodeMetricsCallback
        The companion callback whose metrics are snapshotted at each save.
    name_prefix : str
        Prefix prepended to the saved model filename.
    verbose : int
        Verbosity (0 = silent, 1 = print on save).
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        metrics_callback: EpisodeMetricsCallback,
        name_prefix: str = "ppo_kuka",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.metrics_callback = metrics_callback
        self.name_prefix = name_prefix
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            self._save()
        return True

    def _save(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(
            self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps"
        )
        self.model.save(model_path)

        # Also write a JSON snapshot of metrics so far
        log_path = os.path.join(self.save_path, f"{self.name_prefix}_metrics_log.json")
        snapshot = {
            "num_timesteps": self.num_timesteps,
            "metrics": self.metrics_callback.to_dict(),
        }
        with open(log_path, "w") as fh:
            json.dump(snapshot, fh, indent=2)

        if self.verbose >= 1:
            print(
                f"  [CheckpointCallback] Saved model to {model_path}.zip "
                f"(step {self.num_timesteps})"
            )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a causal (trailing) rolling mean over a 1-D array.

    For each index ``i`` the output is the mean of
    ``arr[max(0, i-window+1) : i+1]``.
    """
    if arr.size == 0:
        return arr
    result = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = arr[start : i + 1].mean()
    return result