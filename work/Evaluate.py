"""
Evaluation utilities for the trained KUKA PPO agent.

This module provides functions to:
    - run a fixed number of evaluation episodes and compute summary statistics
    - render rollout frames for visual inspection
    - produce a per-target-color breakdown of success rate
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports for optional heavy dependencies
try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_policy(
    model,
    env,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    render_path: Optional[str] = None,
    verbose: int = 0,
) -> Dict:
    """
    Roll out the trained policy for *n_episodes* and collect metrics.

    Parameters
    ----------
    model
        A Stable-Baselines3 policy (or any object with a ``predict`` method
        that accepts an observation and returns ``(action, state)``).
    env
        The Gymnasium environment.  Must expose ``info["is_success"]``,
        ``info["distance_to_target"]``, and ``info["target_idx"]`` in its
        step return.
    n_episodes : int
        Number of complete episodes to run.
    deterministic : bool
        Whether to use the deterministic policy (no action-space noise).
    render : bool
        Whether to capture RGB frames for GIF export.  Requires the env to
        have been constructed with ``render_mode="rgb_array"``.
    render_path : str, optional
        File path (including ``.gif`` extension) where the rendered episode
        is saved.  Only the *first* episode is rendered to keep file size
        manageable.
    verbose : int
        Print progress every ``verbose`` episodes (0 = silent).

    Returns
    -------
    dict
        Keys:
            - ``mean_reward``       : float, mean undiscounted episode reward
            - ``std_reward``        : float, std dev of episode rewards
            - ``success_rate``      : float, fraction of successful episodes
            - ``mean_distance``     : float, mean final distance (all episodes)
            - ``mean_distance_success``: float, mean final distance for
                                         successful episodes only
            - ``mean_ep_length``    : float, mean episode length
            - ``per_color_success`` : dict mapping color name to success rate
            - ``all_rewards``       : list of per-episode rewards
            - ``all_successes``     : list of per-episode success flags (0/1)
            - ``all_distances``     : list of per-episode final distances
    """
    COLOR_NAMES = {0: "red", 1: "green", 2: "blue"}

    rewards: List[float] = []
    successes: List[int] = []
    distances: List[float] = []
    ep_lengths: List[int] = []
    target_idxs: List[int] = []

    frames_for_gif: List[np.ndarray] = []

    obs, _ = env.reset()
    ep_idx = 0
    current_reward = 0.0
    current_len = 0
    current_success = False
    current_distance = 0.0
    current_target = -1

    while ep_idx < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        current_reward += reward
        current_len += 1
        current_success = info.get("is_success", False)
        current_distance = info.get("distance_to_target", 0.0)
        current_target = info.get("target_idx", -1)

        if render and ep_idx == 0 and _HAS_IMAGEIO:
            frame = env.render()
            if frame is not None:
                frames_for_gif.append(frame)

        if terminated or truncated:
            rewards.append(current_reward)
            successes.append(int(current_success))
            distances.append(current_distance)
            ep_lengths.append(current_len)
            target_idxs.append(current_target)

            if verbose > 0 and (ep_idx + 1) % verbose == 0:
                print(
                    f"  Eval ep {ep_idx+1:4d}/{n_episodes} | "
                    f"reward={current_reward:7.2f} | "
                    f"success={current_success} | "
                    f"dist={current_distance:.3f} | "
                    f"color={COLOR_NAMES.get(current_target, '?')}"
                )

            ep_idx += 1
            current_reward = 0.0
            current_len = 0
            current_success = False
            obs, _ = env.reset()

    # -- Export GIF if requested -----------------------------------------
    if render and frames_for_gif and _HAS_IMAGEIO and render_path:
        os.makedirs(os.path.dirname(render_path) or ".", exist_ok=True)
        imageio.mimsave(render_path, frames_for_gif, fps=15)
        if verbose > 0:
            print(f"  GIF saved to {render_path}")

    # -- Aggregate statistics --------------------------------------------
    rewards_arr = np.array(rewards, dtype=float)
    successes_arr = np.array(successes, dtype=float)
    distances_arr = np.array(distances, dtype=float)
    target_idxs_arr = np.array(target_idxs)

    # Per-color success rate
    per_color_success: Dict[str, float] = {}
    for idx, name in COLOR_NAMES.items():
        mask = target_idxs_arr == idx
        if mask.sum() > 0:
            per_color_success[name] = float(successes_arr[mask].mean())
        else:
            per_color_success[name] = float("nan")

    successful_distances = distances_arr[successes_arr == 1]
    mean_dist_success = (
        float(successful_distances.mean()) if successful_distances.size > 0 else float("nan")
    )

    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "success_rate": float(successes_arr.mean()),
        "mean_distance": float(distances_arr.mean()),
        "mean_distance_success": mean_dist_success,
        "mean_ep_length": float(np.mean(ep_lengths)),
        "per_color_success": per_color_success,
        "all_rewards": rewards,
        "all_successes": successes,
        "all_distances": distances,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_eval_summary(results: Dict, label: str = "Evaluation") -> None:
    """
    Print a formatted summary of the dictionary returned by
    :func:`evaluate_policy`.

    Parameters
    ----------
    results : dict
        Output of :func:`evaluate_policy`.
    label : str
        Heading string printed at the top of the summary.
    """
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Mean reward        : {results['mean_reward']:>8.3f}  ± {results['std_reward']:.3f}")
    print(f"  Success rate       : {results['success_rate']:>8.1%}")
    print(f"  Mean final distance: {results['mean_distance']:>8.4f} m")
    print(f"  Mean dist (success): {results['mean_distance_success']:>8.4f} m")
    print(f"  Mean episode length: {results['mean_ep_length']:>8.1f} steps")
    print(f"\n  Per-color success rate:")
    for color, rate in results["per_color_success"].items():
        bar = "#" * int(rate * 20)
        print(f"    {color:>5} : {rate:.1%}  |{bar:<20}|")
    print(f"{'='*55}\n")