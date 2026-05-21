"""Reward utilities for the blackbox SWE-agent recipe.

Contains:
- compute_score: thin reward function that reads reward_score from extra_info
- evaluate_in_env: run reward evaluation in Docker env (shared by both runners)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """Read reward_score from extra_info, injected by SWEAgentFramework."""
    if extra_info and "reward_score" in extra_info:
        return float(extra_info["reward_score"])
    return 0.0


def _get_reward_spec(data_source: str):
    """Load reward spec class by data_source name."""
    from uni_agent.reward.registry import REWARD_SPEC_REGISTRY, _load_reward_spec_module

    if data_source not in REWARD_SPEC_REGISTRY:
        _load_reward_spec_module(data_source)
    cls = REWARD_SPEC_REGISTRY.get(data_source)
    if cls is None:
        raise ValueError(f"Unknown data_source: {data_source}. Available: {list(REWARD_SPEC_REGISTRY.keys())}")
    return cls


async def evaluate_in_env(
    env,
    metadata: dict[str, Any],
    eval_timeout: int = 600,
) -> tuple[float, dict]:
    """Run reward evaluation in the Docker env.

    Returns (score, eval_result) where score is 1.0/0.0 and
    eval_result contains details (eval_completed, resolved, etc.).
    """
    data_source = metadata.get("data_source", "unknown")
    reward_model = metadata.get("reward_model", {})

    spec_cls = _get_reward_spec(data_source)
    spec_metadata = reward_model.get("ground_truth", reward_model)

    spec = spec_cls(
        run_id="swe_bb_eval",
        metadata=spec_metadata,
        env=env,
        eval_timeout=eval_timeout,
    )

    resolved, result = await spec.compute_reward()
    score = 1.0 if resolved else 0.0
    return score, result
