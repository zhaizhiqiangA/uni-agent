"""Uniagent runner for the blackbox SWE-agent recipe.

Uses white-box interaction components (AgentInteraction, OpenAICompatibleChatModel,
ToolsManager) with gateway-based LLM routing. Computes reward in-process and
passes it via the gateway's complete_session endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from uni_agent.trainer.framework.types import SessionHandle, SessionRuntime
from uni_agent.interaction.env import AgentEnv, AgentEnvConfig
from uni_agent.interaction.interaction import AgentInteraction
from uni_agent.interaction.model import OpenAICompatibleChatModel
from uni_agent.interaction.tools_manager import ToolsManager, ToolsManagerConfig
from uni_agent.tools import ToolConfig

from examples.swe_agent_blackbox.reward import build_reward_context, evaluate_in_env

logger = logging.getLogger(__name__)


# =====================================================================
# Config helpers (uniagent-specific)
# =====================================================================


def load_agent_config(path: str) -> dict[str, Any]:
    """Load agent config from a YAML file. Returns the first entry."""
    import yaml

    with open(os.path.expanduser(path)) as f:
        configs = yaml.safe_load(f)
    if isinstance(configs, list):
        return configs[0] if configs else {}
    return configs or {}


def _create_agent_env(run_id: str, tools_kwargs: dict, agent_config: dict) -> AgentEnv:
    """Create AgentEnv from agent_config + per-sample tools_kwargs overrides."""
    env_config = dict(agent_config.get("env", {}))
    env_override = dict(tools_kwargs.get("env", {}))
    # Patch deployment image/command from per-sample override
    if env_override:
        deployment = dict(env_config.get("deployment", {}))
        deployment.update({k: env_override.pop(k) for k in ["image", "command"] if k in env_override})
        env_config["deployment"] = deployment
        env_config.update(env_override)
    return AgentEnv(run_id=run_id, env_config=AgentEnvConfig(**env_config))


# =====================================================================
# Agent runner
# =====================================================================


async def swe_agent_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    session_runtime: SessionRuntime,
    tools_kwargs: dict | None = None,
    agent_config_path: str | None = None,
) -> None:
    """Run the uniagent SWE-agent through the gateway with in-process reward."""
    tools_kwargs = tools_kwargs or {}
    agent_config = load_agent_config(agent_config_path) if agent_config_path else {}
    interaction_cfg = agent_config.get("interaction", {})

    messages = (
        list(raw_prompt) if isinstance(raw_prompt, list)
        else [{"role": "user", "content": str(raw_prompt)}]
    )

    env = _create_agent_env(f"swe_bb_{sample_index}", tools_kwargs, agent_config)
    metadata, eval_timeout = build_reward_context(tools_kwargs)

    try:
        await env.start()

        model = OpenAICompatibleChatModel(
            base_url=session.base_url,
            api_key="not-needed",
            model_name="default",
        )

        tools_config = agent_config.get("tools", [])
        tools_manager = ToolsManager(
            tools_manager_config=ToolsManagerConfig(
                tools=[ToolConfig(name=t["name"]) for t in tools_config],
                parser=agent_config.get("tool_parser", "qwen3_coder"),
            ),
        )
        model.set_tools_schemas(tools_manager.tools_schemas)

        interaction = AgentInteraction(
            run_id=f"swe_bb_{sample_index}",
            env=env,
            model=model,
            tools_manager=tools_manager,
            messages=messages,
            action_timeout=interaction_cfg.get("action_timeout", 300),
            timeout_budget=interaction_cfg.get("timeout_budget", -1),
            max_turns=interaction_cfg.get("max_turns", 100),
        )

        result = await interaction.run()
        trajectory = result.get("trajectory", [])
        logger.info("interaction finished, %d steps", len(trajectory))

        # Evaluate reward in the same Docker env
        score, eval_result = await evaluate_in_env(env, metadata, eval_timeout)
        logger.info("reward: score=%s, resolved=%s", score, eval_result.get("resolved"))

        # Signal completion with reward_info
        reward_info = {"reward_score": score, **eval_result}
        await session_runtime.complete_session(session.session_id, reward_info=reward_info)

    except Exception as e:
        logger.warning("Agent runner failed for sample %d: %s", sample_index, e)
        try:
            await session_runtime.complete_session(session.session_id, reward_info={"reward_score": 0.0})
        except Exception:
            pass
        raise
    finally:
        try:
            await env.close()
        except Exception:
            pass
