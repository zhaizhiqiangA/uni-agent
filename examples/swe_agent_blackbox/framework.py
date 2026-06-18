"""SWE-agent framework compatibility alias."""

from __future__ import annotations

from uni_agent.framework.framework import OpenAICompatibleAgentFramework


class SWEAgentFramework(OpenAICompatibleAgentFramework):
    """Recipe FQN kept for existing configs; behavior lives in the main framework."""

