"""KV-cache-aware routing primitives for agentic RL rollout servers."""

from uni_agent.lm_router.kv_cache import KVCacheEvent, KVCacheIndex, normalize_block_hash
from uni_agent.lm_router.load_balancer import KVCacheAwareLoadBalancer
from uni_agent.lm_router.policy import (
    KVCacheAwarePolicy,
    KVCacheAwarePolicyConfig,
    NoAvailableReplicaError,
    ReplicaState,
    RouteDecision,
    RouteRequest,
)

__all__ = [
    "KVCacheAwareLoadBalancer",
    "KVCacheAwarePolicy",
    "KVCacheAwarePolicyConfig",
    "KVCacheEvent",
    "KVCacheIndex",
    "NoAvailableReplicaError",
    "ReplicaState",
    "RouteDecision",
    "RouteRequest",
    "normalize_block_hash",
]
