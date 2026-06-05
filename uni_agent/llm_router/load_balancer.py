"""A KV-cache-aware load balancer for vLLM rollout replicas."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from uni_agent.llm_router.kv_cache import KVCacheIndex
from uni_agent.llm_router.policy import (
    KVCacheAwarePolicy,
    KVCacheAwarePolicyConfig,
    ReplicaState,
    RouteDecision,
    RouteRequest,
    SessionTable,
)


class KVCacheAwareLoadBalancer:
    """Stateful router that can replace verl's least-inflight-only balancer."""

    def __init__(self, config: KVCacheAwarePolicyConfig | None = None) -> None:
        self.config = config or KVCacheAwarePolicyConfig()
        self.policy = KVCacheAwarePolicy(self.config)
        self.cache_index = KVCacheIndex()
        self.sessions = SessionTable(max_sessions=self.config.max_sessions)
        self._replicas: dict[str, ReplicaState] = {}
        self._handles: dict[str, Any] = {}

    def add_replica(
        self,
        replica_id: str,
        handle: Any = None,
        *,
        gpu_memory_utilization: float = 0.0,
        inflight_requests: int = 0,
        active: bool = True,
    ) -> None:
        self._replicas[replica_id] = ReplicaState(
            replica_id=replica_id,
            gpu_memory_utilization=gpu_memory_utilization,
            inflight_requests=inflight_requests,
            active=active,
        )
        self._handles[replica_id] = handle
        self.cache_index.register_replica(replica_id)

    def remove_replica(self, replica_id: str) -> None:
        self._replicas.pop(replica_id, None)
        self._handles.pop(replica_id, None)
        self.cache_index.remove_replica(replica_id)
        self.sessions.remove_replica(replica_id)

    def update_replica_load(
        self,
        replica_id: str,
        *,
        gpu_memory_utilization: float | None = None,
        inflight_requests: int | None = None,
        active: bool | None = None,
    ) -> None:
        replica = self._replicas[replica_id]
        if gpu_memory_utilization is not None:
            replica.gpu_memory_utilization = gpu_memory_utilization
        if inflight_requests is not None:
            replica.inflight_requests = inflight_requests
        if active is not None:
            replica.active = active

    def apply_kv_event(self, raw_event: Any, default_replica_id: str | None = None) -> None:
        event = self.cache_index.apply_event(raw_event, default_replica_id=default_replica_id)
        self.cache_index.register_replica(event.replica_id)

    def select_server(self, session_id: str, prefix_block_hashes: Iterable[Any] = ()) -> RouteDecision:
        request = RouteRequest.from_values(session_id=session_id, prefix_block_hashes=prefix_block_hashes)
        return self.policy.select(request, self._replicas, self.sessions, self.cache_index)

    def acquire_server(self, session_id: str, prefix_block_hashes: Iterable[Any] = ()) -> tuple[str, Any]:
        decision = self.select_server(session_id=session_id, prefix_block_hashes=prefix_block_hashes)
        self.sessions.set(session_id, decision.replica_id)
        self._replicas[decision.replica_id].inflight_requests += 1
        return decision.replica_id, self._handles.get(decision.replica_id)

    def release_server(self, replica_id: str) -> None:
        replica = self._replicas.get(replica_id)
        if replica is None or replica.inflight_requests <= 0:
            return
        replica.inflight_requests -= 1

    def get_state(self) -> dict[str, Any]:
        return {
            "replicas": {
                replica_id: {
                    "gpu_memory_utilization": replica.gpu_memory_utilization,
                    "inflight_requests": replica.inflight_requests,
                    "active": replica.active,
                    "cached_blocks": sorted(self.cache_index.cached_blocks(replica_id)),
                }
                for replica_id, replica in self._replicas.items()
            },
            "sessions": self.sessions.snapshot(),
        }
