"""Routing policy for KV-cache-aware rollout replica selection."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from uni_agent.llm_router.kv_cache import KVCacheIndex, normalize_block_hash


class NoAvailableReplicaError(RuntimeError):
    """Raised when no active replica satisfies the memory threshold."""


@dataclass
class KVCacheAwarePolicyConfig:
    gpu_memory_utilization_threshold: float = 0.9
    max_sessions: int = 10000


@dataclass
class ReplicaState:
    replica_id: str
    gpu_memory_utilization: float = 0.0
    inflight_requests: int = 0
    active: bool = True

    def accepts_new_requests(self, threshold: float) -> bool:
        return self.active and self.gpu_memory_utilization <= threshold


@dataclass(frozen=True)
class RouteRequest:
    session_id: str
    prefix_block_hashes: tuple[str, ...] = ()

    @classmethod
    def from_values(cls, session_id: str, prefix_block_hashes: Iterable[Any] = ()) -> RouteRequest:
        return cls(
            session_id=session_id,
            prefix_block_hashes=tuple(normalize_block_hash(hash_value) for hash_value in prefix_block_hashes),
        )


@dataclass(frozen=True)
class RouteDecision:
    replica_id: str
    reason: str
    prefix_hits: int
    prefix_hit_rate: float
    inflight_requests: int
    gpu_memory_utilization: float


class SessionTable:
    """Small LRU map from agent session ID to its primary replica."""

    def __init__(self, max_sessions: int) -> None:
        self.max_sessions = max_sessions
        self._session_to_replica: OrderedDict[str, str] = OrderedDict()

    def get(self, session_id: str) -> str | None:
        replica_id = self._session_to_replica.get(session_id)
        if replica_id is not None:
            self._session_to_replica.move_to_end(session_id)
        return replica_id

    def set(self, session_id: str, replica_id: str) -> None:
        self._session_to_replica[session_id] = replica_id
        self._session_to_replica.move_to_end(session_id)
        while len(self._session_to_replica) > self.max_sessions:
            self._session_to_replica.popitem(last=False)

    def remove_replica(self, replica_id: str) -> None:
        for session_id, mapped_replica_id in list(self._session_to_replica.items()):
            if mapped_replica_id == replica_id:
                self._session_to_replica.pop(session_id, None)

    def snapshot(self) -> dict[str, str]:
        return dict(self._session_to_replica)


class KVCacheAwarePolicy:
    """Fast sticky-session path plus slow prefix-hit-aware load balancing."""

    def __init__(self, config: KVCacheAwarePolicyConfig | None = None) -> None:
        self.config = config or KVCacheAwarePolicyConfig()

    def select(
        self,
        request: RouteRequest,
        replicas: Mapping[str, ReplicaState],
        sessions: SessionTable,
        cache_index: KVCacheIndex,
    ) -> RouteDecision:
        primary_replica_id = sessions.get(request.session_id)
        if primary_replica_id is not None:
            primary = replicas.get(primary_replica_id)
            if primary is not None and primary.accepts_new_requests(self.config.gpu_memory_utilization_threshold):
                return self._decision(
                    primary,
                    cache_index=cache_index,
                    prefix_block_hashes=request.prefix_block_hashes,
                    reason="session_primary",
                )

        candidates = [
            replica
            for replica in replicas.values()
            if replica.accepts_new_requests(self.config.gpu_memory_utilization_threshold)
        ]
        if not candidates:
            raise NoAvailableReplicaError("no active replica is below the GPU memory utilization threshold")

        prefix_scores = []
        for replica in candidates:
            hits = cache_index.prefix_hits(replica.replica_id, request.prefix_block_hashes)
            prefix_scores.append((hits, replica))
        best_prefix_hits = max((hits for hits, _ in prefix_scores), default=0)
        if best_prefix_hits > 0:
            best_candidates = [replica for hits, replica in prefix_scores if hits == best_prefix_hits]
            selected = min(best_candidates, key=self._load_key)
            return self._decision(
                selected,
                cache_index=cache_index,
                prefix_block_hashes=request.prefix_block_hashes,
                reason="prefix_hit",
            )

        selected = min(candidates, key=self._load_key)
        return self._decision(
            selected,
            cache_index=cache_index,
            prefix_block_hashes=request.prefix_block_hashes,
            reason="least_loaded",
        )

    @staticmethod
    def _load_key(replica: ReplicaState) -> tuple[int, float, str]:
        return (replica.inflight_requests, replica.gpu_memory_utilization, replica.replica_id)

    @staticmethod
    def _decision(
        replica: ReplicaState,
        *,
        cache_index: KVCacheIndex,
        prefix_block_hashes: tuple[str, ...],
        reason: str,
    ) -> RouteDecision:
        prefix_hits = cache_index.prefix_hits(replica.replica_id, prefix_block_hashes)
        prefix_hit_rate = prefix_hits / len(prefix_block_hashes) if prefix_block_hashes else 0.0
        return RouteDecision(
            replica_id=replica.replica_id,
            reason=reason,
            prefix_hits=prefix_hits,
            prefix_hit_rate=prefix_hit_rate,
            inflight_requests=replica.inflight_requests,
            gpu_memory_utilization=replica.gpu_memory_utilization,
        )
