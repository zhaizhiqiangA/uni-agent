"""Prefix KV-cache state derived from vLLM KV-cache events."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

GPU_TIER = "gpu"


def normalize_block_hash(value: Any) -> str:
    """Convert vLLM/AIBrix block hash shapes into a stable string key."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if hasattr(value, "__dict__"):
        return json.dumps(vars(value), sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _read_field(raw: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(raw, dict):
        for name in names:
            if name in raw:
                return raw[name]
        return default

    for name in names:
        if hasattr(raw, name):
            return getattr(raw, name)
    return default


def _event_type(raw: Any) -> str:
    value = _read_field(raw, ("event_type", "type", "event", "name"), "")
    if value:
        return str(value)
    return raw.__class__.__name__ if raw is not None else ""


def _extract_hashes(raw: Any) -> tuple[str, ...]:
    fields = (
        "block_hashes",
        "block_hash",
        "hashes",
        "hash",
        "block_ids",
        "block_id",
        "blocks",
        "removed_block_hashes",
        "stored_block_hashes",
    )
    value = _read_field(raw, fields)
    if value is None:
        return ()

    if isinstance(value, (str, bytes, int)):
        return (normalize_block_hash(value),)

    if not isinstance(value, Iterable):
        return (normalize_block_hash(value),)

    hashes = []
    for item in value:
        if isinstance(item, dict):
            nested = _read_field(item, ("block_hash", "hash", "block_id", "id"), item)
            hashes.append(normalize_block_hash(nested))
        else:
            hashes.append(normalize_block_hash(item))
    return tuple(hash_value for hash_value in hashes if hash_value)


@dataclass(frozen=True)
class KVCacheEvent:
    """A normalized GPU KV-cache event for one replica."""

    event_type: str
    replica_id: str
    block_hashes: tuple[str, ...] = ()
    cache_tier: str = GPU_TIER

    @classmethod
    def from_raw(cls, raw: Any, default_replica_id: str | None = None) -> KVCacheEvent:
        replica_id = _read_field(
            raw,
            ("replica_id", "server_id", "instance_id", "worker_id", "source", "event_source"),
            default_replica_id,
        )
        if replica_id is None:
            raise ValueError("KV cache event is missing replica_id")

        cache_tier = str(_read_field(raw, ("cache_tier", "tier", "location"), GPU_TIER)).lower()
        return cls(
            event_type=_event_type(raw),
            replica_id=str(replica_id),
            block_hashes=_extract_hashes(raw),
            cache_tier=cache_tier,
        )

    @property
    def is_gpu_event(self) -> bool:
        return self.cache_tier == GPU_TIER

    @property
    def is_store(self) -> bool:
        normalized = self.event_type.replace("_", "").replace("-", "").lower()
        return any(marker in normalized for marker in ("stored", "added", "inserted", "created"))

    @property
    def is_remove(self) -> bool:
        normalized = self.event_type.replace("_", "").replace("-", "").lower()
        return any(marker in normalized for marker in ("removed", "evicted", "deleted", "freed"))

    @property
    def is_clear(self) -> bool:
        normalized = self.event_type.replace("_", "").replace("-", "").lower()
        return "clear" in normalized or "reset" in normalized


class KVCacheIndex:
    """In-memory map from replica IDs to cached GPU prefix block hashes."""

    def __init__(self) -> None:
        self._blocks_by_replica: dict[str, set[str]] = {}
        self._replicas_by_block: dict[str, set[str]] = {}

    def register_replica(self, replica_id: str) -> None:
        self._blocks_by_replica.setdefault(replica_id, set())

    def remove_replica(self, replica_id: str) -> None:
        blocks = self._blocks_by_replica.pop(replica_id, set())
        for block_hash in blocks:
            replicas = self._replicas_by_block.get(block_hash)
            if replicas is None:
                continue
            replicas.discard(replica_id)
            if not replicas:
                self._replicas_by_block.pop(block_hash, None)

    def add_blocks(self, replica_id: str, block_hashes: Iterable[Any]) -> None:
        blocks = self._blocks_by_replica.setdefault(replica_id, set())
        for raw_hash in block_hashes:
            block_hash = normalize_block_hash(raw_hash)
            if not block_hash:
                continue
            blocks.add(block_hash)
            self._replicas_by_block.setdefault(block_hash, set()).add(replica_id)

    def remove_blocks(self, replica_id: str, block_hashes: Iterable[Any]) -> None:
        blocks = self._blocks_by_replica.setdefault(replica_id, set())
        for raw_hash in block_hashes:
            block_hash = normalize_block_hash(raw_hash)
            blocks.discard(block_hash)
            replicas = self._replicas_by_block.get(block_hash)
            if replicas is None:
                continue
            replicas.discard(replica_id)
            if not replicas:
                self._replicas_by_block.pop(block_hash, None)

    def clear_replica(self, replica_id: str) -> None:
        self.remove_replica(replica_id)
        self.register_replica(replica_id)

    def apply_event(self, raw_event: Any, default_replica_id: str | None = None) -> KVCacheEvent:
        event = (
            raw_event
            if isinstance(raw_event, KVCacheEvent)
            else KVCacheEvent.from_raw(raw_event, default_replica_id)
        )
        if not event.is_gpu_event:
            return event

        if event.is_clear:
            self.clear_replica(event.replica_id)
        elif event.is_store:
            self.add_blocks(event.replica_id, event.block_hashes)
        elif event.is_remove:
            self.remove_blocks(event.replica_id, event.block_hashes)
        return event

    def contains(self, replica_id: str, block_hash: Any) -> bool:
        return normalize_block_hash(block_hash) in self._blocks_by_replica.get(replica_id, set())

    def prefix_hits(self, replica_id: str, prefix_block_hashes: Iterable[Any]) -> int:
        """Return longest contiguous cached prefix length for a replica."""
        cached_blocks = self._blocks_by_replica.get(replica_id, set())
        hits = 0
        for raw_hash in prefix_block_hashes:
            if normalize_block_hash(raw_hash) not in cached_blocks:
                break
            hits += 1
        return hits

    def prefix_hit_rate(self, replica_id: str, prefix_block_hashes: Iterable[Any]) -> float:
        hashes = tuple(prefix_block_hashes)
        if not hashes:
            return 0.0
        return self.prefix_hits(replica_id, hashes) / len(hashes)

    def cached_blocks(self, replica_id: str) -> frozenset[str]:
        return frozenset(self._blocks_by_replica.get(replica_id, set()))

    def replicas_for_block(self, block_hash: Any) -> frozenset[str]:
        return frozenset(self._replicas_by_block.get(normalize_block_hash(block_hash), set()))
