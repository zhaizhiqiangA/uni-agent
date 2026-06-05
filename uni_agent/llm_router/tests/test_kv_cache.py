from uni_agent.llm_router import KVCacheIndex
from uni_agent.llm_router.vllm_events import iter_kv_events


def test_kv_cache_index_applies_store_remove_and_clear_events() -> None:
    index = KVCacheIndex()

    index.apply_event({"event_type": "BlockStored", "replica_id": "r0", "block_hashes": ["a", "b"]})
    index.apply_event({"event_type": "BlockStored", "replica_id": "r1", "block_hashes": ["a"]})

    assert index.contains("r0", "a")
    assert index.prefix_hits("r0", ["a", "b", "c"]) == 2
    assert index.replicas_for_block("a") == frozenset({"r0", "r1"})

    index.apply_event({"event_type": "BlockRemoved", "replica_id": "r0", "block_hashes": ["a"]})

    assert not index.contains("r0", "a")
    assert index.replicas_for_block("a") == frozenset({"r1"})

    index.apply_event({"event_type": "AllBlocksCleared", "replica_id": "r1"})

    assert index.cached_blocks("r1") == frozenset()
    assert index.replicas_for_block("a") == frozenset()


def test_prefix_hits_are_contiguous() -> None:
    index = KVCacheIndex()
    index.add_blocks("r0", ["a", "c"])

    assert index.prefix_hits("r0", ["a", "b", "c"]) == 1


def test_non_gpu_events_do_not_affect_index() -> None:
    index = KVCacheIndex()

    index.apply_event({"event_type": "BlockStored", "replica_id": "r0", "block_hashes": ["a"], "tier": "cpu"})

    assert not index.contains("r0", "a")


def test_iter_kv_events_adds_batch_replica_id() -> None:
    events = list(
        iter_kv_events(
            {
                "replica_id": "r0",
                "events": [{"event_type": "BlockStored", "block_hashes": ["a"]}],
            }
        )
    )

    assert events == [{"event_type": "BlockStored", "block_hashes": ["a"], "replica_id": "r0"}]
