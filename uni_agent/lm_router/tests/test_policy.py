import pytest

from uni_agent.lm_router import KVCacheAwareLoadBalancer, KVCacheAwarePolicyConfig, NoAvailableReplicaError


def make_balancer() -> KVCacheAwareLoadBalancer:
    balancer = KVCacheAwareLoadBalancer(KVCacheAwarePolicyConfig(gpu_memory_utilization_threshold=0.8))
    balancer.add_replica("r0", handle="handle-r0", gpu_memory_utilization=0.2, inflight_requests=3)
    balancer.add_replica("r1", handle="handle-r1", gpu_memory_utilization=0.2, inflight_requests=0)
    return balancer


def test_fast_path_keeps_session_primary_when_memory_is_below_threshold() -> None:
    balancer = make_balancer()
    balancer.cache_index.add_blocks("r1", ["a", "b"])
    balancer.sessions.set("session-1", "r0")

    decision = balancer.select_server("session-1", ["a", "b"])

    assert decision.replica_id == "r0"
    assert decision.reason == "session_primary"


def test_slow_path_uses_best_prefix_hit_when_primary_is_over_threshold() -> None:
    balancer = make_balancer()
    balancer.sessions.set("session-1", "r0")
    balancer.update_replica_load("r0", gpu_memory_utilization=0.95)
    balancer.cache_index.add_blocks("r0", ["a", "b", "c"])
    balancer.cache_index.add_blocks("r1", ["a", "b"])

    decision = balancer.select_server("session-1", ["a", "b", "c"])

    assert decision.replica_id == "r1"
    assert decision.reason == "prefix_hit"
    assert decision.prefix_hits == 2
    assert decision.prefix_hit_rate == pytest.approx(2 / 3)


def test_slow_path_chooses_least_loaded_when_no_prefix_hits() -> None:
    balancer = make_balancer()

    decision = balancer.select_server("new-session", ["x"])

    assert decision.replica_id == "r1"
    assert decision.reason == "least_loaded"


def test_acquire_sets_primary_and_increments_inflight() -> None:
    balancer = make_balancer()

    replica_id, handle = balancer.acquire_server("session-1", ["x"])

    assert replica_id == "r1"
    assert handle == "handle-r1"
    assert balancer.sessions.snapshot()["session-1"] == "r1"
    assert balancer.get_state()["replicas"]["r1"]["inflight_requests"] == 1

    balancer.release_server("r1")

    assert balancer.get_state()["replicas"]["r1"]["inflight_requests"] == 0


def test_no_available_replica_when_all_are_over_threshold() -> None:
    balancer = make_balancer()
    balancer.update_replica_load("r0", gpu_memory_utilization=0.9)
    balancer.update_replica_load("r1", gpu_memory_utilization=0.9)

    with pytest.raises(NoAvailableReplicaError):
        balancer.select_server("session-1", ["a"])
