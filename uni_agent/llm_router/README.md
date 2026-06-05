# llm_router

`llm_router` is a KV-cache-aware load balancer for agentic RL rollout traffic.
It is intended to replace the least-inflight-only behavior in verl's
`GlobalRequestLoadBalancer` with routing that understands vLLM prefix cache
state.

## Policy

For each generation request, the caller passes a stable `session_id` and the
request prefix block hashes produced with the same hash scheme as vLLM.

1. Fast path: if `session_id` already has a primary replica and that replica is
   below the GPU memory utilization threshold, keep routing to the primary.
2. Slow path: among replicas below the threshold, choose the highest GPU prefix
   hit rate using the longest contiguous cached prefix.
3. If no eligible replica has a prefix hit, choose the least-loaded eligible
   replica.

vLLM KV-cache events are consumed through `VLLMKVEventSubscriber` and folded into
`KVCacheIndex`. The subscriber supports JSON payloads by default and lazily uses
`msgspec` for msgpack payloads.
