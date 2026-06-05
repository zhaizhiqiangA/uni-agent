"""vLLM KV-cache event decoding and optional ZMQ subscription helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

from uni_agent.llm_router.kv_cache import KVCacheIndex


def decode_vllm_kv_event_payload(payload: bytes) -> Any:
    """Decode a vLLM KV event payload.

    JSON is supported without extra dependencies. If vLLM publishes msgpack
    payloads, msgspec is imported lazily so the router package stays lightweight.
    """
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            import msgspec
        except ImportError as exc:
            raise RuntimeError("msgpack KV event payloads require the optional msgspec package") from exc
        return msgspec.msgpack.decode(payload)


def iter_kv_events(decoded_payload: Any) -> Iterable[Any]:
    if isinstance(decoded_payload, dict) and "events" in decoded_payload:
        default_replica_id = decoded_payload.get("replica_id") or decoded_payload.get("server_id")
        for event in decoded_payload["events"]:
            if default_replica_id is not None and isinstance(event, dict) and "replica_id" not in event:
                event = {**event, "replica_id": default_replica_id}
            yield event
    elif isinstance(decoded_payload, list):
        yield from decoded_payload
    else:
        yield decoded_payload


class VLLMKVEventSubscriber:
    """Subscribe to vLLM KV-cache events and update a local KVCacheIndex."""

    def __init__(self, endpoint: str, replica_id: str, cache_index: KVCacheIndex, topic: bytes = b"") -> None:
        self.endpoint = endpoint
        self.replica_id = replica_id
        self.cache_index = cache_index
        self.topic = topic
        self._closed = asyncio.Event()

    def consume_payload(self, payload: bytes) -> None:
        decoded_payload = decode_vllm_kv_event_payload(payload)
        for event in iter_kv_events(decoded_payload):
            self.cache_index.apply_event(event, default_replica_id=self.replica_id)

    async def run_forever(self) -> None:
        try:
            import zmq
            import zmq.asyncio
        except ImportError as exc:
            raise RuntimeError("VLLMKVEventSubscriber requires pyzmq") from exc

        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        socket.connect(self.endpoint)
        try:
            while not self._closed.is_set():
                message = await socket.recv_multipart()
                payload = message[-1]
                self.consume_payload(payload)
        finally:
            socket.close(linger=0)

    def close(self) -> None:
        self._closed.set()
