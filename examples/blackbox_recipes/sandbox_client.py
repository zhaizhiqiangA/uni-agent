"""AKernel remote sandbox command execution.

AKernel is an agent sandbox infra collaboratively developed by the
OpenYuanrong team and the Ant AKernel team.
Uses ``akernel_sdk.Sandbox`` with sidecar ``Mount`` to inject the
mini-swe-agent tool image.  Supports upstream tunnel so the agent
inside the sandbox can reach the gateway via ``http://127.0.0.1:<proxy_port>``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass
class CommandResult:
    """Result of a command executed inside a sandbox."""

    stdout: str
    stderr: str
    exit_code: int


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

DEFAULT_PROXY_PORT = 38197

# ── Patch akernel_sdk RPC timeout ───────────────────────────────────────
# The SDK hardcodes YR_GET_DEFAULT_TIMEOUT = 300 in types.py.
# When the proxy (netentsec) kills the CONNECT tunnel after 30s idle,
# every subsequent yr.get() call waits the full 300s before giving up.
#
# We patch it down to a reasonable value (60s by default, configurable
# via AKERNEL_RPC_TIMEOUT env var).  Both types.py and commands.py hold
# their own references (from … import …), so both must be patched.
_RPC_TIMEOUT = int(os.getenv("AKERNEL_RPC_TIMEOUT", "60"))


def _patch_yr_timeout() -> None:
    import akernel_sdk.types
    import akernel_sdk.commands

    old = akernel_sdk.types.YR_GET_DEFAULT_TIMEOUT
    akernel_sdk.types.YR_GET_DEFAULT_TIMEOUT = _RPC_TIMEOUT
    akernel_sdk.commands.YR_GET_DEFAULT_TIMEOUT = _RPC_TIMEOUT
    logger.info(
        "Patched YR_GET_DEFAULT_TIMEOUT: %ds → %ds (env AKERNEL_RPC_TIMEOUT=%s)",
        old,
        _RPC_TIMEOUT,
        os.getenv("AKERNEL_RPC_TIMEOUT", "default"),
    )

    # Also teach _is_fatal_poll_error about our repair: after
    # yr.finalize() the in-flight yr.get() calls fail with
    # "does not exist in storeMap" / "already finalized".  Without this,
    # _poll_pid_until_done treats them as transient and keeps polling
    # (spamming memory_store.cpp errors every second) until its deadline.
    _orig_is_fatal = akernel_sdk.commands._is_fatal_poll_error

    def _is_fatal_poll_error(exc: Exception) -> bool:
        s = str(exc)
        if "does not exist in storeMap" in s or "already finalized" in s:
            return True
        return _orig_is_fatal(exc)

    akernel_sdk.commands._is_fatal_poll_error = _is_fatal_poll_error
    logger.info("Patched _is_fatal_poll_error: storeMap/finalized → fatal (stop polling)")


# Safe to call unconditionally — the modules are already importable.
_patch_yr_timeout()


# ── Connection-death detection & yr rebuild ─────────────────────────────
# The netentsec proxy drops the CONNECT tunnel after ~30s idle.  When that
# happens the shared yr RPC connection (gw_client) is dead and every
# subsequent operation waits the RPC timeout before failing.  Detection
# markers below are the SDK's failure signatures.

def is_dead_connection(result) -> bool:
    """Return True if a ``commands.run`` result indicates the yr connection
    (or the remote instance) is irrecoverably gone."""
    if getattr(result, "exit_code", 0) == 0:
        return False
    stderr = getattr(result, "stderr", "") or ""
    markers = (
        "timed out after",          # our asyncio.wait_for wrapper
        "object timeout",           # yr: Get object timeout (code 4005)
        "stream truncated",         # yr: HTTP stream truncated
        "failed to get object",     # yr: Get object timeout
        "already finalized",        # yr: runtime finalized mid-call
    )
    return any(m in stderr for m in markers)


def repair_yr_connection(reason: str = "") -> None:
    """Rebuild the yr RPC connection.

    yr runtime is process-global and initialized once by akernel_sdk
    (``ensure_yr_init``).  When the shared connection dies (proxy idle
    timeout), every subsequent sandbox operation times out.  This function
    tears the runtime down and resets akernel_sdk's init flag so the next
    ``Sandbox()`` construction re-initializes with a fresh connection.

    Must be called *between* sessions — any live sandbox is invalidated.
    """
    import yr
    import akernel_sdk.sandbox as sb_mod

    logger.warning("[sandbox_repair] rebuilding yr connection (reason: %s)", reason or "connection dead")
    try:
        yr.finalize()
        logger.info("[sandbox_repair] yr.finalize() OK — old connection destroyed")
    except Exception as e:
        logger.warning("[sandbox_repair] yr.finalize() error: %s", e)
    try:
        sb_mod._yr_initialized = False
        logger.info("[sandbox_repair] akernel_sdk._yr_initialized reset → next sandbox create will re-init")
    except Exception as e:
        logger.warning("[sandbox_repair] failed to reset _yr_initialized: %s", e)


def _configure_akernel_env() -> None:
    """Validate AKernel credentials and map the tunnel SSL flag for akernel_sdk.

    ``akernel_sdk`` reads ``AKERNEL_SERVER_ADDRESS`` / ``AKERNEL_TOKEN`` directly,
    so only the tunnel SSL flag needs to be translated to ``TUNNEL_SSL_VERIFY``.
    """
    server = os.getenv("AKERNEL_SERVER_ADDRESS")
    token = os.getenv("AKERNEL_TOKEN")
    if not server or not token:
        raise ValueError("AKERNEL_SERVER_ADDRESS and AKERNEL_TOKEN environment variables must be set for sandbox")
    os.environ["TUNNEL_SSL_VERIFY"] = os.getenv("AKERNEL_TUNNEL_SSL_VERIFY", "0")


def _resolve_sandbox_name() -> str | None:
    """Return ``{prefix}{random}`` when ``SANDBOX_NAME_PREFIX`` env is set."""
    prefix = os.getenv("SANDBOX_NAME_PREFIX")
    if not prefix:
        return None
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def extract_upstream(gateway_url: str) -> str:
    """Extract host:port from a gateway URL for upstream tunnel config.

    Example: "http://8.92.9.155:40169/sessions/abc/v1" -> "8.92.9.155:40169"
    """
    parsed = urlparse(gateway_url)
    return f"{parsed.hostname}:{parsed.port}"


def rewrite_gateway_url(
    gateway_url: str,
    proxy_port: int = DEFAULT_PROXY_PORT,
    *,
    strip_v1: bool = False,
) -> str:
    """Rewrite gateway URL to use the sandbox-internal tunnel.

    Replaces host:port with 127.0.0.1:<proxy_port>, keeps path intact.

    Example:
        "http://8.92.9.155:40169/sessions/abc/v1"
        -> "http://127.0.0.1:8766/sessions/abc/v1"
    """
    parsed = urlparse(gateway_url)
    path = parsed.path.removesuffix("/v1") if strip_v1 else parsed.path
    return f"http://127.0.0.1:{proxy_port}{path}"


class SandboxClient:
    """Command execution via remote sandbox."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    @property
    def sandbox_id(self) -> str:
        return getattr(self._sandbox, "sandbox_id", "unknown")

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        sidecar_image: str,
        upstream: str = "",
        proxy_port: int = DEFAULT_PROXY_PORT,
        env: dict[str, str] | None = None,
        cpu: int = 2000,
        memory: int = 4096,
        cpu_limit: int = 8000,
        mem_limit: int = 12288,
        idle_timeout: int = 7200,
        sidecar_target: str = "/opt/mini-swe-agent",
        max_retries: int = 10,
        **sandbox_kwargs: Any,
    ) -> SandboxClient:
        """Create an sandbox client with sidecar tool mounted.

        The sidecar image is mounted at ``sidecar_target`` inside the
        sandbox via ``akernel_sdk.Mount``.

        If ``upstream`` is provided, a tunnel is set up so the sandbox can
        reach the local gateway via ``http://127.0.0.1:<proxy_port>``.
        """
        _configure_akernel_env()
        from akernel_sdk import Mount, Sandbox

        sb_kwargs: dict[str, Any] = {
            "image": image,
            "cpu": cpu,
            "memory": memory,
            "cpu_limit": cpu_limit,
            "mem_limit": mem_limit,
            "idle_timeout": idle_timeout,
            "mounts": [
                Mount(target=sidecar_target, image_url=sidecar_image),
            ],
        }
        if upstream:
            sb_kwargs["upstream"] = upstream
            sb_kwargs["proxy_port"] = proxy_port
        if env:
            sb_kwargs["env"] = env
        name = _resolve_sandbox_name()
        if name is not None:
            sb_kwargs["name"] = name
        sb_kwargs.update(sandbox_kwargs)

        logger.info(
            "Creating sandbox (image=%s, cpu=%d, memory=%d, sidecar=%s:%s, upstream=%s, name=%s)",
            image,
            cpu,
            memory,
            sidecar_image,
            sidecar_target,
            upstream or "none",
            name or "auto",
        )
        last_error: Exception | None = None
        for retry in range(max_retries):
            sandbox = None
            try:
                sandbox = await asyncio.to_thread(lambda: Sandbox(**sb_kwargs))
                logger.info("sandbox created: %s", getattr(sandbox, "sandbox_id", "?"))
                return cls(sandbox=sandbox)
            except Exception as exc:
                last_error = exc
                sandbox_id = getattr(sandbox, "sandbox_id", None)
                logger.critical(
                    "Failed to create sandbox (sandbox_id=%s): %s",
                    sandbox_id or "n/a",
                    exc,
                )
                if sandbox is not None:
                    try:
                        await asyncio.to_thread(sandbox.kill)
                    except Exception:
                        pass
                if retry < max_retries - 1:
                    sleep_time = min(30, 2**retry)
                    logger.info("Retrying sandbox creation in %d seconds...", sleep_time)
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to create sandbox after {max_retries} retries") from last_error

    async def run(self, cmd: str, *, timeout: int = 600) -> CommandResult:
        """Execute *cmd* inside the sandbox via ``sandbox.commands.run``.

        Wrapped with ``asyncio.wait_for`` so that a dead yr RPC connection
        (e.g. proxy-killed CONNECT tunnel) fails in *timeout* + 30 s instead
        of the SDK-default 300 s.
        """
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._sandbox.commands.run,
                    cmd,
                    timeout=timeout,
                ),
                timeout=timeout + 30,
            )
            return CommandResult(
                stdout=getattr(result, "stdout", ""),
                stderr=getattr(result, "stderr", ""),
                exit_code=getattr(result, "exit_code", -1),
            )
        except asyncio.TimeoutError:
            return CommandResult(
                stdout="",
                stderr=f"sandbox run timed out after {timeout + 30}s",
                exit_code=-1,
            )
        except Exception as e:
            return CommandResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Kill the sandbox if still running.

        Both ``is_running()`` and ``kill()`` are wrapped with short timeouts
        so a dead yr RPC connection doesn't block for 300 s.
        """
        if self._sandbox is not None:
            sandbox_id = getattr(self._sandbox, "sandbox_id", "?")
            try:
                is_alive = False
                try:
                    is_alive = await asyncio.wait_for(
                        asyncio.to_thread(self._sandbox.is_running),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    logger.warning("is_running() timed out for sandbox %s — assuming dead", sandbox_id)
                except Exception:
                    pass

                if is_alive:
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(self._sandbox.kill),
                            timeout=15,
                        )
                        logger.info("sandbox %s killed", sandbox_id)
                    except asyncio.TimeoutError:
                        logger.warning("kill() timed out for sandbox %s", sandbox_id)
                else:
                    logger.info("sandbox %s already stopped", sandbox_id)
            except Exception as e:
                logger.warning("Failed to kill sandbox %s: %s", sandbox_id, e)
            self._sandbox = None
