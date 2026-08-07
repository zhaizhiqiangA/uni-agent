"""Parallel gold-patch verification for the blackbox-recipe SWE-bench dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid

import ray
from datasets import load_dataset

from examples.blackbox_recipes.mini_swe_agent.mini_swe_agent_runner import (
    SandboxEnvForReward,
    extract_image,
)
from examples.blackbox_recipes.mini_swe_agent.reward import (
    build_reward_context,
    evaluate_in_env,
)
from examples.blackbox_recipes.sandbox_client import SandboxClient
from uni_agent.reward import load_reward_spec

LOGGER = logging.getLogger("parallel_verify_golden_patch")
DEFAULT_DATA_PATH = "swe_bench_verified.parquet"
DEFAULT_TOOL_IMAGE = os.environ.get(
    "SWE_AGENT_TOOL_IMAGE",
    "swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest",
)


def instance_id(sample: dict) -> str:
    return str(sample["extra_info"]["tools_kwargs"]["reward"]["metadata"]["instance_id"])


def disable_install_commands() -> None:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    for repo_specs in MAP_REPO_VERSION_TO_SPECS.values():
        for specs in repo_specs.values():
            specs.pop("install", None)


async def run_sample(
    sample: dict,
    *,
    eval_timeout: int,
    skip_install: bool,
    use_image_initial_workspace: bool,
    upstream: str,
) -> dict:
    run_id = str(uuid.uuid4())
    tools_kwargs = sample["extra_info"]["tools_kwargs"]
    env_config = tools_kwargs.get("env", {})
    reward_config = tools_kwargs.get("reward", {})
    iid = instance_id(sample)
    sandbox = None
    started = time.perf_counter()
    result = {
        "instance_id": iid,
        "run_id": run_id,
        "resolved": False,
        "eval_completed": False,
        "eval_valid": False,
        "score_valid": False,
        "classification": "infra_invalid",
        "image_mode": "tag",
        "install_mode": "skipped" if skip_install else "official",
    }
    try:
        image = extract_image(env_config)
        if not image:
            raise RuntimeError("missing sandbox image")
        sandbox = await SandboxClient.create(
            image=image,
            sidecar_image=DEFAULT_TOOL_IMAGE,
            upstream=upstream,
        )
        result["sandbox_id"] = sandbox.sandbox_id

        post_setup = env_config.get("post_setup_cmd", "")
        if post_setup:
            setup = await sandbox.run(post_setup, timeout=600)
            if setup.exit_code != 0:
                raise RuntimeError(f"post_setup_cmd failed (rc={setup.exit_code})")

        ground_truth = reward_config.get("metadata", {})
        base_commit = ground_truth.get("base_commit")
        if not base_commit:
            raise RuntimeError("missing base_commit")
        initial_state = await sandbox.run(
            "cd /testbed && git rev-parse HEAD && git status --porcelain && "
            f"(git diff {json.dumps(str(base_commit))}..HEAD -- tox.ini || true) && "
            "(grep -n pytest tox.ini || true)",
            timeout=120,
        )
        result["initial_workspace_state"] = initial_state.stdout
        if use_image_initial_workspace:
            verify = await sandbox.run("cd /testbed && git rev-parse --verify HEAD", timeout=120)
        else:
            verify = await sandbox.run(
                "cd /testbed && actual=$(git rev-parse HEAD) && "
                f'test "$actual" = {json.dumps(str(base_commit))} && '
                'test -z "$(git status --porcelain)"',
                timeout=120,
            )
        if verify.exit_code != 0:
            raise RuntimeError("initial base/workspace verification failed")

        reward_env = SandboxEnvForReward(sandbox)
        reward_spec = load_reward_spec(
            {
                "name": reward_config["name"],
                "run_id": run_id,
                "metadata": ground_truth,
                "env": reward_env,
                "eval_timeout": eval_timeout,
            }
        )
        await reward_spec.apply_gold_patch()

        metadata, context_timeout = build_reward_context(tools_kwargs)
        score, eval_result = await evaluate_in_env(
            reward_env,
            metadata,
            context_timeout or eval_timeout,
        )
        result.update(eval_result or {})
        result["reward_score"] = score
        result["resolved"] = bool(result.get("resolved"))
        result["eval_execution_time"] = time.perf_counter() - started
        result["eval_valid"] = bool(result.get("eval_completed") and result.get("eval_report"))
        result["score_valid"] = result["eval_valid"]
        if result["score_valid"]:
            result["classification"] = "resolved" if result["resolved"] else "unresolved"
        else:
            result["classification"] = "protocol_invalid"
    except Exception as exc:
        result["eval_error"] = f"{type(exc).__name__}: {exc}"
        result["classification"] = (
            "patch_apply_failure" if "patch" in str(exc).lower() else "infra_invalid"
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox.cleanup()
                result["cleanup_status"] = "completed"
            except Exception as exc:
                result["cleanup_status"] = "error"
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                result["score_valid"] = False
                result["classification"] = "infra_invalid"
    return result


@ray.remote
class VerifyActor:
    def __init__(
        self,
        concurrency: int,
        eval_timeout: int,
        skip_install: bool,
        use_image_initial_workspace: bool,
        upstream: str,
    ):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.eval_timeout = eval_timeout
        self.skip_install = skip_install
        self.use_image_initial_workspace = use_image_initial_workspace
        self.upstream = upstream
        if skip_install:
            disable_install_commands()

    async def run_batch(self, samples: list[dict]) -> list[dict]:
        async def guarded(sample: dict) -> dict:
            async with self.semaphore:
                return await run_sample(
                    sample,
                    eval_timeout=self.eval_timeout,
                    skip_install=self.skip_install,
                    use_image_initial_workspace=self.use_image_initial_workspace,
                    upstream=self.upstream,
                )

        return await asyncio.gather(*(guarded(sample) for sample in samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", DEFAULT_DATA_PATH))
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("MAX_SAMPLES", "500")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("MAX_CONCURRENCY", "2")))
    parser.add_argument("--eval-timeout", type=int, default=int(os.environ.get("SWE_AGENT_EVAL_TIMEOUT", "600")))
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--use-image-initial-workspace", action="store_true")
    parser.add_argument("--upstream", default=os.environ.get("OPENYUANRONG_UPSTREAM", ""))
    parser.add_argument("--output", default=os.environ.get("RESULT_PATH", "dn-record/0727/golden-results.jsonl"))
    args = parser.parse_args()
    if args.max_samples <= 0 or args.workers <= 0 or args.concurrency <= 0:
        raise SystemExit("max-samples, workers and concurrency must be positive")

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    if args.max_samples > len(dataset):
        raise SystemExit(f"requested {args.max_samples}, dataset has {len(dataset)}")
    samples = dataset.select(range(args.max_samples)).to_list()

    ray.init()
    actors = [
        VerifyActor.remote(
            args.concurrency,
            args.eval_timeout,
            args.skip_install,
            args.use_image_initial_workspace,
            args.upstream,
        )
        for _ in range(args.workers)
    ]
    chunk_size = (len(samples) + len(actors) - 1) // len(actors)
    futures = [
        actor.run_batch.remote(samples[i * chunk_size : (i + 1) * chunk_size])
        for i, actor in enumerate(actors)
    ]
    results = [item for chunk in ray.get(futures) for item in chunk]

    sample_ids = [instance_id(sample) for sample in samples]
    result_ids = [str(item.get("instance_id")) for item in results]
    if len(results) != len(samples) or sorted(sample_ids) != sorted(result_ids):
        raise RuntimeError("sample/result IDs are inconsistent")

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for item in results:
        key = str(item.get("classification"))
        counts[key] = counts.get(key, 0) + 1
    LOGGER.info("gold verification complete: samples=%d classifications=%s", len(results), counts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
