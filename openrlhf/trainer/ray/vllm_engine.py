from itertools import pairwise
import asyncio
import json
import os
from time import perf_counter
import queue
from collections import defaultdict
from typing import Any, List
from pymongo import MongoClient
from datetime import datetime


import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.inputs import TokensPrompt

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.interface import AsyncVLLM

from .utils import get_bundle_indices, ray_noset_visible_devices
from openrlhf.utils import print_gpu_memory_usage

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


def cumulative_sum(xs):
    sum = 0
    yield sum
    for x in xs:
        sum += x
        yield sum


@ray.remote
class LLMRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        noset_visible_devices = kwargs.pop("noset_visible_devices")
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        if kwargs.get("truncate_prompt_tokens") is not None:
            self.truncate_prompt_tokens = kwargs.get("truncate_prompt_tokens")
            del kwargs["truncate_prompt_tokens"]
        else:
            self.truncate_prompt_tokens = None
            del kwargs["truncate_prompt_tokens"]

        self.compact_filtering = kwargs.pop("compact_filtering", False)
        self.filter_max_steps = kwargs.pop("filter_max_steps", False)
        
        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        self.responses = {}
        self.full_data = {}

        # Extract MongoDB configuration
        self.mongo_uri = kwargs.pop("mongo_uri", None)
        self.mongo_db_name = kwargs.pop("mongo_db_name", None)
        self.mongo_collection_name = kwargs.pop("mongo_collection_name", None)
        
        # Extract transcripts folder
        self.transcripts_folder = kwargs.pop("transcripts_folder", None)

        import vllm

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # self.llm = vllm.LLM(*args, **kwargs)
        self.async_event_loop = asyncio.new_event_loop()
        self.llm_engine = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(*args, **kwargs, disable_log_requests=True)
        )

        self.rollouts = None

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.async_event_loop.run_until_complete(
            self.llm_engine.collective_rpc(
                "init_process_group",
                args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
            )
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.async_event_loop.run_until_complete(
            self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))
        )

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.async_event_loop.run_until_complete(
            self.llm_engine.collective_rpc(
                "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
            )
        )

    def reset_prefix_cache(self):
        self.async_event_loop.run_until_complete(self.llm_engine.reset_prefix_cache())

    def sleep(self, level=1):
        self.async_event_loop.run_until_complete(self.llm_engine.sleep(level=level))

    def wake_up(self):
        print("LLMRayActor.wake_up called")
        print_gpu_memory_usage()
        self.async_event_loop.run_until_complete(self.llm_engine.wake_up())
        print("LLMRayActor.wake_up finished")
        print_gpu_memory_usage()

    def reset_rollout_cache(self) -> None:
        self.env_data_for_rollout = {}
        self.rollouts = None

    def remember_env_data_for_rollout(self, rank: int, data_for_rank: list[dict]) -> None:
        assert hasattr(self, "env_data_for_rollout"), (
            "You must call LLMRayActor.reset_rollout_cache before calling LLMRayActor.remember_env_data_for_rollout"
        )

        print(f"LLMRayActor.remember_env_data_for_rollout called with {self=} {rank=} {len(data_for_rank)=}")

        self.env_data_for_rollout[rank] = data_for_rank

    def generate_env_rollout(self, rank: int, sampling_params, env_makers, is_eval: bool, vllm_engine_index: int, step: int) -> list:
        print(f"LLMRayActor.generate_env_rollout called with {self=} {rank=} {vllm_engine_index=}")

        if self.rollouts is not None:
            return self.rollouts[rank]

        assert hasattr(self, "env_data_for_rollout"), (
            "You must call LLMRayActor.reset_rollout_cache before calling LLMRayActor.generate_env_rollout"
        )
        assert all(data_for_rank is not None for data_for_rank in self.env_data_for_rollout), (
            "You must call LLMRayActor.remember_env_data_for_rollout for each rank before calling LLMRayActor.generate_env_rollout"
        )

        # TO DO: truncate_prompt_tokens should only be in sampling_params
        # assert self.truncate_prompt_tokens == sampling_params.truncate_prompt_tokens
        if self.truncate_prompt_tokens is not None:
            sampling_params.truncate_prompt_tokens = self.truncate_prompt_tokens

        full_data = sum(self.env_data_for_rollout.values(), [])

        # Separate data by filename
        data_by_env = {}
        for item in full_data:
            env_name = item.get("datasource")
            if env_name not in data_by_env:
                data_by_env[env_name] = []
            data_by_env[env_name].append(item)

        async_llm = AsyncVLLM(llm_engine=self.llm_engine, sampling_params=sampling_params)

        # Create environments and run them simultaneously
        async def run_all_environments():
            tasks = []
            for env_name, data_for_env in data_by_env.items():
                print(f"Creating {env_name} environment with {len(data_for_env)} samples")
                if env_name in env_makers:
                    env = env_makers[env_name](
                        vllm_engine_index=vllm_engine_index,
                        compact_filtering=self.compact_filtering,
                        filter_max_steps=self.filter_max_steps,
                    )
                    task = env.generate_rollouts(
                        llm=async_llm,
                        full_data=data_for_env,
                        env_name=env_name,
                        is_eval=is_eval,
                    )
                    tasks.append((env_name, task))

            # Run all environments simultaneously
            results = await asyncio.gather(*[task for _, task in tasks])

            # Combine results in the same order as the original data
            count_by_env = {}
            env_indices = {name: i for i, (name, _) in enumerate(tasks)}
            all_rollouts = []
            for item in full_data:
                env_name = item.get("datasource")
                if env_name not in count_by_env:
                    count_by_env[env_name] = 0

                env_index = env_indices[env_name]
                env_results = results[env_index]
                output = env_results[count_by_env[env_name]]
                all_rollouts.append(output)

                count_by_env[env_name] += 1

            assert len(all_rollouts) == len(full_data), f"Expected {len(full_data)} rollouts, got {len(all_rollouts)}"
            return all_rollouts

        rollouts = self.async_event_loop.run_until_complete(run_all_environments())
        if self.mongo_uri is not None and self.mongo_db_name is not None and self.mongo_collection_name is not None:
            try:
                mongo_client = MongoClient(self.mongo_uri)
                db = mongo_client[self.mongo_db_name]
                collection = db[self.mongo_collection_name]
            except Exception as e:
                print(f"Error connecting to MongoDB: {e}")
                
            try:
                now = datetime.now()
                messages = [
                    {
                        "conversation": conversation.messages, 
                        "reward": reward, 
                        "timestamp": now,
                        "step": step,
                    } for (conversation, reward) in rollouts
                ]
                print(f"Logging {len(messages)} rollouts to MongoDB")
                collection.insert_many(messages)
            except Exception as e:
                print(f"Error logging rollouts to MongoDB: {e}")
            
        if self.transcripts_folder is not None:
            messages = [{"conversation": conversation.messages, "reward": reward} for (conversation, reward) in rollouts]
            with open(os.path.join(self.transcripts_folder, f"rollouts_rank{rank}_{datetime.now().strftime('%m%dT%H:%M')}.json"), "w") as f:
                json.dump(messages, f)

        self.rollouts = {
            rank: rollouts[i:j]
            for rank, (i, j) in zip(
                self.env_data_for_rollout.keys(),
                pairwise(cumulative_sum(len(data_for_rank) for data_for_rank in self.env_data_for_rollout.values())),
                strict=True,
            )
        }

        self.env_data_for_rollout = {}

        return self.rollouts[rank]

    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        return self.responses.pop(actor_rank)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    mongo_uri=None,
    mongo_db_name=None,
    mongo_collection_name=None,
    transcripts_folder=None,
    rollout_batch_size=None,
    n_samples_per_prompt=None,
    actor_num_nodes=None,
    actor_num_gpus_per_node=None,
    max_cpus=None,
    truncate_prompt_tokens=None,
    compact_filtering=False,
    filter_max_steps=False,
):
    import vllm

    assert vllm.__version__ >= "0.8.1", "OpenRLHF only supports vllm >= 0.8.1"

    vllm_engines = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        if (
            rollout_batch_size is None
            or n_samples_per_prompt is None
            or actor_num_nodes is None
            or actor_num_gpus_per_node is None
            or max_cpus is None
        ):
            cpu_per_actor = 1
        else:
            optimal_cpu_amt = rollout_batch_size * n_samples_per_prompt
            if max_cpus > 0:
                optimal_cpu_amt = min(optimal_cpu_amt, max_cpus)

            cpu_per_actor = optimal_cpu_amt / (actor_num_nodes * actor_num_gpus_per_node)
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": cpu_per_actor} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        engine = LLMRayActor.options(
            num_cpus=num_gpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
        ).remote(
            model=pretrain,
            enforce_eager=enforce_eager,
            worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed + i,
            distributed_executor_backend=distributed_executor_backend,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
            dtype="bfloat16",
            trust_remote_code=True,
            full_determinism=full_determinism,
            num_actors=num_actors,
            gpu_memory_utilization=gpu_memory_utilization,
            bundle_indices=bundle_indices,
            num_gpus=0.2 if use_hybrid_engine else 1,
            enable_sleep_mode=vllm_enable_sleep,
            noset_visible_devices=noset_visible_devices,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
            mongo_collection_name=mongo_collection_name,
            transcripts_folder=transcripts_folder,
            truncate_prompt_tokens=truncate_prompt_tokens,
            compact_filtering=compact_filtering,
            filter_max_steps=filter_max_steps,
        )

        vllm_engines.append(engine)

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
