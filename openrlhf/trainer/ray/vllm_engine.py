from itertools import pairwise
import os
import time
import queue
from collections import defaultdict
from typing import Any, List

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.inputs import TokensPrompt

from openrlhf.utils.logging_utils import init_logger

from .utils import ray_noset_visible_devices

from openrlhf.rl_envs import SweBenchEnv, DummyEnv

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

        import vllm

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.llm = vllm.LLM(*args, **kwargs)

        self.rollouts = None

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def reset_rollout_cache(self, world_size: int) -> None:
        self.env_data_for_rollout = [None] * world_size
        self.rollouts = None

    def remember_env_data_for_rollout(self, rank: int, data_for_rank: list[dict]) -> None:
        assert hasattr(self, "env_data_for_rollout"), (
            "You must call LLMRayActor.reset_rollout_cache before calling LLMRayActor.remember_env_data_for_rollout"
        )

        self.env_data_for_rollout[rank] = data_for_rank

    def generate_env_rollout(self, rank: int, sampling_params, env_maker) -> list:
        if self.rollouts is not None:
            return self.rollouts[rank]

        assert hasattr(self, "data_remembered_for_rollout"), (
            "You must call LLMRayActor.reset_rollout_cache before calling LLMRayActor.generate_env_rollout"
        )
        assert all(data_for_rank is not None for data_for_rank in self.env_data_for_rollout), (
            "You must call LLMRayActor.remember_env_data_for_rollout for each rank before calling LLMRayActor.generate_env_rollout"
        )

        env = env_maker(
            full_data=sum(self.env_data_for_rollout, []),
            sampling_params=sampling_params,
            vllm_engine=self.llm,
            mongo_uri=self.mongo_uri,
            mongo_db_name=self.mongo_db_name,
            mongo_collection_name=self.mongo_collection_name,
        )

        rollouts = env.generate_many()

        self.rollouts = [
            rollouts[i:j]
            for i, j in pairwise(cumulative_sum(len(data_for_rank) for data_for_rank in self.env_data_for_rollout))
        ]

        self.env_data_for_rollout = [None] * len(self.env_data_for_rollout)

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
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 8} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size,
        )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        vllm_engines.append(
            LLMRayActor.options(
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
            )
        )

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
