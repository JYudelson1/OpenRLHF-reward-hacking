import os
import time
import torch
import numpy as np
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

from openrlhf.utils.logging_utils import init_logger

from openrlhf.rl_envs import SweBenchEnv, DummyEnv

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayActor:

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating CUDA_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.2"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        self.responses = {}
        self.full_data = {}

        self.llm = LLM(*args, **kwargs)

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

    def add_requests(self, actor_rank, *, sampling_params, prompt_token_ids, multiturn=False, env_maker=None, full_data=None):
        # """
        # Save the requests from actors and generate responses when all actors have sent their requests
        # """
        # start_time = time.time()
        # logger.info(f"Engine {id(self)} received request from actor {actor_rank}, counter={self.actor_counter}/{self.num_actors}")
        
        # self.requests[actor_rank] = prompt_token_ids
        # self.full_data[actor_rank] = full_data
        # self.actor_counter += 1
        # if self.actor_counter == self.num_actors:
        #     start_time = time.time()
        #     logger.info(f"Engine {id(self)} starting generation for all actors at {start_time}")
        #     assert len(self.requests) == self.num_actors
        #     num_requests = []
        #     requests = []
            
        #     if not multiturn:
        #         for actor_rank, request in self.requests.items():
        #             num_requests.append((actor_rank, len(request)))
        #             requests.extend(request)
        #     else:
        #         for actor_rank, data in self.full_data.items():
        #             num_requests.append((actor_rank, len(data)))
        #             requests.extend(data)

        #     if len(requests) > 0:
        #         # For now we assume that all requests have the same sampling params
        #         if not multiturn:
        #             responses = self.llm.generate(sampling_params=sampling_params, prompt_token_ids=requests)
        #         else:
        #             env = env_maker(full_data=requests, sampling_params=sampling_params, vllm_engine=self.llm)
        #             responses = env.generate_many()
        #     else:
        #         responses = []
                
        #     logger.info(f"Engine {id(self)} completed generation in {time.time() - start_time:.2f}s")    

        #     offset = 0
        #     self.responses = {}
        #     for actor_rank, num in num_requests:
        #         self.responses[actor_rank] = responses[offset : offset + num]
        #         offset += num

        #     self.actor_counter = 0
        #     self.requests = {}
        
        """
        Process the request immediately and return the results.
        No actor counting or waiting for multiple actors - just direct processing.
        """
        start_time = time.time()
        logger.info(f"Engine {id(self)} processing request from actor {actor_rank}")

        # Generate responses immediately
        if not multiturn:
            responses = self.llm.generate(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
        else:
            env = env_maker(full_data=full_data, sampling_params=sampling_params, vllm_engine=self.llm)
            responses = env.generate_many()
            
        logger.info(f"Engine {id(self)} completed generation in {time.time() - start_time:.2f}s")
        
        # Return the responses directly
        return responses

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
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
):
    import vllm

    assert vllm.__version__ >= "0.7.0", "OpenRLHF only supports vllm >= 0.7.0"

    vllm_engines = []
    num_gpus = int(tensor_parallel_size == 1)
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    for i in range(num_engines):
        bundle_indices = None
        scheduling_strategy = None

        # Hybrid engine
        if shared_pg is not None:
            assert vllm.__version__ >= "0.7.2", "Only vllm >= 0.7.2 supports hybrid engine"

            if tensor_parallel_size > 1:
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=i * tensor_parallel_size
                )
                bundle_indices = np.arange(i * tensor_parallel_size, (i + 1) * tensor_parallel_size).tolist()
            else:
                num_gpus = 0.2
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=i
                )
        # Distributed RLHF
        elif tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=i
            )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                num_actors=num_actors,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices if shared_pg else None,
                enable_sleep_mode=vllm_enable_sleep,
            )
        )

    return vllm_engines
