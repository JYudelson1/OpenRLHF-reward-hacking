import itertools
import json
import math
import os
import socket
import time, datetime
import logging
from typing import Callable, Dict, List, Optional, Union, Tuple

import deepspeed
import ray
import torch
import torch.distributed
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, GPTLMLoss
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, masked_mean, unpacking_samples
from openrlhf.trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker, NaiveReplayBuffer, FixedKLController, AdaptiveKLController
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils import blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils import AgentInterface

from .launcher import BasePPORole
from .utils import get_physical_gpu_id


logger = logging.getLogger(__name__)

class ActorPPOTrainer(BasePPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

        self.experience_maker = RemoteExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.strategy.args.colocate_all_models:
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch.distributed.barrier()

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        if args.one_off_pipeline:
            self._fit_one_off(args, prompts_dataloader, pretrain_dataloader, consumed_samples, num_update_steps_per_episodes)
        else:
            self._fit(args, prompts_dataloader, pretrain_dataloader, consumed_samples, num_update_steps_per_episodes)

    def _fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(rand_prompts, **self.generate_kwargs)
                ):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )[0]
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)

                    self.log_rollouts_wandb(experience.json_rollouts, episode=episode, steps=steps, i_experience=i, train_or_eval="train")

                status = self._train(steps, args)
                
                # progress bar
                pbar.set_postfix(status)
                pbar.update()
                
                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, status, client_states)
                
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
            
    def _fit_one_off(self, args, prompts_dataloader, pretrain_dataloader, consumed_samples, num_update_steps_per_episodes):
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)
        
        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for (rollout_num, rand_prompts) in enumerate(self.prompts_dataloader):
                rm_fn = self.experience_maker.reward_fn
                del self.experience_maker.reward_fn
                rollout_ref = make_experience_list_remote.remote(
                    self.experience_maker, rand_prompts, **self.generate_kwargs
                )
                self.experience_maker.reward_fn = rm_fn
                if rollout_num > 0:
                    train_ref = get_train_ref(self, steps, args)
                    rollout_experiences, (status, actor, ema_model) = ray.get([rollout_ref, train_ref])
                    
                    # Do the stateful update
                    if "kl" in status:
                        self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                    self.actor = actor
                    self.ema_model = ema_model
                    
                    # progress bar
                    pbar.set_postfix(status)
                    pbar.update()
                    
                    # logs/checkpoints
                    client_states = {"consumed_samples": steps * args.rollout_batch_size}
                    self.save_logs_and_checkpoints(args, steps, status, client_states)
                
                    steps = steps + 1
                else:
                    # First rollout logic
                    logger.info(f"First rollout, waiting for rollout to complete...")
                    rollout_experiences = ray.get(rollout_ref)
                    
                for i, experience in enumerate(
                    rollout_experiences
                ):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)

                    self.log_rollouts_wandb(experience.json_rollouts, episode=episode, steps=steps, i_experience=i, train_or_eval="train")

                
                

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
            
    def _train(self, steps, args):
        if self.args.advantage_estimator not in ["group_norm", "dr_grpo"]:
            self.replay_buffer.normalize(
                "advantages", 
                divide_by_std=not self.args.no_advantage_std_norm, 
                world_size=self.strategy.world_size,
                env_maker=vars(self.strategy.args).get("env_maker", False),
            )
        status = self.ppo_train(steps)
        self.replay_buffer.clear()

        if "kl" in status:
            self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
        
        return status

    def ppo_train(self, global_steps):
        status, actor, ema_model, experience_maker = ppo_train_bare(
            global_steps=global_steps,
            freezing_actor_steps=self.strategy.args.freezing_actor_steps,
            experience_maker=self.experience_maker,
            actor=self.actor,
            critic_train_remote=self.critic_train_remote,
            colocate_all_models=self.strategy.args.colocate_all_models,
            deepspeed_enable_sleep=self.strategy.args.deepspeed_enable_sleep,
            vllm_engines=self.vllm_engines,
            vllm_enable_sleep=self.strategy.args.vllm_enable_sleep,
            model_update_group=self._model_update_group,
            use_prefix_cache=self.strategy.args.enable_prefix_caching,
            use_cuda_ipc=self.strategy.args.use_cuda_ipc,
            use_ray=self.strategy.args.vllm_sync_with_ray,
            zero_stage=self.strategy.args.zero_stage,
            world_size=self.strategy.world_size,
            max_epochs=self.strategy.args.max_epochs,
            dataloader_pin_memory=self.dataloader_pin_memory,
            replay_buffer=self.replay_buffer,
            ring_attn_group=self.strategy.ring_attn_group,
            actor_optim=self.actor_optim,
            ema_model=self.ema_model,
            ema_beta=self.strategy.args.ema_beta,
            actor_scheduler=self.actor_scheduler,
            use_kl_loss=self.strategy.args.use_kl_loss,
            kl_estimator=self.strategy.args.kl_estimator,
            use_packing_samples=self.strategy.args.use_packing_samples,
            use_aux_loss=self.strategy.args.use_aux_loss,
            aux_loss_coef=self.strategy.args.aux_loss_coef,
            actor_loss_fn=self.actor_loss_fn,
            initial_model=self.initial_model,
            pretrain_dataloader=self.pretrain_dataloader,
            kl_ctl=self.kl_ctl,
            ptx_coef=self.strategy.args.ptx_coef,
            strategy_time_steps=self.strategy.time_steps,
            strategy_stage=self.strategy.stage,
            strategy_accumulated_gradient=self.strategy.accumulated_gradient,
            ptx_loss_fn=self.strategy.args.ptx_loss_fn,
        )
        self.actor = actor
        self.ema_model = ema_model
        self.experience_maker = experience_maker
        return status

    def ppo_train_actor(self, global_steps):
        status, actor, ema_model = ppo_train_actor_bare(
            global_steps=global_steps,
            replay_buffer=self.replay_buffer,
            ring_attn_group=self.strategy.ring_attn_group,
            dataloader_pin_memory=self.dataloader_pin_memory,
            max_epochs=self.strategy.args.max_epochs,
            ema_model=self.ema_model,
            ema_beta=self.strategy.args.ema_beta,
            actor_scheduler=self.actor_scheduler,
            use_kl_loss=self.strategy.args.use_kl_loss,
            kl_estimator=self.strategy.args.kl_estimator,
            use_packing_samples=self.strategy.args.use_packing_samples,
            use_aux_loss=self.strategy.args.use_aux_loss,
            aux_loss_coef=self.strategy.args.aux_loss_coef,
            actor_loss_fn=self.actor_loss_fn,
            initial_model=self.initial_model,
            pretrain_dataloader=self.pretrain_dataloader,
            kl_ctl=self.kl_ctl,
            ptx_coef=self.strategy.args.ptx_coef,
            strategy_time_steps=self.strategy.time_steps,
            strategy_stage=self.strategy.stage,
            strategy_accumulated_gradient=self.strategy.accumulated_gradient,
            ptx_loss_fn=self.strategy.args.ptx_loss_fn,
        )
        self.actor = actor
        self.ema_model = ema_model
        return status

    def _broadcast_to_vllm(self):
        return _broadcast_to_vllm_bare(
            vllm_engines=self.vllm_engines,
            use_prefix_cache=self.strategy.args.enable_prefix_caching,
            use_cuda_ipc=self.use_cuda_ipc,
            use_ray=self.strategy.args.vllm_sync_with_ray,
            zero_stage=self.strategy.args.zero_stage,
            actor_model=self.actor.model,
            model_update_group=self._model_update_group,
        )

    def save_logs_and_checkpoints(self, args, global_step, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step, args.temperature, args.n_samples_per_prompt, args.eval_steps)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        print(f"{args.save_steps=}")
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
        torch.distributed.barrier()
    
    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1, eval_steps=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        if self.strategy.is_rank_0():
            logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # Only run evaluation on ring attention rank0
        if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:

            with torch.no_grad():
                # First collect all prompts and labels
                all_prompts = []
                all_datasources = []

                for prompts in eval_dataloader:
                    all_prompts.extend(prompts)
                    all_datasources.extend([p.get("datasource", "") for p in prompts])

                # Generate samples and calculate rewards
                generate_kwargs = self.generate_kwargs.copy()
                generate_kwargs["temperature"] = temperature
                generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
                samples = self.experience_maker.generate_samples(all_prompts, **generate_kwargs)

                self.log_rollouts_wandb([sample.json_rollouts for sample in samples], global_step=global_step, train_or_eval="eval")

                # duplicate prompts and labels for each sample
                all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])

                # Calculate rewards
                if samples[0].reward is None:
                    assert False, "Reward model and remote reward are not currently supported with evaluations"
                else:
                    rewards = torch.tensor([sample.reward for sample in samples])

                # Reshape rewards to (num_prompts, n_samples_per_prompt)
                rewards = rewards.reshape(-1, n_samples_per_prompt)

                # Collect local statistics for each data source
                local_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

                for i, datasource in enumerate(all_datasources):
                    if datasource not in local_metrics:
                        local_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                    # Calculate pass@k and pass@1
                    prompt_rewards = rewards[i]
                    local_metrics[datasource][f"pass{n_samples_per_prompt}"] += prompt_rewards.max().float().item()
                    local_metrics[datasource]["pass1"] += prompt_rewards.mean().float().item()
                    local_metrics[datasource]["count"] += 1

                # All gather metrics from all ranks
                gathered_metrics = [None] * (self.strategy.world_size // self.strategy.ring_attn_size)
                if self.strategy.ring_attn_group is not None:
                    # Only rank 0 in ring attention group gathers metrics
                    torch.distributed.all_gather_object(
                        gathered_metrics, local_metrics, group=self.experience_maker.ring_rank0_group
                    )
                else:
                    torch.distributed.all_gather_object(gathered_metrics, local_metrics)

                # Only rank0 processes the gathered metrics
                if self.strategy.is_rank_0():
                    logger.info(f"Evaluating {len(gathered_metrics)} datasources")
                    # Combine metrics from all ranks
                    global_metrics = {}
                    for rank_metrics in gathered_metrics:
                        for datasource, metrics in rank_metrics.items():
                            if datasource not in global_metrics:
                                global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}
                            global_metrics[datasource][f"pass{n_samples_per_prompt}"] += metrics[
                                f"pass{n_samples_per_prompt}"
                            ]
                            global_metrics[datasource]["pass1"] += metrics["pass1"]
                            global_metrics[datasource]["count"] += metrics["count"]

                    # Calculate global averages
                    logs = {}
                    for datasource, metrics in global_metrics.items():
                        logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                            metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                        )
                        logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

                    # Log to wandb/tensorboard
                    if self._wandb is not None:
                        logger.info(f"Logging to wandb")
                        # Convert metrics to use eval/ prefix
                        eval_logs = {"eval/%s" % k: v for k, v in logs.items()}
                        # Add the epoch counter (different from global_step)
                        eval_logs["eval/epoch"] = global_step // eval_steps
                        self._wandb.log(eval_logs)
                    elif self._tensorboard is not None:
                        for k, v in logs.items():
                            self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()

        end_time = time.time()
        duration = end_time - start_time
        if self.strategy.is_rank_0():
            time_str = str(datetime.timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Evaluation completed in {time_str}")

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def log_rollouts_wandb(self, json_rollouts, episode=None, steps=None, i_experience=None, global_step=None, train_or_eval=None) -> None:
        if self._wandb is None:
            return
        if not self.strategy.is_rank_0():
            return

        name = "rollouts"
        if train_or_eval is not None:
            name += f"-{train_or_eval}"
        if episode is not None:
            name += f"-episode-{episode}"
        if steps is not None:
            name += f"-steps-{steps}"
        if i_experience is not None:
            name += f"-experience-{i_experience}"
        if global_step is not None:
            name += f"-global-step-{global_step}"

        try:
            text_rollouts = json.dumps(json_rollouts)
            filename = name + ".json"
        except json.decoder.JSONDecodeError:
            text_rollouts = str(json_rollouts)
            filename = name + ".txt"

        artifact = self._wandb.Artifact(name=name, type="rollouts")
        with artifact.new_file(filename) as f:
            f.write(text_rollouts)
        self._wandb.log_artifact(artifact)

def ppo_train_bare(
        global_steps: int,
        freezing_actor_steps: int,
        experience_maker: RemoteExperienceMaker,
        actor: Actor,
        critic_train_remote: bool,
        colocate_all_models: bool,
        deepspeed_enable_sleep: bool,
        vllm_engines: List[ray.actor.ActorHandle],
        vllm_enable_sleep: bool,
        model_update_group: str,
        use_prefix_cache: bool,
        use_cuda_ipc: bool,
        use_ray: bool,
        zero_stage: int,
        world_size: int,
        max_epochs: int,
        dataloader_pin_memory: bool,
        replay_buffer: NaiveReplayBuffer,
        ring_attn_group: str,
        actor_optim: Optimizer,
        ema_model: Actor,
        ema_beta: float,
        actor_scheduler,
        use_kl_loss: bool,  
        kl_estimator: str,
        use_packing_samples: bool,
        use_aux_loss: bool,
        aux_loss_coef: float,
        actor_loss_fn,
        initial_model: Optional[Actor],
        pretrain_dataloader: Optional[DataLoader],
        kl_ctl: Union[FixedKLController, AdaptiveKLController],
        ptx_coef: float,
        strategy_time_steps,
        strategy_stage,
        strategy_accumulated_gradient,
        ptx_loss_fn: GPTLMLoss,
    ) -> Tuple[Dict[str, float], Actor, Actor, RemoteExperienceMaker]:
        # 1. ensure all experience makers done
        #torch.distributed.barrier()
        status = {}

        # 2. triger remote critic model training
        if critic_train_remote:
            raise NotImplementedError("Critic training not implemented")

        if colocate_all_models:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > freezing_actor_steps:
            if deepspeed_enable_sleep:
                reload_deepspeed_states(actor.model)

            status, actor, ema_model = ppo_train_actor_bare(
                global_steps=global_steps,
                replay_buffer=replay_buffer,
                ring_attn_group=ring_attn_group,
                dataloader_pin_memory=dataloader_pin_memory,
                max_epochs=max_epochs,
                world_size=world_size,
                actor=actor,
                actor_optim=actor_optim,
                ema_model=ema_model,
                ema_beta=ema_beta,
                actor_scheduler=actor_scheduler,
                use_kl_loss=use_kl_loss,
                kl_estimator=kl_estimator,
                use_packing_samples=use_packing_samples,
                use_aux_loss=use_aux_loss,
                aux_loss_coef=aux_loss_coef,
                actor_loss_fn=actor_loss_fn,
                initial_model=initial_model,
                pretrain_dataloader=pretrain_dataloader,
                kl_ctl=kl_ctl,
                ptx_coef=ptx_coef,
                strategy_time_steps=strategy_time_steps,
                strategy_stage=strategy_stage,
                strategy_accumulated_gradient=strategy_accumulated_gradient,
                ptx_loss_fn=ptx_loss_fn,
            )

            status.update(status)

            if deepspeed_enable_sleep:
                offload_deepspeed_states(actor.model)

            torch.cuda.empty_cache()

            # 4. broadcast weights to vllm engines
            if vllm_engines is not None:
                if vllm_enable_sleep:
                    batch_vllm_engine_call(vllm_engines, "wake_up")

                torch.distributed.barrier()
                torch.cuda.synchronize()
                _broadcast_to_vllm_bare(
                    vllm_engines=vllm_engines,
                    use_prefix_cache=use_prefix_cache,
                    use_cuda_ipc=use_cuda_ipc,
                    use_ray=use_ray,
                    zero_stage=zero_stage,
                    actor=actor.model,
                    model_update_group=model_update_group,
                )

                if vllm_enable_sleep:
                    batch_vllm_engine_call(vllm_engines, "sleep")
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        # 5. wait remote critic model training done
        if critic_train_remote and not colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status, actor, ema_model
    
def ppo_train_actor_bare(
    global_steps: int,
    replay_buffer: NaiveReplayBuffer,
    ring_attn_group,
    dataloader_pin_memory: bool,
    max_epochs: int,
    world_size: int,
    actor: Actor,
    ema_model: Actor,
    ema_beta: float,
    actor_scheduler,
    use_kl_loss: bool,  
    kl_estimator: str,
    use_packing_samples: bool,
    use_aux_loss: bool,
    aux_loss_coef: float,
    actor_loss_fn,
    initial_model: Optional[Actor],
    pretrain_dataloader: Optional[DataLoader],
    kl_ctl: Union[FixedKLController, AdaptiveKLController],
    ptx_coef: float,
    strategy_time_steps,
    strategy_stage,
    strategy_accumulated_gradient,
    ptx_loss_fn: GPTLMLoss,
) -> Tuple[Dict[str, float], Actor, Actor]:
    torch.cuda.empty_cache()
    # replay buffer may be empty at first, we should rebuild at each training
    dataloader = DataLoader(
        replay_buffer,
        batch_size=replay_buffer.sample_batch_size,
        shuffle=False if ring_attn_group is not None else True,
        drop_last=True,
        pin_memory=dataloader_pin_memory,
        collate_fn=replay_buffer.collate_fn,
    )
    device = torch.cuda.current_device()

    status_list = []
    status_mean = {}
    for epoch in range(max_epochs):
        pbar = tqdm(
            dataloader,
            desc=f"Train epoch [{epoch + 1}/{max_epochs}]",
            disable=torch.distributed.get_rank() != 0,
        )
        for experience in pbar:
            experience.to_device(device)
            status, actor, ema_model = training_step_bare(
                experience=experience,
                actor=actor,
                ema_model=ema_model,
                ema_beta=ema_beta,
                actor_scheduler=actor_scheduler,
                use_kl_loss=use_kl_loss,
                kl_estimator=kl_estimator,
                use_packing_samples=use_packing_samples,
                use_aux_loss=use_aux_loss,
                aux_loss_coef=aux_loss_coef,
                actor_loss_fn=actor_loss_fn,
                initial_model=initial_model,
                pretrain_dataloader=pretrain_dataloader,
                kl_ctl=kl_ctl,
                ptx_coef=ptx_coef,
                strategy_time_steps=strategy_time_steps,
                strategy_stage=strategy_stage,
                strategy_accumulated_gradient=strategy_accumulated_gradient,
                ptx_loss_fn=ptx_loss_fn,
            )

            # for DP
            # weighted mean for kl
            if "kl" in status:
                status["kl"] *= status["response_length"]
                status = all_reduce_bare(status, world_size)
                status["kl"] /= status["response_length"]

            short_status = {}

            if "policy_loss" in status:
                short_status = {
                    "pg": status["policy_loss"],
                    "rm": status["reward"],
                    "ret": status["return"],
                    "glen": status["response_length"],
                    "tlen": status["total_length"],
                    "kl": status["kl"],
                    "act_lr": status["actor_lr"],
                }

            if "critic_loss" in status:
                short_status["cri"] = status["critic_loss"]
                short_status["vals"] = status["values"]
                short_status["cri_lr"] = status["critic_lr"]

            if "ptx_loss" in status:
                short_status["ptx"] = status["ptx_loss"]

            status_list.append(status)
            pbar.set_postfix(short_status)

    if status_list:
        status_mean = status_list[0]
        for m in status_list[1:]:
            for k, v in m.items():
                status_mean[k] += v
        for k in status_mean.keys():
            status_mean[k] /= len(status_list)
    torch.cuda.empty_cache()
    return status_mean, actor, ema_model

def all_reduce_bare(data, world_size: int, op="mean"):
    assert op in ("mean", "max", "sum")
    if isinstance(data, dict):
        ret = {}
        for k, v in data.items():
            ret[k] = all_reduce_bare(v, world_size, op)
        return ret
    else:
        is_tensor = True
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor([data])
            is_tensor = False
        is_cpu_tensor = data.device.type == "cpu"

        if is_cpu_tensor:
            data = data.to(torch.cuda.current_device())
        if op == "mean":
            data /= world_size
        torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.MAX if op == "max" else torch.distributed.ReduceOp.SUM)
        if is_cpu_tensor:
            data = data.cpu()
        return data.item() if not is_tensor else data

def training_step_bare(
    actor: Actor,
    ema_model: Actor,
    ema_beta: float,
    actor_scheduler,
    experience: Experience,
    ring_attn_group: str,
    use_kl_loss: bool,
    kl_estimator: str,
    use_packing_samples: bool,
    use_aux_loss: bool,
    aux_loss_coef: float,
    actor_loss_fn,
    ptx_loss_fn: GPTLMLoss,
    initial_model: Optional[Actor],
    pretrain_dataloader: Optional[DataLoader],
    kl_ctl: Union[FixedKLController, AdaptiveKLController],
    ptx_coef: float,
    strategy_time_steps,
    strategy_stage,
    strategy_accumulated_gradient,
) -> Tuple[Dict[str, float], Actor, Actor]:
    actor.train()

    # TODO: this is a bad indicator to say that data is packed...
    if isinstance(experience.sequences, list):
        sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
        old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
        advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
        num_actions = [v.numel() for v in experience.advantages]
        packed_seq_lens = [s.numel() for s in experience.sequences]
        attention_mask = torch.cat(
            [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
        ).unsqueeze(0)
        # pad seq makes the sequence a multiple of ring_attention_size.
        if ring_attn_group is not None:
            pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                sequences, attention_mask, num_actions, packed_seq_lens, ring_attn_group
            )
        if use_kl_loss and experience.base_action_log_probs is not None:
            base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
    else:
        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        num_actions = experience.action_mask.size(1)
        packed_seq_lens = None
        attention_mask = experience.attention_mask
        if use_kl_loss and experience.base_action_log_probs is not None:
            base_action_log_probs = experience.base_action_log_probs

    # actor loss
    action_log_probs, output = actor(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        return_output=True,
        ring_attn_group=ring_attn_group,
        logps_allgather=True,
        packed_seq_lens=packed_seq_lens,
    )
    # unpad sequence ensures that pad tokens do not contribute to the loss calculation.
    if ring_attn_group is not None:
        assert pad_len is not None
        sequences, attention_mask, num_actions, packed_seq_lens, action_log_probs, _, _ = unpad_sequences(
            pad_len=pad_len,
            sequences=sequences,
            attention_mask=attention_mask,
            num_actions=num_actions,
            packed_seq_lens=packed_seq_lens,
            action_log_probs=action_log_probs,
            ring_attn_group=ring_attn_group,
        )

    # loss function
    actor_loss = actor_loss_fn(
        action_log_probs,
        old_action_log_probs,
        advantages,
        action_mask=experience.action_mask,
    )

    if use_kl_loss:
        if initial_model is not None:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                experience.action_mask,
                kl_estimator=kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        if not use_packing_samples:
            kl_mean = masked_mean(kl, experience.action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=action_log_probs.device)

        kl_loss = kl_mean.mean()
        experience.info["kl"] = kl_loss.item()
    else:
        kl_loss = 0

    # mixtral
    if use_aux_loss:
        aux_loss = output.aux_loss
    else:
        aux_loss = 0
    loss = actor_loss + aux_loss * aux_loss_coef + kl_loss * kl_ctl.value
    
    # Note: this modifies in place, so we have to return the actor at the end
    actor.model.backward(loss)

    # ptx loss
    if pretrain_dataloader is not None:
        data = next(pretrain_dataloader)
        inputs = data[1].squeeze(1).to(torch.cuda.current_device())
        attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
        label = torch.where(
            attention_mask.bool(),
            inputs,
            ptx_loss_fn.IGNORE_INDEX,
        )

        output = actor(inputs, attention_mask=attention_mask, return_output=True)
        ptx_log_probs = output["logits"]

        # loss function
        ptx_loss = ptx_loss_fn(ptx_log_probs, label)
        # mixtral
        if use_aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = ptx_loss + aux_loss * aux_loss_coef
        actor.model.backward(ptx_coef * loss)

    actor.model.step()
    if ema_model:
        strategy_time_steps["ema"] += 1
        if strategy_time_steps["ema"] % strategy_accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(actor.parameters(), ema_model.parameters()):
                    if param.requires_grad:
                        if strategy_stage != 3:
                            data = param.data.to("cuda")
                            param_ema.data.copy_((1 - ema_beta) * data + ema_beta * param_ema.data)
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = [p for p in [param, param_ema] if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]
                            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                                data = param.data.to("cuda")
                                param_ema.data.copy_((1 - ema_beta) * data + ema_beta * param_ema.data)

    # status
    status = {"policy_loss": actor_loss.item(), "actor_lr": actor_scheduler.get_last_lr()[0]}
    if pretrain_dataloader is not None:
        status["ptx_loss"] = ptx_loss.item()
    for k, v in experience.info.items():
        if k == "kl":
            status[k] = (
                (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
            ).item()
        else:
            status[k] = v.mean().item()
    return (status, actor, ema_model)

def _broadcast_to_vllm_bare(
    vllm_engines: List[ray.actor.ActorHandle],
    use_prefix_cache: bool,
    use_cuda_ipc: bool,
    use_ray: bool,
    zero_stage: int,
    actor_model,
    model_update_group: str,
):
    cache_reset_refs = []
    if use_prefix_cache and torch.distributed.get_rank() == 0:
        # clear prefix cache
        for engine in vllm_engines:
            cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = actor_model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not use_cuda_ipc:
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=zero_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=zero_stage == 3):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=count == num_params,
                            )
                            for engine in vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step, args.n_samples_per_prompt, args.eval_steps)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        print(f"{args.save_steps=}")
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
        torch.distributed.barrier()
    
    def evaluate(self, eval_dataloader, global_step, n_samples_per_prompt=1, eval_steps=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        if self.strategy.is_rank_0():
            logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # Only run evaluation on ring attention rank0
        if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:

            with torch.no_grad():
                # First collect all prompts and labels
                all_prompts = []
                all_datasources = []
                
                for prompts in iter(eval_dataloader):
                    all_prompts.extend(prompts)
                    all_datasources.extend([p.get("datasource", "") for p in prompts])
                    
                all_prompts = all_prompts[:8]
                all_datasources = all_datasources[:8]
                    
                # Logging
                logger.info(f"Evaluating {len(all_prompts)} prompts")

                # Generate samples and calculate rewards
                generate_kwargs = self.generate_kwargs.copy()
                generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
                
                samples = self.experience_maker.generate_samples(all_prompts, **generate_kwargs)

                self.log_rollouts_wandb([sample.json_rollouts for sample in samples], global_step=global_step, train_or_eval="eval")

                # Calculate rewards
                if samples[0].reward is None:
                    assert False, "Reward model and remote reward are not currently supported with evaluations"
                else:
                    rewards = torch.tensor([sample.reward for sample in samples])

                # Reshape rewards to (num_prompts, n_samples_per_prompt)
                rewards = rewards.reshape(-1, n_samples_per_prompt)

                # Collect local statistics for each data source
                local_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

                for i, datasource in enumerate(all_datasources):
                    if datasource not in local_metrics:
                        local_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                    # Calculate pass@k and pass@1
                    prompt_rewards = rewards[i]
                    local_metrics[datasource][f"pass{n_samples_per_prompt}"] += prompt_rewards.max().float().item()
                    local_metrics[datasource]["pass1"] += prompt_rewards.mean().float().item()
                    local_metrics[datasource]["count"] += 1

                # All gather metrics from all ranks
                gathered_metrics = [None] * (self.strategy.world_size // self.strategy.ring_attn_size)
                if self.strategy.ring_attn_group is not None:
                    # Only rank 0 in ring attention group gathers metrics
                    torch.distributed.all_gather_object(
                        gathered_metrics, local_metrics, group=self.experience_maker.ring_rank0_group
                    )
                else:
                    torch.distributed.all_gather_object(gathered_metrics, local_metrics)

                # Only rank0 processes the gathered metrics
                if self.strategy.is_rank_0():
                    logger.info(f"Evaluating {len(gathered_metrics)} datasources")
                    # Combine metrics from all ranks
                    global_metrics = {}
                    for rank_metrics in gathered_metrics:
                        for datasource, metrics in rank_metrics.items():
                            if datasource not in global_metrics:
                                global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}
                            global_metrics[datasource][f"pass{n_samples_per_prompt}"] += metrics[
                                f"pass{n_samples_per_prompt}"
                            ]
                            global_metrics[datasource]["pass1"] += metrics["pass1"]
                            global_metrics[datasource]["count"] += metrics["count"]

                    # Calculate global averages
                    logs = {}
                    for datasource, metrics in global_metrics.items():
                        logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                            metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                        )
                        logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

                    # Log to wandb/tensorboard
                    if self._wandb is not None:
                        logger.info(f"Logging to wandb")
                        # Convert metrics to use eval/ prefix
                        eval_logs = {"eval/%s" % k: v for k, v in logs.items()}
                        # Add the epoch counter (different from global_step)
                        eval_logs["eval/epoch"] = global_step // eval_steps
                        self._wandb.log(eval_logs)
                    elif self._tensorboard is not None:
                        for k, v in logs.items():
                            self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        end_time = time.time()
        duration = end_time - start_time
        if self.strategy.is_rank_0():
            time_str = str(datetime.timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Evaluation completed in {time_str}")
            

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def log_rollouts_wandb(self, json_rollouts, episode=None, steps=None, i_experience=None, global_step=None, train_or_eval=None) -> None:
        if self._wandb is None:
            return
        if not self.strategy.is_rank_0():
            return

        name = "rollouts"
        if train_or_eval is not None:
            name += f"-{train_or_eval}"
        if episode is not None:
            name += f"-episode-{episode}"
        if steps is not None:
            name += f"-steps-{steps}"
        if i_experience is not None:
            name += f"-experience-{i_experience}"
        if global_step is not None:
            name += f"-global-step-{global_step}"

        try:
            text_rollouts = json.dumps(json_rollouts)
            filename = name + ".json"
        except json.decoder.JSONDecodeError:
            text_rollouts = str(json_rollouts)
            filename = name + ".txt"

        # artifact = self._wandb.Artifact(name=name, type="rollouts")
        # with artifact.new_file(filename) as f:
        #     f.write(text_rollouts)
        # self._wandb.log_artifact(artifact)


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_datasets()

        # configure scheduler
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=args.eval_steps > 0,  # Only get eval split if we're doing evaluation
            train_split=args.prompt_split,
            eval_ratio=args.eval_ratio
        )

        # Handle train/eval split if needed
        if args.eval_steps > 0:
            train_data = data["train"]
            eval_data = data["validation"]
        else:
            train_data = data

        # Create train dataset and dataloader (existing code)
        self.prompts_dataset = PromptDataset(
            train_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
            True,
            shuffle=True, collate_fn=custom_collate_fn 
        )

        # Create eval dataloader if needed
        if args.eval_steps > 0:
            self.eval_dataset = PromptDataset(
                eval_data, self.tokenizer, strategy, input_template=args.input_template
            )
            self.eval_dataloader = strategy.setup_dataloader(
                self.eval_dataset,
                args.rollout_batch_size // strategy.world_size,
                True,
                shuffle=False,  # No need to shuffle eval data
                collate_fn=custom_collate_fn
            )
        else:
            self.eval_dataloader = None

        # Handle pretrain data (existing code)
        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(min(len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt))
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # for GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

            trainer._broadcast_to_vllm()

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "sleep")
                torch.distributed.barrier()
                torch.cuda.synchronize()

        trainer.eval_dataloader = self.eval_dataloader

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )


def custom_collate_fn(batch):
    return batch

## Ray Remote Functions

@ray.remote
def make_experience_list_remote(experience_maker: RemoteExperienceMaker, prompts, **generate_kwargs):
    return experience_maker.make_experience_list(prompts, **generate_kwargs)

@ray.remote
def train_remote(
    trainer: ActorPPOTrainer,
    args,
    replay_buffer,
    global_steps,
    freezing_actor_steps,
    actor,
    critic_train_remote,
    colocate_all_models,
    deepspeed_enable_sleep,
    vllm_engines,
    vllm_enable_sleep,
    model_update_group,
    use_prefix_cache,
    use_cuda_ipc,
    use_ray,
    zero_stage,
    world_size,
    max_epochs,
    dataloader_pin_memory,
    ring_attn_group,
    actor_optim,
    ema_model,
    ema_beta,
    actor_scheduler,
    use_kl_loss,
    kl_estimator,
    use_packing_samples,
    use_aux_loss,
    aux_loss_coef,
    actor_loss_fn,
    initial_model,
    pretrain_dataloader,
    kl_ctl,
    ptx_coef,
    strategy_time_steps,
    strategy_stage,
    strategy_accumulated_gradient,
    ptx_loss_fn,    
):
    if trainer.args.advantage_estimator not in ["group_norm", "dr_grpo"]:
        replay_buffer.normalize(
            "advantages", 
            divide_by_std=not trainer.args.no_advantage_std_norm, 
            world_size=trainer.strategy.world_size,
            env_maker=vars(trainer.strategy.args).get("env_maker", False),
        )
    status, actor, ema_model = ppo_train_bare(
        global_steps=global_steps,
        freezing_actor_steps=freezing_actor_steps,
        actor=actor,
        critic_train_remote=critic_train_remote,
        colocate_all_models=colocate_all_models,
        deepspeed_enable_sleep=deepspeed_enable_sleep,
        vllm_engines=vllm_engines,
        vllm_enable_sleep=vllm_enable_sleep,
        model_update_group=model_update_group,
        use_prefix_cache=use_prefix_cache,
        use_cuda_ipc=use_cuda_ipc,
        use_ray=use_ray,
        zero_stage=zero_stage,
        world_size=world_size,
        max_epochs=max_epochs,
        dataloader_pin_memory=dataloader_pin_memory,
        replay_buffer=replay_buffer,
        ring_attn_group=ring_attn_group,
        actor_optim=actor_optim,
        ema_model=ema_model,
        ema_beta=ema_beta,
        actor_scheduler=actor_scheduler,
        use_kl_loss=use_kl_loss,
        kl_estimator=kl_estimator,
        use_packing_samples=use_packing_samples,
        use_aux_loss=use_aux_loss,
        aux_loss_coef=aux_loss_coef,
        actor_loss_fn=actor_loss_fn,
        initial_model=initial_model,
        pretrain_dataloader=pretrain_dataloader,
        kl_ctl=kl_ctl,
        ptx_coef=ptx_coef,
        strategy_time_steps=strategy_time_steps,
        strategy_stage=strategy_stage,
        strategy_accumulated_gradient=strategy_accumulated_gradient,
        ptx_loss_fn=ptx_loss_fn,
    )

    replay_buffer.clear()

    if "kl" in status:
        trainer.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
    
    return status, actor, ema_model

def get_train_ref(trainer, steps, args):
    return train_remote.remote(
        trainer=trainer,
        steps=steps,
        args=args,
        global_steps=steps,
        freezing_actor_steps=trainer.freezing_actor_steps,
        experience_maker=trainer.experience_maker,
        actor=trainer.actor,
        critic_train_remote=trainer.critic_train_remote,
        colocate_all_models=trainer.colocate_all_models,
        deepspeed_enable_sleep=trainer.deepspeed_enable_sleep,
        vllm_engines=trainer.vllm_engines,
        vllm_enable_sleep=trainer.vllm_enable_sleep,
        model_update_group=trainer._model_update_group,
        use_prefix_cache=trainer.strategy.args.enable_prefix_caching,
        use_cuda_ipc=trainer.use_cuda_ipc,
        use_ray=trainer.strategy.args.vllm_sync_with_ray,
        zero_stage=trainer.strategy.args.zero_stage,
        world_size=trainer.strategy.world_size,
        max_epochs=trainer.max_epochs,
        dataloader_pin_memory=trainer.dataloader_pin_memory,
        replay_buffer=trainer.replay_buffer,
        ring_attn_group=trainer.strategy.ring_attn_group,
        actor_optim=trainer.actor_optim,
        ema_model=trainer.ema_model,
        ema_beta=trainer.ema_beta,
        actor_scheduler=trainer.actor_scheduler,
        use_kl_loss=trainer.use_kl_loss,
        kl_estimator=trainer.kl_estimator,
        use_packing_samples=trainer.use_packing_samples,
        use_aux_loss=trainer.use_aux_loss,
        aux_loss_coef=trainer.aux_loss_coef,
        actor_loss_fn=trainer.actor_loss_fn,
        initial_model=trainer.initial_model,
        pretrain_dataloader=trainer.pretrain_dataloader,
        kl_ctl=trainer.kl_ctl,
        ptx_coef=trainer.ptx_coef,
        strategy_time_steps=trainer.strategy.time_steps,
        strategy_stage=trainer.strategy.stage,
        strategy_accumulated_gradient=trainer.strategy.accumulated_gradient,
        ptx_loss_fn=trainer.ptx_loss_fn,
    )
