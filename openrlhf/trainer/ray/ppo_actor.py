from copy import deepcopy
import itertools
import math
import os
import random
import socket
import time
from typing import Any, Callable, Dict, List

import deepspeed
import ray
import torch
import torch.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.datasets.prompts_dataset import RedTeamGamePromptDataset
from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, masked_mean, unpacking_samples
from openrlhf.trainer import BasePPOTrainer
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker
from openrlhf.utils import blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy

from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.distributed_util import init_process_group
from red_team import GameOutcome
from red_team.prompts import DEFENDER_INSTRUCTION_COT_PROMPT
from red_team.utils import convert_game_history_to_messages

from .launcher import BasePPORole
from .utils import get_physical_gpu_id


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
        self.defender_vllm_engines = kwargs.get("defender_vllm_engines", None)
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
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("game_log", step_metric="train/global_step")

            # update config
            from red_team.utils import REWARD_COEFF_CONFIG
            self._wandb.config.update({"REWARD_COEFF_CONFIG": REWARD_COEFF_CONFIG})

        # counter for SFT samples trained
        self.total_sft_samples_trained = torch.tensor(0.0)

        # PTX-SFT-chatability boost
        # self.sftchat_loss = self.args.sftchat_loss_coef > 0
        self.postfill_cot_loss = self.args.postfill_cot_loss_coef > 0

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
            defender_vllm_engines=self.defender_vllm_engines,
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
        eval_dataloader,
        sft_dataloader,
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
        self.holdout_dataloader = eval_dataloader
        self.sft_dataloader = sft_dataloader

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

            for rand_prompts, labels in self.prompts_dataloader:
                experiences = self.experience_maker.make_experience_list(rand_prompts, labels, **self.generate_kwargs)
                for i, experience in enumerate(experiences):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)
                
                status = {}

                # compute training diversity metrics
                if steps % args.diversity_score_steps == 0:
                    start_time = time.time()
                    self_bleu_scores = self.replay_buffer.compute_training_diversity_metrics(self.tokenizer, self.strategy, bleu_limit=32)
                    end_time = time.time()
                    self.strategy.print(f"Time taken for compute_training_diversity_metrics: {end_time - start_time:.2f} seconds")
                    status.update(self_bleu_scores)

                    if not self.args.custom_configs.get("no_attacker_turn", False):
                        revision_similarity_metrics = self.replay_buffer.compute_revised_similarity_metrics(self.strategy, self.tokenizer)
                        status.update(revision_similarity_metrics)

                # compute game metrics
                game_status = self.replay_buffer.compute_game_metrics(self.strategy)
                status.update(game_status)

                # remove ties
                if self.args.custom_configs.get("remove_ties", False):
                    self.replay_buffer.remove_ties(self.strategy)
                if self.args.custom_configs.get("no_defender_turn", False):
                    self.replay_buffer.remove_defender_turn(self.strategy)
                                                            
                # truncate to same length between different actor's buffers
                self.replay_buffer.truncate_buffer(self.strategy, mode='batch')
                
                if self.strategy.is_rank_0():
                    em_lost_samples = self.experience_maker.lost_samples 
                    rb_lost_samples = self.replay_buffer.lost_samples
                    status.update({
                        "debug/em_lost_samples": em_lost_samples,
                        "debug/rb_lost_samples": rb_lost_samples
                    })
                
                # training
                torch.cuda.empty_cache()
                # self.replay_buffer.normalize("advantages", self.strategy)
                if self.args.advantage_estimator not in ["group_norm", "dr_grpo"]:
                    if not self.args.custom_configs.get('no_attacker_turn', False):
                        self.replay_buffer.normalize(strategy=self.strategy, attribute="advantages", role="attacker")
                    if not self.args.custom_configs.get('no_defender_turn', False):
                        self.replay_buffer.normalize(strategy=self.strategy, attribute="advantages", role="defender")
                    else:
                        self.replay_buffer.normalize(strategy=self.strategy, attribute="advantages", divide_by_std=not self.args.no_advantage_std_norm)
                                            
                # RL + PTX + holdout eval
                train_status = self.ppo_train(steps)
                status.update(train_status)

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(
                    args, steps, pbar, status, client_states, 
                    replay_buffer=self.replay_buffer.items, 
                    game_history=self.experience_maker.game_history)
            
                # NEW: Simplified stop logic after step 200 if HF checkpoint for step 200 exists
                if args.stop_at_step_200 and steps >= 200:
                    # Only proceed with this check if HF checkpoints are configured to be saved
                    if args.save_hf_ckpt:
                        hf_checkpoint_dir_for_step_200 = os.path.join(args.ckpt_path, "global_step200_hf")
                        if os.path.exists(hf_checkpoint_dir_for_step_200):
                            self.strategy.print(f"HF Checkpoint for step 200 found at {hf_checkpoint_dir_for_step_200}. Terminating training due to stop_at_step_200.")
                            pbar.close()
                            if self._wandb is not None and self.strategy.is_rank_0():
                                self._wandb.finish()
                            if self._tensorboard is not None and self.strategy.is_rank_0():
                                self._tensorboard.close()
                            self.strategy.print("Exiting training as step 200 HF checkpoint exists and current step >= 200 (stop_at_step_200 is True).")
                            return # Exit the fit method

                        # Optional: Log that we are waiting if the HF checkpoint isn't found yet (and HF saving is on)
                        # elif steps % args.logging_steps == 0:
                        #     self.strategy.print(f"Step {steps} > 200. Waiting for HF checkpoint for step 200 (save_steps: {args.save_steps}, save_hf_ckpt: {args.save_hf_ckpt}). Path checked: {hf_checkpoint_dir_for_step_200}")
                    # elif steps % args.logging_steps == 0: # If HF checkpoints are not saved, this logic won't trigger termination.
                        # self.strategy.print(f"Step {steps} > 200. save_hf_ckpt is False, so termination based on HF step 200 checkpoint is skipped.")

                self.replay_buffer.clear()
                torch.cuda.empty_cache()
                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}

        # 2. triger remote critic model training
        if self.critic_train_remote:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.reload_states.remote())

            critic_status_ref = self.critic.fit.remote()

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref))
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.offload_states.remote())

        if self.strategy.args.colocate_all_models:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.reload_states()

            status.update(self.ppo_train_actor(global_steps))

            if self.strategy.args.deepspeed_enable_sleep:
                self.offload_states()

            torch.cuda.empty_cache()

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                torch.distributed.barrier()
                torch.cuda.synchronize()
                self._broadcast_to_vllm()

                # 4.1. after vllm sync, use the vllm engines to evaluate diversity on holdout set
                if global_steps >= self.args.eval_start_steps and global_steps % self.args.eval_steps == 0:
                    # Print a beautiful message indicating evaluation has started
                    if self.strategy.is_rank_0():
                        self.strategy.print("\n" + "="*50)
                        self.strategy.print("🔍 Starting diversity evaluation on holdout set...")
                        self.strategy.print(f"📊 Evaluation at step {global_steps}")
                        self.strategy.print("="*50 + "\n")
                    status.update(self.evaluate_holdout(global_steps))

                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        # 5. wait remote critic model training done
        if self.critic_train_remote and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status
    
    def evaluate_holdout(self, global_steps: int) -> dict:
        """
        Evaluate model diversity on holdout dataset using vLLM engines.
        1. Generate completions on holdout prompts using vLLM
        2. Extract valid answers from CoT format
        3. Compute diversity metrics (self-BLEU and SBERT)
        4. Aggregate metrics across ranks
        5. Return metrics for logging
        """
        
        is_cot_enabled = not self.strategy.args.custom_configs.get('direct_chat_no_cot', False)

        if not hasattr(self, 'holdout_dataloader') or self.holdout_dataloader is None:
            self.strategy.print("No holdout dataloader available for evaluation")
            return {}
        
        if self.vllm_engines is None:
            self.strategy.print("No vLLM engines available for evaluation")
            return {}

        if self.args.custom_configs.get("no_attacker_turn", False):
            self.strategy.print("Under no_attacker_turn mode, we skip holdout evaluation")
            return {}
        
        # Setup
        from red_team.utils import cot_format_check_and_extract
        
        # Diversity metrics dictionary
        holdout_metrics = {}
        
        # 1. Get prompts from holdout dataloader
        # prompts=all_prompts[0]
        # completions=all_prompts[1]
        # data_types=all_labels

        holdout_prompts = []
        holdout_labels = []
        
        # Take first batch from holdout dataloader
        for (prompts, completions), labels in self.holdout_dataloader:
            holdout_prompts = prompts
            holdout_labels = labels
            break
            
        if not holdout_prompts:
            self.strategy.print("No prompts in holdout dataloader")
            return {}
            
        # 2. Generate completions using vLLM engines through RemoteExperienceMaker
        holdout_prompts_chat_messages = []
        for prompt, label in zip(holdout_prompts, holdout_labels):
            assert label == "vanilla_harmful" # does not make sense other than vanilla_harmful labels, as holdout dataset does not need to mark data as generated_harmful
            holdout_prompts_chat_messages.append(
                convert_game_history_to_messages(
                    history=[], 
                    player_role="attacker", 
                    prompt=prompt, 
                    prompt_type="generated_harmful", 
                    tokenizer=self.tokenizer
                    )
                )

        all_outputs = self.experience_maker._generate_vllm(self.vllm_engines, holdout_prompts_chat_messages, holdout_labels)
        
        # 3. Extract answers from CoT responses
        extracted_answers = []
        valid_count = 0
        
        # Store original prompts and revised answers for similarity calculation
        original_prompts = []
        revised_prompts = []
        
        for idx, output in enumerate(all_outputs):
            completion = self.tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)
            if is_cot_enabled:
                (_, answer), format_violation = cot_format_check_and_extract(completion)
                if not format_violation and answer:
                    extracted_answers.append(answer)
                    # Save original prompt and revised answer for similarity calculation
                    # Check if we're within the range of holdout_prompts
                    if idx < len(holdout_prompts):
                        original_prompts.append(holdout_prompts[idx])
                        revised_prompts.append(answer)
                    valid_count += 1
            else:
                extracted_answers.append(completion)
                if idx < len(holdout_prompts):
                    original_prompts.append(holdout_prompts[idx])
                    revised_prompts.append(completion)
                valid_count += 1

        self.strategy.print(f"Valid answer count: {valid_count} / {len(all_outputs)}")
                
        # 4. Compute diversity metrics
        # Import required functions
        from openrlhf.trainer.ppo_utils.replay_buffer import compute_self_bleu, compute_batch_sbert_similarity
        from openrlhf.utils.self_bleu import compute_bleu_score
        
        # Calculate metrics locally
        local_metrics = {}
        start_time = time.time()
        if extracted_answers:
            local_metrics['holdout_self_bleu'] = torch.tensor(compute_self_bleu(extracted_answers)).unsqueeze(0)
            local_metrics['holdout_sbert'] = torch.tensor(compute_batch_sbert_similarity(extracted_answers, self.replay_buffer.sbert_model)).unsqueeze(0)
            local_metrics['holdout_sample_count'] = torch.tensor(len(extracted_answers)).unsqueeze(0)
            
            # Compute similarity metrics between original and revised prompts
            if original_prompts and revised_prompts:
                # Compute SBERT similarity for each pair
                sbert_similarities = []
                for orig, rev in zip(original_prompts, revised_prompts):
                    # Compute embedding similarity between original and revised
                    embeddings = self.replay_buffer.sbert_model.encode([orig, rev])
                    similarity = self.replay_buffer.sbert_model.similarity(
                        torch.tensor(embeddings[0]).unsqueeze(0), 
                        torch.tensor(embeddings[1]).unsqueeze(0)
                    ).item()
                    sbert_similarities.append(similarity)
                    
                # Compute BLEU scores for each pair
                bleu_scores = []
                for orig, rev in zip(original_prompts, revised_prompts):
                    bleu_score = compute_bleu_score(orig, rev)
                    bleu_scores.append(bleu_score)
                
                # Compute WANLI scores for each pair
                wanli_scores = []
                for orig, rev in zip(original_prompts, revised_prompts):
                    wanli_score = self.replay_buffer.nli_model.compute_wanli_score(orig, rev, mode="non_contradiction")
                    wanli_scores.append(wanli_score)

                # Calculate average scores
                import numpy as np
                avg_sbert_similarity = np.mean(sbert_similarities) if sbert_similarities else 0.0
                avg_bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0
                avg_wanli_score = np.mean(wanli_scores) if wanli_scores else 0.0
                
                # Add to local metrics
                local_metrics['holdout_revision_inv_sbert'] = torch.tensor(1. - avg_sbert_similarity).unsqueeze(0)
                local_metrics['holdout_revision_inv_bleu_score'] = torch.tensor(1. - avg_bleu_score).unsqueeze(0)
                local_metrics['holdout_revision_wanli_non_contradiction_prob'] = torch.tensor(avg_wanli_score).unsqueeze(0)
            end_time = time.time()
            self.strategy.print(f"Time taken for compute_self_bleu and compute_batch_sbert_similarity: {end_time - start_time:.2f} seconds")
        else:
            local_metrics['holdout_self_bleu'] = torch.tensor([float('nan')]).unsqueeze(0)
            local_metrics['holdout_sbert'] = torch.tensor([float('nan')]).unsqueeze(0)
            local_metrics['holdout_sample_count'] = torch.tensor([0]).unsqueeze(0)
            local_metrics['holdout_revision_inv_sbert'] = torch.tensor([float('nan')]).unsqueeze(0)
            local_metrics['holdout_revision_inv_bleu_score'] = torch.tensor([float('nan')]).unsqueeze(0)
            local_metrics['holdout_revision_wanli_non_contradiction_prob'] = torch.tensor([float('nan')]).unsqueeze(0)

        # 5. Aggregate metrics across ranks
        all_metrics = self.strategy.all_gather(local_metrics)
        
        # Calculate means (excluding NaN values)
        def _safe_tensor_means(tensors_dict):
            means = {}
            for k, v in tensors_dict.items():
                if k == 'holdout_sample_count':
                    # Sum the sample counts across all ranks
                    means[k] = v.sum().item()
                else:
                    # For other metrics, compute mean (excluding NaNs)
                    # Filter out NaNs
                    valid_values = v[~torch.isnan(v)]
                    means[k] = valid_values.mean().item() if len(valid_values) > 0 else float('nan')
            return means
            
        metric_means = _safe_tensor_means(all_metrics)
        
        # 6. Update diversity metrics dictionary
        holdout_metrics.update({
            'eval/holdout_inv_self_bleu': 1.0 - metric_means['holdout_self_bleu'],
            'eval/holdout_inv_sbert': 1.0 - metric_means['holdout_sbert'],
            'eval/holdout_sample_count': metric_means['holdout_sample_count']
        })
        
        # Add revision similarity metrics if available
        holdout_metrics.update({
            'eval/holdout_revision_inv_sbert': metric_means['holdout_revision_inv_sbert'],
            'eval/holdout_revision_inv_bleu_score': metric_means['holdout_revision_inv_bleu_score'],
            'eval/holdout_revision_wanli_non_contradiction_prob': metric_means['holdout_revision_wanli_non_contradiction_prob']
        })
            
        # Print metrics if rank 0
        if self.strategy.is_rank_0():
            self.strategy.print(f"Holdout evaluation metrics: {holdout_metrics}")
            
        return holdout_metrics

    def ppo_train_actor(self, global_steps):
        torch.cuda.empty_cache()
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=False if self.strategy.ring_attn_group is not None else True,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
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
                    if k not in status_mean:
                        status_mean[k] = v
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        torch.cuda.empty_cache()
        return status_mean

    def training_step(self, experience: Experience, global_steps: int) -> Dict[str, float]:
        self.actor.train()

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
            infos = experience.info
            # pad seq makes the sequence a multiple of ring_attention_size.
            if self.strategy.ring_attn_group is not None:
                pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                    sequences, attention_mask, num_actions, packed_seq_lens, self.strategy.ring_attn_group
                )
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            infos = experience.info
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )
        # unpad sequence ensures that pad tokens do not contribute to the loss calculation.
        if self.strategy.ring_attn_group is not None:
            assert pad_len is not None
            sequences, attention_mask, num_actions, packed_seq_lens, action_log_probs, _, _ = unpad_sequences(
                pad_len=pad_len,
                sequences=sequences,
                attention_mask=attention_mask,
                num_actions=num_actions,
                packed_seq_lens=packed_seq_lens,
                action_log_probs=action_log_probs,
                ring_attn_group=self.strategy.ring_attn_group,
            )

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )

        if self.args.use_kl_loss:
            if self.initial_model is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    experience.action_mask,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

            if not self.args.packing_samples:
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
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss * self.args.actor_loss_coef + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
        self.strategy.backward(loss, self.actor, self.actor_optim)
            
        sft_samples_this_step = 0 # Counter for SFT samples processed in this step on this rank
        if self.postfill_cot_loss:
            latest_postfill_cot_loss = None # Initialize here to ensure definition
            # Check if we should train SFT in this global step
            if global_steps % self.strategy.args.sft_steps == 0:
                from red_team.prompts import DEFENDER_INSTRUCTION_COT_PROMPT

                if self.sft_dataloader is None:
                    # === SFT Training using Experience Buffer (run once) ===
                    seq_len = len(experience.sequences)
                    sft_buffer = []
                    for idx in range(seq_len):
                        if infos['game_role'][idx] == 'defender' and infos['prompt_type'][idx] in ['vanilla_benign', 'generated_benign']:
                            sft_buffer.append((infos['prompts'][idx], infos['completion'][idx]))
                    
                    all_sft_buffer_len = self.strategy.all_gather(len(sft_buffer))
                    self.strategy.print(f"SFT Buffer Status - Lengths across ranks: {all_sft_buffer_len}, All non-empty: {all(all_sft_buffer_len)}")
                    
                    if all(all_sft_buffer_len):
                        sft_samples_this_step = len(sft_buffer) # Count samples from buffer
                        self.strategy.print(f"Start PTX SFT train step (from experience buffer)...")
                        sft_prompt_responses = []
                        prefixes = []
                        for prompt, completion in sft_buffer:
                            prompt_message =[{"role": "user", "content": DEFENDER_INSTRUCTION_COT_PROMPT.format(user_query=prompt)}]
                            prefix = self.tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
                            response = prefix + completion + self.tokenizer.eos_token
                            sft_prompt_responses.append(response)
                            prefixes.append(prefix)

                        if sft_prompt_responses:
                            tokenized_sequences, attention_masks, prefix_lengths, action_lengths = [], [], [], []
                            for i, (response, prefix) in enumerate(zip(sft_prompt_responses, prefixes)):
                                tokenized = self.tokenizer(response, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).to(torch.cuda.current_device())
                                input_ids = tokenized.input_ids[0]
                                attention_mask = tokenized.attention_mask[0]
                                prefix_token_len = len(self.tokenizer.encode(prefix, add_special_tokens=False))
                                action_len = input_ids.size(0) - prefix_token_len
                                tokenized_sequences.append(input_ids)
                                attention_masks.append(attention_mask)
                                prefix_lengths.append(prefix_token_len)
                                action_lengths.append(action_len)
                            
                            packed_seq_lens = [seq.numel() for seq in tokenized_sequences]
                            packed_input_ids = torch.cat(tokenized_sequences, dim=0).unsqueeze(0)
                            packed_attention_mask = torch.cat([torch.full_like(seq, i + 1) for i, seq in enumerate(tokenized_sequences)], dim=0).unsqueeze(0)
                            packed_labels = torch.where(packed_attention_mask.bool(), packed_input_ids, self.sftchat_loss_fn.IGNORE_INDEX)
                            
                            current_idx = 0
                            for seq_idx, (seq_len, prefix_len) in enumerate(zip(packed_seq_lens, prefix_lengths)):
                                packed_labels[0, current_idx:current_idx+prefix_len] = self.sftchat_loss_fn.IGNORE_INDEX
                                current_idx += seq_len
                            
                            sft_action_log_probs, sft_outputs = self.actor(
                                packed_input_ids, num_actions=action_lengths, attention_mask=packed_attention_mask,
                                return_output=True, packed_seq_lens=packed_seq_lens
                            )
                            
                            postfill_cot_loss_val = self.sftchat_loss_fn(sft_outputs["logits"], packed_labels)
                            latest_postfill_cot_loss = postfill_cot_loss_val.item() # Log the loss
                            
                            self.strategy.backward(self.args.postfill_cot_loss_coef * postfill_cot_loss_val, self.actor, self.actor_optim)
                else:
                    # === SFT Training using Dataloader (run sft_batches_per_step times) ===
                    assert self.args.packing_samples, "SFT training requires packing_samples to be enabled when using sft_dataloader"
                    self.strategy.print(f"Start PTX SFT train step (from dataloader, {self.strategy.args.sft_batches_per_step} batches)...")
                    
                    batch_sample_count = 0 # Track samples within the mini-batch loop
                    for i in range(self.strategy.args.sft_batches_per_step):
                        try:
                            data = next(self.sft_dataloader)
                        except StopIteration: # Should not happen with itertools.cycle, but safety first
                            self.strategy.print("Warning: SFT dataloader unexpectedly exhausted.")
                            break 
                            
                        inputs = data[1].to(torch.cuda.current_device())
                        attention_mask = data[2].to(torch.cuda.current_device())
                        packed_seq_lens = data[3]["input_length"]
                        batch_sample_count += len(packed_seq_lens) # Count samples from this batch
                        
                        label = torch.where(attention_mask.bool(), inputs, self.sftchat_loss_fn.IGNORE_INDEX)
                        
                        dump_labels = torch.full(label.size(), self.sftchat_loss_fn.IGNORE_INDEX).to(label.device)
                        if "response_ranges" in data[3] and data[3]["response_ranges"] is not None:
                            for response_ranges in data[3]["response_ranges"]:
                                if response_ranges:
                                    for response_range in response_ranges:
                                        dump_labels[0][response_range[0]:response_range[1]+1] = label[0][response_range[0]:response_range[1]+1]
                            label = dump_labels
                        else:
                            index = 0
                            for input_length, source_len in zip(packed_seq_lens, data[0]):
                                label[0][index:index+source_len] = self.sftchat_loss_fn.IGNORE_INDEX
                                index += input_length
                        
                        kwargs = {}
                        if self.strategy.ring_attn_group is not None:
                            kwargs = {"ring_attn_group": self.strategy.ring_attn_group, "packed_seq_lens": packed_seq_lens}
                        
                        output = self.actor(inputs, attention_mask=attention_mask, return_output=True, **kwargs)
                        sft_log_probs = output["logits"]

                        postfill_cot_loss_val = self.sftchat_loss_fn(sft_log_probs, label)
                        # Log the loss of the last mini-batch if multiple batches are run
                        if i == self.strategy.args.sft_batches_per_step - 1: 
                            latest_postfill_cot_loss = postfill_cot_loss_val.item() 

                        self.strategy.backward(self.args.postfill_cot_loss_coef * postfill_cot_loss_val, self.actor, self.actor_optim)
                    
                    sft_samples_this_step = batch_sample_count # Total samples for this step from dataloader

        # Aggregate SFT samples trained in this step across all ranks and update cumulative counter
        if sft_samples_this_step > 0:
            all_reduced_sft_samples = self.strategy.all_reduce(torch.tensor(sft_samples_this_step), op="sum")
            self.total_sft_samples_trained += all_reduced_sft_samples

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        if self.postfill_cot_loss > 0:
            torch.cuda.empty_cache()

        # status
        METRIC_TO_COMPUTE_MEAN = [
            'kl', 
            'reward', 
            'response_length', 
            'total_length', 
            'return'
        ]

        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        if self.postfill_cot_loss:
            # Log the SFT loss if it was computed in this step
            if latest_postfill_cot_loss is not None:
                 status["postfill_cot_loss"] = latest_postfill_cot_loss
            # Log the cumulative SFT samples trained
            status["total_sft_samples_trained"] = self.total_sft_samples_trained.item()

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            elif k in METRIC_TO_COMPUTE_MEAN:
                status[k] = v.mean().item()
            else:
                continue
        return status

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=count == num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _log_game_tables(self, global_step, table_data):
        from red_team.utils import convert_game_history_to_conversation
        from wildguard.wildguard.utils import PromptHarmfulness, ResponseHarmfulness, ResponseRefusal

        is_cot_enabled = not self.custom_configs.get('direct_chat_no_cot', False)

        # Get the full data for local logging
        game_history = table_data['game_history']
        prompt = table_data['prompts']
        try:
            defender_game_outcome = [{
                "harmlessness": GameOutcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE in dgo,
                "correct_refusal": GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in dgo,} 
                for dgo in table_data['defender_game_outcome']]
        except:
            defender_game_outcome = []
        
        # If CoT is disabled, game_history_with_cot might be the same as game_history
        game_history_with_cot = table_data.get('game_history_with_cot', game_history)
        wildguard_labels = table_data['wildguard_labels']
        
        # Sample subset for wandb logging
        if len(game_history) > self.strategy.args.wandb_max_log:
            indices = random.sample(range(len(game_history)), self.strategy.args.wandb_max_log)
            sampled_games = [game_history[i] for i in indices]
            sampled_games_cot = [game_history_with_cot[i] for i in indices] if is_cot_enabled else ["<--- CoT disabled --->"] * len(indices)
            sampled_labels = [wildguard_labels[i] for i in indices]
            sampled_prompt = [prompt[i] for i in indices]
            sampled_defender_game_outcome = [defender_game_outcome[i] for i in indices] if defender_game_outcome else []
        else:
            sampled_games = game_history
            sampled_games_cot = game_history_with_cot if is_cot_enabled else ["<--- CoT disabled --->"] * len(game_history)
            sampled_labels = wildguard_labels
            sampled_prompt = prompt
            sampled_defender_game_outcome = defender_game_outcome if defender_game_outcome else []

        # Add sampled data to wandb table
        new_table = wandb.Table(columns=['iter', "prompt", 'conversation', "cot_conversation", "wildguard_labels", "did_defender_win"])
        for game, game_with_cot, label, prompt, did_defender_win in zip(sampled_games, sampled_games_cot, sampled_labels, sampled_prompt, sampled_defender_game_outcome):
            wg_labels_list = {k:v for k,v in label.items() if k in ["prompt_harmfulness", "response_harmfulness", "response_refusal", "is_parsing_error"]}
            # convert None to string "None"
            wg_labels_list = {k: "None" if v is None else v for k, v in wg_labels_list.items()}
            
            # Convert game history to conversation format
            conversation = convert_game_history_to_conversation(game)
            
            # Handle CoT conversation based on whether CoT is enabled
            if is_cot_enabled:
                cot_conversation = convert_game_history_to_conversation(game_with_cot)
            else:
                cot_conversation = "<--- CoT disabled --->"
            
            new_table.add_data(
                global_step,
                prompt,
                conversation,
                cot_conversation,
                wg_labels_list,
                did_defender_win
            )
        
        self._wandb.log({"game_log": new_table}, commit=False)
        
        if self.strategy.args.wandb_table_csv_path is not None:
            os.makedirs(self.strategy.args.wandb_table_csv_path, exist_ok=True)

            try:
                csv_path = os.path.join(self.strategy.args.wandb_table_csv_path, 'game_log.csv')
                # Check if file exists to determine whether to write header
                write_header = not os.path.exists(csv_path)
                new_table.get_dataframe().to_csv(csv_path, encoding="utf-8", mode='a', index=False, header=write_header)
            except:
                pass

    def _log_response_tables(self, global_step, replay_buffer):
        is_cot_enabled = not self.custom_configs.get('direct_chat_no_cot', False)

        # Sample subset for wandb logging 
        attack_table = wandb.Table(columns=['iter', 'seed', 'seed_type', 'prompt', "response", "cot_format_violation", "outcomes", 'reward'])
        defense_table = wandb.Table(columns=['iter', 'seed', 'seed_type', 'prompt', "response", "cot_format_violation", "outcomes", "sft_completion", 'reward'])
        
        for item in random.sample(replay_buffer, min(self.strategy.args.wandb_max_log, len(replay_buffer))):
            if item.action_mask is not None:
                p = item.action_mask.sum()
            else:
                p = int(item.info['response_length'])

            game_outcomes = item.info.get("game_outcomes")
            game_role = item.info.get("game_role")
            
            # Handle cot_format_violation based on whether CoT is enabled
            if is_cot_enabled:
                cot_format_violation = True if item.info.get("cot_format_violation") else False
            else:
                cot_format_violation = "n/a"  # Not applicable when CoT is disabled
            
            seed = item.info.get("prompts", "")
            seed_type = item.info.get("prompt_type", "")
            prompt = self.tokenizer.decode(item.sequences[:-p], skip_special_tokens=False)
            response = self.tokenizer.decode(item.sequences[-p:], skip_special_tokens=False)
            sft_completion = item.info.get('completion', '')
            
            # Handle thinking and answer differently based on whether CoT is enabled
            if is_cot_enabled and 'text_cot_and_answer' in item.info:
                think = item.info['text_cot_and_answer'][0] if item.info['text_cot_and_answer'][0] != "" else "<--- redacted --->"
                answer = item.info['text_cot_and_answer'][1] if item.info['text_cot_and_answer'][1] != "" else "<--- redacted --->"
            else:
                # When CoT is disabled, there is no separate thinking and answer
                think = "<--- CoT disabled --->"
                answer = "<--- CoT disabled --->"  # Use the full response as the answer
            
            reward = item.info.get("reward")
            responses = {"original": response, "thinking": think, "answer": answer}

            if game_role == 'attacker':
                outcomes = {
                    "is_faithful_revision": GameOutcome.ATTACKER_FAITHFUL_REVISION in game_outcomes,
                    "did_attack_succeed": GameOutcome.ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST in game_outcomes or GameOutcome.ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST in game_outcomes,
                    "attack_correct_refused": GameOutcome.ATTACKER_WIN_BENIGN_REFUSAL in game_outcomes or GameOutcome.ATTACKER_WIN_HARMFUL_COMPLIANCE in game_outcomes,
                }
                
                attack_table.add_data(
                    global_step,
                    seed,
                    seed_type,
                    prompt,
                    responses,
                    cot_format_violation,
                    outcomes,
                    reward
                )
        
            elif game_role == 'defender':
                outcomes = {
                    "is_response_harmless": GameOutcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE in game_outcomes,
                    "correct_refusal": GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in game_outcomes
                }
                
                defense_table.add_data(
                    global_step,
                    seed,
                    seed_type,
                    prompt,
                    responses,
                    cot_format_violation,
                    outcomes,
                    sft_completion,
                    reward
                )
        self._wandb.log({"attacker_log": attack_table}, commit=False)
        self._wandb.log({"defender_log": defense_table}, commit=False)

        if self.strategy.args.wandb_table_csv_path is not None:
            os.makedirs(self.strategy.args.wandb_table_csv_path, exist_ok=True)

            try:
                csv_path = os.path.join(self.strategy.args.wandb_table_csv_path, 'attacker_response_log.csv')
                # Check if file exists to determine whether to write header
                write_header = not os.path.exists(csv_path)
                attack_table.get_dataframe().to_csv(csv_path, encoding="utf-8", mode='a', index=False, header=write_header)
            except:
                pass
            
            try:
                csv_path = os.path.join(self.strategy.args.wandb_table_csv_path, 'defender_response_log.csv')
                # Check if file exists to determine whether to write header
                write_header = not os.path.exists(csv_path)
                defense_table.get_dataframe().to_csv(csv_path, encoding="utf-8", mode='a', index=False, header=write_header)
            except:
                pass

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}, **kwargs):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():

                # Define prefixes that don't need "train/" prefix
                direct_log_prefixes = ["game/", "debug/", "defender/", "attacker/", "length/", "bleu/", "sbert/", "eval/"]
                
                logs = {"train/global_step": global_step}
                for k, v in logs_dict.items():
                    # Check if key starts with any of the direct log prefixes
                    if any(k.startswith(prefix) for prefix in direct_log_prefixes):
                        logs[k] = v
                    else:
                        logs[f"train/{k}"] = v
                
                # Add performance stats
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})

                # Game visualization
                if 'game_history' in kwargs and kwargs['game_history'] is not None and (
                    self.strategy.args.wandb_table_log_interval is not None and 
                    global_step % self.strategy.args.wandb_table_log_interval == 0
                ):
                    self._log_game_tables(global_step, kwargs['game_history'])
                    
                # Response visualization
                if 'replay_buffer' in kwargs and kwargs['replay_buffer'] is not None and (
                    self.strategy.args.wandb_table_log_interval is not None and 
                    global_step % self.strategy.args.wandb_table_log_interval == 0
                ):
                    self._log_response_tables(global_step, kwargs['replay_buffer'])

                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # diversity evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            pass        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
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

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)


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

        if not self.strategy.args.custom_configs.get("no_attacker_turn", False) and not self.strategy.args.custom_configs.get("no_defender_turn", False):
            max_steps *= 1.5 # TODO: we are assuming there will be 1.5x more update because the attacker generation is an additional 50% of training experiences.
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
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        # self.prompts_dataset = PromptDataset(
        #     prompts_data, self.tokenizer, strategy, input_template=args.input_template
        # )
        self.prompts_dataset = RedTeamGamePromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
            True,
            True,
        )

        if args.sft_data:
            sft_strategy = deepcopy(strategy)
            sft_strategy.args.apply_chat_template = True
            sft_strategy.args.prompt_input_template = DEFENDER_INSTRUCTION_COT_PROMPT
            
            sft_data = blending_datasets(
                args.sft_data,
                args.sft_data_probs,
                sft_strategy,
                args.seed,
                return_eval=False,
                train_split=args.sft_split,
            )
            sft_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            sft_dataset = SFTDataset(
                sft_data.select(
                    range(
                        min(
                            len(sft_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                sft_max_len,
                sft_strategy,
                pretrain_mode=False,
                prompt_input_template=DEFENDER_INSTRUCTION_COT_PROMPT,
            )
            # Create rank-aware dataloader to ensure different actors get different data batches
            self.sft_dataloader = itertools.cycle(
                iter(
                    sft_strategy.setup_dataloader(
                        sft_dataset,
                        batch_size = args.micro_train_batch_size // 2,
                        pin_memory=True,
                        shuffle=True,
                        collate_fn=sft_dataset.packing_collate_fn,

                    )
                )
            )
        else:
            self.sft_dataloader = None

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
                    range(
                        min(
                            len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
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
        
        # prepare holdout dataset
        if args.eval_data:            
            holdout_dataset = blending_datasets(
                args.eval_data,
                "1.0",
                strategy,
                args.seed,
                max_count=100*strategy.world_size,
                return_eval=False,
            )
            self.holdout_dataset = RedTeamGamePromptDataset(
                holdout_dataset, self.tokenizer, strategy, input_template=args.input_template, mark_to_generate=False
            )
            self.holdout_dataloader = strategy.setup_dataloader(
                self.holdout_dataset,
                len(self.holdout_dataset) // (strategy.world_size // strategy.ring_attn_size),
                True,
                True,
            )
        else:
            self.holdout_dataloader = None

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
        **kwargs,
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
            defender_vllm_engines=kwargs.get("defender_vllm_engines", None),
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
            # for multi-turn self-play
            custom_configs=args.custom_configs,
            # fro GPT generation
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

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.pretrain_dataloader,
            self.holdout_dataloader,
            self.sft_dataloader,
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
