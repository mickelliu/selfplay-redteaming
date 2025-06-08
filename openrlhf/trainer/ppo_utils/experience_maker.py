import re
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray
from red_team.utils import cot_format_check_and_extract, get_cot_formatting_reward, get_redteaming_game_reward_general_sum, get_redteaming_game_reward_zero_sum

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    game_rewards: Optional[torch.Tensor]
    additional_infos: Optional[dict]
    prompts: Optional[list[str]]
    labels: list[str]
    pad_len: Optional[int]


class BaseExperienceMaker(ABC):
    """
    Base experience maker that only handles initialization.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = {}
        self.advantage_estimator = strategy.args.advantage_estimator
        self.ring_rank0_group = None

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


class RemoteExperienceMaker(BaseExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

        self.game_history = None
        if self.strategy.is_rank_0():
            self.lost_samples = 0

        self.defender_vllm_engines = kwargs.get("defender_vllm_engines", None)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences, **generate_kwargs)

        # send experience to critic
        if self.critic is not None:
            for experience in experiences:
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples_list: List[Samples]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        if dist.get_rank() == 0:
            logger.info(f"🚀 Starting experience making with {len(samples_list) * dist.get_world_size()} batches")

        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()
        experiences = []

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        num_actions_list = [s.num_actions for s in samples_list]
        packed_seq_lens_list = [s.packed_seq_lens for s in samples_list]
        prompts_list = [p for s in samples_list for p in s.prompts]
        labels_list = [l for s in samples_list for l in s.labels]

        # Move data to CPU for remote processing
        sequences_cpu_list = [seq.to("cpu") for seq in sequences_list]
        attention_mask_cpu_list = [mask.to("cpu") for mask in attention_mask_list]

        # Batch call initial model
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=sequences_cpu_list,
                num_actions=num_actions_list,
                attention_mask=attention_mask_cpu_list,
                logps_allgather=[True] * len(samples_list),
                packed_seq_lens=packed_seq_lens_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(samples_list))

        # Batch call critic model
        if self.critic is not None:
            value_ref = self.critic.forward_batch.remote(
                sequences=sequences_cpu_list,
                num_actions=num_actions_list,
                attention_mask=attention_mask_cpu_list,
                packed_seq_lens=packed_seq_lens_list,
            )
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put([None] * len(samples_list))

        # Batch call reward model
        r_refs = []

        # if reward is provided, use it as the rewards
        if hasattr(samples_list[0], "game_rewards") and samples_list[0].game_rewards is not None:
            # support remote RM API with ray
            assert self.remote_rm_url is not None, "Remote RM URL is not set"
            rewards_list = [samples.game_rewards for samples in samples_list]
        else:
            if not self.remote_rm_url:
                for rm in self.reward_model:
                    r_refs.append(
                        rm.forward_batch.remote(
                            sequences=sequences_cpu_list,
                            attention_mask=attention_mask_cpu_list,
                            packed_seq_lens=packed_seq_lens_list,
                            pad_sequence=[True] * len(samples_list),
                        )
                    )
            else:
                if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                    queries_list = []
                    for i, (seq, packed_lens) in enumerate(zip(sequences_cpu_list, packed_seq_lens_list)):
                        if not self.packing_samples:
                            queries = self.tokenizer.batch_decode(seq, skip_special_tokens=False)
                        else:
                            sequences_list = []
                            offset = 0
                            tokens_list = seq.tolist()[0]
                            for length in packed_lens:
                                sequences_list.append(tokens_list[offset : offset + length])
                                offset += length
                            queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                        queries_list.extend(queries)

                    if self.custom_reward_func:
                        r = self.custom_reward_func.remote(queries_list, prompts_list, labels_list)
                    else:
                        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
                        rm = self.remote_rm_url[rank % len(self.remote_rm_url)]
                        r = remote_rm_fn_ray.remote(rm, queries=queries_list, prompts=prompts_list, labels=labels_list)
                    r_refs.append(r)
                else:
                    r_refs.append(ray.put([None] * len(samples_list)))
                
                rewards_list = ray.get(r_refs)

            if args.colocate_all_models and not self.remote_rm_url:
                ray.get(r_refs)
                ray.get([self.reward_model[0].empty_cache.remote()])

        # Batch call actor model
        action_log_probs_list = []
        for seq, num_acts, attn_mask, packed_lens in zip(
            sequences_cpu_list, num_actions_list, attention_mask_cpu_list, packed_seq_lens_list
        ):
            action_log_probs = self.actor(
                seq.to(device),
                num_acts,
                attn_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=True,
                packed_seq_lens=packed_lens,
            )
            action_log_probs_list.append(action_log_probs)

        actor_value_rm_time = time.time() - start_time

        # Wait for all remote calls to complete
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs_list, value_list = ref_values[0], ref_values[1]
        if self.remote_rm_url is not None and isinstance(rewards_list, torch.Tensor):
            rewards_list = rewards_list.chunk(len(samples_list))

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value, rewards) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list, rewards_list)
        ):
            if base_action_log_probs is not None:
                base_action_log_probs = base_action_log_probs.to(device)
            if value is not None:
                value = value.to(device)

            # Broadcast rewards to all ring attention ranks when using remote RM
            rewards = [rewards]
            if self.remote_rm_url and self.strategy.ring_attn_group is not None:
                if self.strategy.ring_attn_rank == 0:
                    dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
                else:
                    dist.broadcast_object_list(
                        rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                    )
            r = rewards[0].to(device)

            if (self.initial_model is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    action_mask=samples.action_mask,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

            sequences = samples.sequences
            attention_mask = samples.attention_mask
            if not self.packing_samples:
                kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            else:
                num_actions = samples.num_actions
                packed_seq_lens = samples.packed_seq_lens
                if self.strategy.ring_attn_group is not None:
                    assert samples.pad_len is not None
                    sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                        pad_len=samples.pad_len,
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        action_log_probs=action_log_probs,
                        values=value,
                        kl=kl,
                    )
                # Convert tensor into list of tensors for easier manipulation within dataset
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None
                action_log_probs = unpacking_samples(action_log_probs, num_actions)
                if value is not None:
                    value = unpacking_samples(value, num_actions)
                if base_action_log_probs is not None:
                    base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

            if not args.use_kl_loss:
                base_action_log_probs = None

            info = {
                "kl": kl_mean,
                "reward": r,
                "response_length": samples.response_length,
                "total_length": samples.total_length,
                "num_actions": samples.num_actions,
            }
            # Merge additional_infos into info
            info.update(samples.additional_infos)
            # assert len(info["game_outcomes"]) == len(info["game_role"]) == len(info["game_rewards"]), "Game outcomes, roles, and rewards must have the same length"

            if self.strategy.args.perf:
                self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
                self.perf_stats["wait_time"] += wait_time

            experience = Experience(
                sequences,
                action_log_probs,
                base_action_log_probs,
                value,
                None,
                None,
                attention_mask,
                samples.action_mask,
                info,
                kl,
            )

            experiences.append(experience)

        self.actor.train()  # Reset model state

        end_time = time.time()
        duration = end_time - start_time
        if dist.get_rank() == 0:
            time_str = str(timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Experience making completed in {time_str}")
        return experiences

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # get rewards from experiences
        rewards = [experience.info["reward"] for experience in experiences]

        # reward shaping
        if args.advantage_estimator == "rloo":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat(rewards).reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    kwargs["gamma"],
                    kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if kwargs["gamma"] != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    if dist.get_rank() == 0:
                        logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    kwargs["gamma"] = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return self._generate_with_hf(all_prompts, all_labels, **generate_kwargs)
        elif self.strategy.args.custom_configs is not None:
            return self.generate_multi_turn_with_reward_model(all_prompts, all_labels, self.strategy.args.custom_configs, **generate_kwargs)
        # vLLM generation
        return self._generate_vllm(all_prompts, all_labels, **generate_kwargs)

    @torch.no_grad()
    def _generate_with_hf(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

    def generate_multi_turn_with_reward_model(
        self,
        all_prompts: List[Tuple[List[str], List[str], Dict[str, str]]],  # (prompts, data_types, completions)
        all_labels: List[Dict[str, Any]],
        custom_configs: Dict[str, Any],
        **kwargs
    ) -> Tuple[List[Samples], List[Dict[str, Any]]]:
        """
        Generate multi-turn dialogue samples between attacker and defender roles using DialogueGameManager.
        """
        from openrlhf.trainer.ppo_utils.language_game import DialogueGameManager
        
        # Initialize the dialogue game manager
        game_manager = DialogueGameManager(
            tokenizer=self.tokenizer,
            remote_rm_url=self.remote_rm_url,
            strategy=self.strategy,
            custom_configs=custom_configs
        )
        
        # Create generator functions for attacker and defender
        def attacker_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):
            return self._generate_vllm(self.vllm_engines, batch_chat_messages, all_labels, **gen_kwargs)
        
        # If no_defender_turn is enabled, use defender_vllm_engines for defender_llm_generator
        if custom_configs.get("no_defender_turn", False) and self.defender_vllm_engines is not None:            
            def defender_llm_generator(batch_chat_messages, all_labels, **gen_kwargs):
                return self._generate_vllm(self.defender_vllm_engines, batch_chat_messages, all_labels, **gen_kwargs)
        else:
            # If no_defender_turn is not enabled or defender_vllm_engines is not available, 
            # use the same generator for both
            defender_llm_generator = attacker_llm_generator
        
        # Initialize games from prompts
        game_manager.initialize_games(
            prompts=all_prompts[0],
            completions=all_prompts[1],
            data_types=all_labels,
        )
        
        # Play the dialogue games with separate generators for attacker and defender
        active_games = game_manager.play_games(
            attacker_llm_generator=attacker_llm_generator, 
            defender_llm_generator=defender_llm_generator, 
            **kwargs
        )

        # Evaluate game outcomes
        batch_labels_dict = game_manager.evaluate_game_outcomes()
        
        # Filter and compute rewards
        attacker_outputs, attacker_turn_states, defender_outputs, defender_turn_states, batch_labels_dict = game_manager.filter_and_compute_rewards(
            batch_labels_dict
        )

        filtered_outputs = attacker_outputs + defender_outputs
        filtered_turn_states = attacker_turn_states + defender_turn_states
        
        assert len(filtered_outputs) == len(filtered_turn_states), "Filtered outputs and turn states must have the same length"

        # Handle buffer sizes for distributed training
        if self.strategy.stage == 3:
            micro_rollout_batch_size = self.strategy.args.micro_rollout_batch_size
            # Allreduce to cut to the same length to make sure make_experience doesn't get stuck
            import math
            local_n_batches = math.ceil(len(filtered_outputs) / micro_rollout_batch_size)
            local_len = len(filtered_outputs)
            
            # Gather lengths from all processes
            all_n_batches = self.strategy.all_gather(local_n_batches)
            min_n_batches = int(min(all_n_batches))
            all_len = self.strategy.all_gather(local_len)
            min_len = int(min(all_len))
            
            # Sanity check
            assert min_n_batches != 0, "No samples in at least one replay buffer"
            
            # If current buffer is longer than min_length, randomly sample min_length items
            if local_n_batches > min_n_batches:
                # Find all vanilla_benign samples
                vanilla_benign_indices = [i for i, turn_state in enumerate(filtered_turn_states) 
                                        if turn_state.get('prompt_type') == 'vanilla_benign']
                
                # Calculate how many samples we need in total
                required_samples = min_n_batches * micro_rollout_batch_size
                
                if len(vanilla_benign_indices) >= required_samples:
                    # If we have more vanilla_benign samples than needed, randomly select from them
                    selected_indices = torch.tensor(vanilla_benign_indices)[torch.randperm(len(vanilla_benign_indices))[:required_samples]]
                else:
                    # Take all vanilla_benign samples
                    selected_indices = vanilla_benign_indices.copy()
                    
                    # Get indices of non-vanilla_benign samples
                    other_indices = [i for i in range(len(filtered_turn_states)) if i not in vanilla_benign_indices]
                    
                    # Randomly select additional samples to fill the quota
                    remaining_needed = required_samples - len(vanilla_benign_indices)
                    if remaining_needed > 0 and other_indices:
                        additional_indices = torch.tensor(other_indices)[torch.randperm(len(other_indices))[:remaining_needed]]
                        selected_indices.extend(additional_indices.tolist())
                
                # Select the samples based on our chosen indices
                selected_indices = sorted(selected_indices)  # Sort to maintain original order
                filtered_outputs = [filtered_outputs[i] for i in selected_indices]
                filtered_turn_states = [filtered_turn_states[i] for i in selected_indices]

                # Log statistics if rank 0
                if self.strategy.is_rank_0():
                    vanilla_benign_count = len([i for i in selected_indices if i in vanilla_benign_indices])
                    self.strategy.print(f"Kept {vanilla_benign_count}/{len(vanilla_benign_indices)} vanilla_benign samples " 
                                       f"out of {len(selected_indices)} total samples after filtering")

            data_loss = sum(all_len) - sum(self.strategy.all_gather(len(filtered_outputs)))
            if self.strategy.is_rank_0():
                self.strategy.print("=" * 50)
                self.strategy.print(f"📊 Experience Maker Synchronization...")
                self.strategy.print(f"📈 Buffer lengths across ranks: {all_len}, Minimum buffer length: {min_len}")
                self.strategy.print(f"📊 n_batches across ranks: {all_n_batches}, Minimum n_batches: {min_n_batches}")
                # compute data loss across ranks
                if data_loss == 0:
                    self.strategy.print("No data need to sync, skipping...")
                self.strategy.print("=" * 50)
                self.lost_samples += data_loss

        # Post process sequences for experience replay
        samples_list = self._post_process_sequences(filtered_outputs, filtered_turn_states, None, None, **kwargs)
        
        # Helper function to safely access nested dictionary values
        def safe_get_game_outcome(game_dict):
            try:
                if 'processed_output_history' not in game_dict or not game_dict['processed_output_history']:
                    return None
                if 'game_states' not in game_dict['processed_output_history'][-1]:
                    return None
                return game_dict['processed_output_history'][-1]['game_states'].get('game_outcomes', None)
            except (IndexError, KeyError, TypeError):
                return None
        
        # Prepare data for logging
        self.game_history = {
            "game_history": [active_games[idx]["history"] for idx in active_games.keys()],
            "game_history_with_cot": [active_games[idx]["raw_history"] for idx in active_games.keys()],
            "defender_game_outcome": [safe_get_game_outcome(active_games[idx]) for idx in active_games.keys()],
            "prompts": [active_games[idx]["prompts"] for idx in active_games.keys()],
            "wildguard_labels": [batch_labels_dict[idx] for idx in active_games.keys()],
        }
        
        return samples_list

    def _generate_vllm(self, vllm_engines, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(vllm_engines) <= world_size:
            llms = [vllm_engines[rank % len(vllm_engines)]]
        else:
            llms = vllm_engines[rank::world_size]
        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Waiting for all requests to be sent
        if self.strategy.ring_attn_group is not None:
            if self.ring_rank0_group is None:
                world_size = dist.get_world_size()
                ring_rank0 = [
                    i * self.strategy.ring_attn_size for i in range(world_size // self.strategy.ring_attn_size)
                ]
                self.ring_rank0_group = dist.new_group(ranks=ring_rank0)
            dist.barrier(group=self.ring_rank0_group)
        else:
            dist.barrier()
        torch.cuda.synchronize()

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])
        
        # Add simple assertions to verify alignment
        assert len(all_outputs) == len(all_prompt_token_ids), (
            f"Output count mismatch: expected {len(all_prompt_token_ids)}, "
            f"got {len(all_outputs)}"
        )
        
        # # Check first and last items at minimum
        # check_indices = [0, -1]  # Add more indices for stricter verification
        # checkk all outputs
        for idx in range(len(all_outputs)):
            assert torch.equal(
                torch.tensor(all_outputs[idx].prompt_token_ids),
                torch.tensor(all_prompt_token_ids[idx])
            ), f"Prompt mismatch at index {idx}"
        
        return all_outputs

    def _post_process_sequences(self, all_outputs, all_gamedata, all_prompts, all_labels, **kwargs):
        """
        Process generated sequences into batched Samples objects.
        
        Args:
            all_outputs: Model outputs containing generated token sequences
            all_gamedata: Game metadata for each generated sequence
            all_prompts: Original prompts used for generation
            all_labels: Labels for each prompt
            
        Returns:
            List of Samples objects ready for experience collection
        """
        args = self.strategy.args
        samples_list = []
        
        # Define all fields to extract from gamedata
        gamedata_fields = [
            'reward', 'game_outcomes', 'game_role', 'prompt_type', 'cot_format_violation',
            'text_cot_and_answer', 'length_cot_and_answer', 'prompts', 'completion', "is_generated_attack"
        ]
        
        # Extract all gamedata fields in one pass
        extracted_data = {}
        for field in gamedata_fields:
            extracted_data[field] = [gamedata.get(field) for gamedata in all_gamedata]
        
        # Process in batches according to micro_rollout_batch_size
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            batch_end = min(i + args.micro_rollout_batch_size, len(all_outputs))
            
            # Extract current batch
            outputs = all_outputs[i:batch_end]
            prompts = [data['prompts'] for data in all_gamedata[i:batch_end]]
            labels = [data['prompt_type'] for data in all_gamedata[i:batch_end]]
            # prompts = all_prompts[i:batch_end]
            # labels = all_labels[i:batch_end]
            
            # Extract gamedata for current batch
            batch_gamedata = {}
            for field in gamedata_fields:
                batch_gamedata[field] = extracted_data[field][i:batch_end]
            
            # Handle different processing approaches
            if not self.packing_samples:
                raise NotImplementedError("packing_samples=False is not supported for the current implementation")
            else:
                # Prepare for packed sequences
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                
                # Process each output and pack them together
                for j, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    
                    # Track sequence lengths and content
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([j + 1] * (input_len + output_len))
                    num_actions.append(max(1, output_len))
                
                # Handle padding for ring attention if needed
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )
                
                # Convert to tensors and move to GPU
                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                game_rewards = torch.tensor(batch_gamedata['reward'], device="cuda", dtype=torch.float)
                
                # Create additional info dictionary, excluding reward which is handled separately
                additional_infos = {field: batch_gamedata[field] for field in batch_gamedata if field != 'reward'}
                
                # Create and append Samples object
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        game_rewards=game_rewards,
                        additional_infos=additional_infos,
                        labels=labels,
                        pad_len=pad_len,
                    )
                )
        
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
