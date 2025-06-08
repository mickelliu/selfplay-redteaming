import random
import time
from abc import ABC
from dataclasses import dataclass
from typing import Counter, List, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np
import math

from openrlhf.utils.wanli import RobertaLargeWanli
from red_team import GameOutcome


from .experience_maker import Experience
from openrlhf.utils.self_bleu import compute_self_bleu
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from openrlhf.utils.sbert import compute_batch_sbert_similarity



@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        if isinstance(v, torch.Tensor):
            vals = torch.unbind(v)
        elif isinstance(v, list):
            vals = v
        else:
            raise TypeError(f"Unsupported type for info value: {type(v)}")
            
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                vv = vv.tolist() if vv.numel() > 1 else vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = [item.info[key] for item in items]
        # Only convert to tensor if the values are numeric
        if isinstance(vals[0], (int, float)):
            vals = torch.tensor(vals)
        kwargs["info"][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False, custom_configs: Optional[Dict] = None
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.target_device)
        self.nli_model = RobertaLargeWanli(device=self.target_device)
        self.custom_configs = custom_configs
        
    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def normalize(self, strategy, attribute: str, role=None, divide_by_std: bool = True) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        role_items = []
        
        # Filter items by role if specified, otherwise use all items
        if role is not None:
            for item in self:
                if item.info['game_role'] == role:
                    role_items.append(item)
        else:
            role_items = list(self)
        
        # If no items for the specified role, return early
        if not role_items:
            return
        
        # Collect advantages and action masks
        for item in role_items:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        if divide_by_std:
            std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
            all_std = strategy.all_reduce(std, "sum")
            rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()
        else:
            rstd = 1

        # Apply normalization only to the relevant items
        for i, item in enumerate(role_items):
            setattr(item, attribute, (items[i] - mean) * rstd)
    
    def compute_game_metrics(self, strategy) -> Dict[str, float]:
        from red_team import GameOutcome        
        no_attacker_turn = strategy.args.custom_configs.get('no_attacker_turn', False)
        no_defender_turn = strategy.args.custom_configs.get('no_defender_turn', False)
        assert not no_attacker_turn or not no_defender_turn, "no_attacker_turn and no_defender_turn cannot both be True"
        game_status = {}

        # Separate items by role
        attacker_items = [item for item in self.items if item.info['game_role'] == 'attacker' and item.info['prompt_type'] in ('generated_harmful', 'generated_benign')]
        n_generated_attacks = len(attacker_items)
        
        defender_items = [item for item in self.items if item.info['game_role'] == 'defender']
        n_defender_samples = len(defender_items)
        
        benign_prompts = [item for item in defender_items if item.info['prompt_type'] in ('vanilla_benign', 'generated_benign')]
        n_benign_samples = len(benign_prompts)

        # Compute attacker metrics if there are any attacker samples
        if n_generated_attacks > 0 and not no_attacker_turn:
            attacker_metrics = self._compute_attacker_metrics(
                attacker_items, 
                n_generated_attacks,
                strategy
            )
            game_status.update(attacker_metrics)

        # Compute defender metrics
        if n_defender_samples > 0 and not no_defender_turn:
            defender_metrics = self._compute_defender_metrics(
                defender_items,
                n_defender_samples,
                n_benign_samples, 
                strategy
            )
            game_status.update(defender_metrics)

        # Compute general game metrics
        n_tie_wgfailed = len([item for item in self.items if GameOutcome.TIE in item.info['game_outcomes']])
        tie_rate = n_tie_wgfailed / max(len(self.items), 1)
        
        general_metrics = {
            'game/tie': strategy.all_reduce(tie_rate, "mean"),
            'debug/n_samples': len(self.items),
            'debug/n_attacker_samples': n_generated_attacks,
            'debug/n_defender_samples': n_defender_samples
        }
        game_status.update(general_metrics)

        return game_status

    def _compute_attacker_metrics(self, attacker_items, n_generated_attacks, strategy):
        is_cot_enabled = not self.custom_configs.get('direct_chat_no_cot', False)
        
        # Rewards
        attacker_rewards = [item.info['reward'] for item in attacker_items]
        avg_attacker_reward = sum(attacker_rewards) / max(len(attacker_rewards), 1)

        # Success metrics
        n_faithful_revision = sum(GameOutcome.ATTACKER_FAITHFUL_REVISION in item.info['game_outcomes'] for item in attacker_items)
        n_win_compliance = sum((GameOutcome.ATTACKER_WIN_HARMFUL_COMPLIANCE in item.info['game_outcomes'] or 
                               GameOutcome.ATTACKER_WIN_BENIGN_REFUSAL in item.info['game_outcomes']) 
                               for item in attacker_items)

        # Additional metrics
        n_successful_benign_request = sum(GameOutcome.ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST in item.info['game_outcomes'] and item.info['prompt_type'] == "generated_benign" for item in attacker_items)
        n_successful_harmful_request = sum(GameOutcome.ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST in item.info['game_outcomes'] and item.info['prompt_type'] == "generated_harmful" for item in attacker_items)
        n_win_harmful_compliance = sum(GameOutcome.ATTACKER_WIN_HARMFUL_COMPLIANCE in item.info['game_outcomes'] and item.info['prompt_type'] == "generated_harmful" for item in attacker_items)
        n_win_benign_refusal = sum(GameOutcome.ATTACKER_WIN_BENIGN_REFUSAL in item.info['game_outcomes'] and item.info['prompt_type'] == "generated_benign" for item in attacker_items)
        n_harmful_revision_faithfulness = sum(GameOutcome.ATTACKER_FAITHFUL_REVISION in item.info['game_outcomes'] and item.info['prompt_type'] == 'generated_harmful' for item in attacker_items)
        n_benign_revision_faithfulness = sum(GameOutcome.ATTACKER_FAITHFUL_REVISION in item.info['game_outcomes'] and item.info['prompt_type'] == 'generated_benign' for item in attacker_items)

        n_generated_harmful_total = sum(item.info['prompt_type'] == 'generated_harmful' for item in attacker_items)
        n_generated_benign_total = sum(item.info['prompt_type'] == 'generated_benign' for item in attacker_items)

        # Calculate rates
        compliance_rate = n_win_compliance / max(n_generated_attacks, 1)
        revised_faithfulness_rate = n_faithful_revision / max(n_generated_attacks, 1)
        successful_rate = (n_successful_benign_request + n_successful_harmful_request) / max(n_generated_attacks, 1)

        successful_harmful_rate = n_successful_harmful_request / max(n_generated_harmful_total, 1)
        successful_benign_rate = n_successful_benign_request / max(n_generated_benign_total, 1)
        win_harmful_compliance_rate = n_win_harmful_compliance / max(n_generated_harmful_total, 1)
        win_benign_refusal_rate = n_win_benign_refusal / max(n_generated_benign_total, 1)
        harmful_revision_faithfulness_rate = n_harmful_revision_faithfulness / max(n_generated_harmful_total, 1)
        benign_revision_faithfulness_rate = n_benign_revision_faithfulness / max(n_generated_benign_total, 1)

        metrics = {
            'attacker/avg_reward': strategy.all_reduce(avg_attacker_reward, "mean"),
            'attacker/request_success_rate': strategy.all_reduce(successful_rate, "mean"), # successful = benign and harmful both leads to harmful reaction
            'attacker/request_correct_refusal_rate': strategy.all_reduce(compliance_rate, "mean"), # successful = benign leads to refusal, harmful leads to compliance
            'attacker/revision_faithfulness_rate': strategy.all_reduce(revised_faithfulness_rate, "mean"),

            'attacker/info/successful_harmful_request_rate': strategy.all_reduce(successful_harmful_rate, "mean"), # successful = harmful leads to compliance
            'attacker/info/successful_benign_request_rate': strategy.all_reduce(successful_benign_rate, "mean"), # successful = benign leads to refusal
            'attacker/info/win_harmful_compliance_rate': strategy.all_reduce(win_harmful_compliance_rate, "mean"),
            'attacker/info/win_benign_refusal_rate': strategy.all_reduce(win_benign_refusal_rate, "mean"),
            'attacker/info/harmful_revision_faithfulness_rate': strategy.all_reduce(harmful_revision_faithfulness_rate, "mean"),
            'attacker/info/benign_revision_faithfulness_rate': strategy.all_reduce(benign_revision_faithfulness_rate, "mean"),
        
            'attacker/debug/n_generated_harmful': n_generated_harmful_total,
            'attacker/debug/n_generated_benign': n_generated_benign_total,
        }
        
        # Only include CoT format violation metrics if CoT is enabled
        if is_cot_enabled:
            attacker_cot_violations = sum(item.info['cot_format_violation'] for item in attacker_items)
            attacker_cot_rate = attacker_cot_violations / max(n_generated_attacks, 1)
            metrics['attacker/cot_format_violation'] = strategy.all_reduce(attacker_cot_rate, "mean")

        return {k: v for k, v in metrics.items() if v is not None and not math.isnan(v)}

    def _compute_defender_metrics(self, defender_items, n_defender_samples, n_benign_samples, strategy):
        """Compute defender-specific metrics from the replay buffer.
        
        Args:
            defender_items: List of BufferItems containing defender experiences
            n_defender_samples: Total number of defender samples
            n_benign_samples: Total number of benign samples
            strategy: Training strategy object for distributed training
            
        Returns:
            Dictionary of defender metrics
        """
        is_cot_enabled = not self.custom_configs.get('direct_chat_no_cot', False)
        
        # Rewards
        defender_rewards = [item.info['reward'] for item in defender_items]
        avg_defender_reward = sum(defender_rewards) / max(len(defender_rewards), 1)

        # Success metrics
        n_harmless_responses = sum(GameOutcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE in item.info['game_outcomes'] for item in defender_items)
        n_correct_refusal = sum(GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in item.info['game_outcomes'] for item in defender_items)

        # Track refusal accuracy by prompt source
        n_generated_total = sum(item.info['prompt_type'] in ('generated_harmful', 'generated_benign') for item in defender_items)
        n_vanilla_total = sum(item.info['prompt_type'] in ('vanilla_benign', "vanilla_harmful") for item in defender_items)
        n_generated_correct_refusal = sum(GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in item.info['game_outcomes'] and item.info['prompt_type'] in ('generated_harmful', 'generated_benign') for item in defender_items)
        n_vanilla_correct_refusal = sum(GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in item.info['game_outcomes'] and item.info['prompt_type'] in ('vanilla_benign', "vanilla_harmful") for item in defender_items)


        # Track finegrained category metrics
        n_generated_harmful_total = sum(item.info['prompt_type'] == 'generated_harmful' for item in defender_items)
        n_generated_benign_total = sum(item.info['prompt_type'] == 'generated_benign' for item in defender_items)
        n_vanilla_benign_total = sum(item.info['prompt_type'] == 'vanilla_benign' for item in defender_items)
        n_vanilla_harmful_total = sum(item.info['prompt_type'] == 'vanilla_harmful' for item in defender_items)

        n_generated_harmful_correct_harmfulness = sum(GameOutcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE in item.info['game_outcomes'] and item.info['prompt_type'] == 'generated_harmful' for item in defender_items)
        n_generated_benign_correct_harmfulness = sum(GameOutcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE in item.info['game_outcomes'] and item.info['prompt_type'] == 'generated_benign' for item in defender_items)
        n_vanilla_benign_correct_harmfulness = sum(GameOutcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE in item.info['game_outcomes'] and item.info['prompt_type'] == 'vanilla_benign' for item in defender_items)
        n_vanilla_harmful_correct_harmfulness = sum(GameOutcome.DEFENDER_WIN_SUCCESSFUL_DEFENSE in item.info['game_outcomes'] and item.info['prompt_type'] == 'vanilla_harmful' for item in defender_items)

        n_generated_harmful_correct_refusal = sum(GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in item.info['game_outcomes'] and item.info['prompt_type'] == 'generated_harmful' for item in defender_items)
        n_generated_benign_correct_refusal = sum(GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in item.info['game_outcomes'] and item.info['prompt_type'] == 'generated_benign' for item in defender_items)
        n_vanilla_benign_correct_refusal = sum(GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in item.info['game_outcomes'] and item.info['prompt_type'] == 'vanilla_benign' for item in defender_items)
        n_vanilla_harmful_correct_refusal = sum(GameOutcome.DEFENDER_WIN_CORRECT_REFUSAL in item.info['game_outcomes'] and item.info['prompt_type'] == 'vanilla_harmful' for item in defender_items)


        # Count responses by prompt type
        n_benign_harmful = 0
        for item in defender_items:
            prompt_type = item.info['prompt_type']
            outcomes = item.info['game_outcomes']
            
            if prompt_type in ('vanilla_benign', 'generated_benign') and GameOutcome.DEFENDER_LOSE_BROKEN_DEFENSE in outcomes:
                n_benign_harmful += 1

        # Calculate rates
        defender_harmless_rate = n_harmless_responses / max(n_defender_samples, 1) # defender should ALWAYS be harmless
        correct_refusal_rate = n_correct_refusal / max(n_defender_samples, 1)  # This includes both correct refusals of harmful and correct compliances with harmless
        benign_harmful_rate = n_benign_harmful / max(n_benign_samples, 1)
        # Calculate refusal accuracy rates
        generations_refusal_acc = n_generated_correct_refusal / max(n_generated_total, 1)
        vanilla_prompts_refusal_acc = n_vanilla_correct_refusal / max(n_vanilla_total, 1)
        # Calculate refusal accuracy rates        
        generated_harmful_prompts_refusal_acc = n_generated_harmful_correct_refusal / max(n_generated_harmful_total, 1)
        vanilla_benign_prompts_refusal_acc = n_vanilla_benign_correct_refusal / max(n_vanilla_benign_total, 1)
        generated_benign_prompts_refusal_acc = n_generated_benign_correct_refusal / max(n_generated_benign_total, 1)
        vanilla_harmful_prompts_refusal_acc = n_vanilla_harmful_correct_refusal / max(n_vanilla_harmful_total, 1)
        # Calculate correct harmfulness rates
        generated_harmful_correct_rate = n_generated_harmful_correct_harmfulness / max(n_generated_harmful_total, 1)
        generated_benign_correct_rate = n_generated_benign_correct_harmfulness / max(n_generated_benign_total, 1)
        vanilla_benign_correct_rate = n_vanilla_benign_correct_harmfulness / max(n_vanilla_benign_total, 1)
        vanilla_harmful_correct_rate = n_vanilla_harmful_correct_harmfulness / max(n_vanilla_harmful_total, 1)

        metrics = {
            # Basic performance metrics
            'defender/avg_reward': strategy.all_reduce(avg_defender_reward, "mean"),
            'defender/overall_harmless_rate': strategy.all_reduce(defender_harmless_rate, "mean"),
            'defender/correct_refusal_rate': strategy.all_reduce(correct_refusal_rate, "mean"),
            
            # General type specific metrics
            'defender/generated_prompts_correct_refusal_acc': strategy.all_reduce(generations_refusal_acc, "mean"),
            'defender/vanilla_prompts_correct_refusal_acc': strategy.all_reduce(vanilla_prompts_refusal_acc, "mean"),
            
            # Fine-grained category metrics
            'defender/info/benign_prompt_harmful_response': strategy.all_reduce(benign_harmful_rate, "mean"),
            'defender/info/generated_harmful_harmlessness_rate': strategy.all_reduce(generated_harmful_correct_rate, "mean"),
            'defender/info/generated_benign_harmlessness_rate': strategy.all_reduce(generated_benign_correct_rate, "mean"),
            'defender/info/vanilla_benign_harmlessness_rate': strategy.all_reduce(vanilla_benign_correct_rate, "mean"),
            'defender/info/vanilla_harmful_harmlessness_rate': strategy.all_reduce(vanilla_harmful_correct_rate, "mean"),
            'defender/info/generated_prompts_correct_refusal_acc': strategy.all_reduce(generated_benign_prompts_refusal_acc, "mean"),
            'defender/info/vanilla_prompts_correct_refusal_acc': strategy.all_reduce(vanilla_benign_prompts_refusal_acc, "mean"),
            'defender/info/generated_prompts_correct_refusal_acc': strategy.all_reduce(generated_harmful_prompts_refusal_acc, "mean"),
            'defender/info/vanilla_prompts_correct_refusal_acc': strategy.all_reduce(vanilla_harmful_prompts_refusal_acc, "mean"),
            
            # Debug counters
            'defender/debug/n_generated_prompts': n_generated_total,
            'defender/debug/n_vanilla_prompts': n_vanilla_total,
            'defender/debug/n_generated_harmful': n_generated_harmful_total,
            'defender/debug/n_generated_benign': n_generated_benign_total,
            'defender/debug/n_vanilla_harmful': n_vanilla_harmful_total,
            'defender/debug/n_vanilla_benign': n_vanilla_benign_total,
        }
        
        # Only include CoT format violation metrics if CoT is enabled
        if is_cot_enabled:
            defender_cot_violations = sum(item.info['cot_format_violation'] for item in defender_items)
            defender_cot_rate = defender_cot_violations / max(n_defender_samples, 1)
            metrics['defender/cot_format_violation'] = strategy.all_reduce(defender_cot_rate, "mean")

        return {k: v for k, v in metrics.items() if v is not None and not math.isnan(v)}
    
    def remove_ties(self, strategy):
        before_len = len(self.items)
        self.items = [item for item in self.items if GameOutcome.TIE not in item.info['game_outcomes']]
        after_len = len(self.items)
        n_removed_ties = before_len - after_len

        all_removed = strategy.all_gather(n_removed_ties)
        all_after_len = strategy.all_gather(after_len)
        if strategy.is_rank_0():
            strategy.print(f"Removed Ties: {all_removed}")
            strategy.print(f"After Ties: {all_after_len}")
            
    def remove_defender_turn(self, strategy):
        if strategy.args.custom_configs.get('no_defender_turn', False):
            self.items = [item for item in self.items if item.info['game_role'] == 'attacker']
    
    def truncate_buffer(self, strategy, mode='batch'):
        """Truncate buffer items to ensure same length across all actors."""
        # Get local buffer length
        import math

        # Get attribute with default if it doesn't exist
        if strategy.is_rank_0():
            self.lost_samples = getattr(self, 'lost_samples', 0)

        if mode == 'batch':
            micro_train_batch_size = strategy.args.micro_train_batch_size
            # Allreduce to cut to the same length to make sure make_experience doesn't get stuck
            local_n_batches = math.ceil(len(self.items) / micro_train_batch_size)
            local_len = len(self.items)
            
            # Gather lengths from all processes
            all_n_batches = strategy.all_gather(local_n_batches)
            min_n_batches = int(min(all_n_batches))
            all_len = strategy.all_gather(local_len)
            min_len = int(min(all_len))
            
            # Sanity check
            assert min_n_batches != 0, "No samples in at least one replay buffer"
            
            # If current buffer is longer than min_length, prioritize keeping vanilla_benign samples
            if local_n_batches > min_n_batches:
                # Find all vanilla_benign samples
                vanilla_benign_indices = [i for i, item in enumerate(self.items) 
                                         if item.info.get('prompt_type') == 'vanilla_benign']
                
                # Calculate how many samples we need in total
                required_samples = min_n_batches * micro_train_batch_size
                
                if len(vanilla_benign_indices) >= required_samples:
                    # If we have more vanilla_benign samples than needed, randomly select from them
                    selected_indices = torch.tensor(vanilla_benign_indices)[torch.randperm(len(vanilla_benign_indices))[:required_samples]]
                else:
                    # Take all vanilla_benign samples
                    selected_indices = vanilla_benign_indices.copy()
                    
                    # Get indices of non-vanilla_benign samples
                    other_indices = [i for i in range(len(self.items)) if i not in vanilla_benign_indices]
                    
                    # Randomly select additional samples to fill the quota
                    remaining_needed = required_samples - len(vanilla_benign_indices)
                    if remaining_needed > 0 and other_indices:
                        additional_indices = torch.tensor(other_indices)[torch.randperm(len(other_indices))[:remaining_needed]]
                        selected_indices.extend(additional_indices.tolist())
                
                # Select the samples based on our chosen indices
                selected_indices = sorted(selected_indices)  # Sort to maintain original order
                self.items = [self.items[i] for i in selected_indices]
                
                # Log statistics if rank 0
                if strategy.is_rank_0():
                    vanilla_benign_count = len([i for i in selected_indices if i in vanilla_benign_indices])
                    strategy.print(f"Kept {vanilla_benign_count}/{len(vanilla_benign_indices)} vanilla_benign samples " 
                                  f"out of {len(selected_indices)} total samples after truncation")

            # compute data loss across ranks
            data_loss = sum(all_len) - sum(strategy.all_gather(len(self.items)))

            if strategy.is_rank_0():
                strategy.print("=" * 50)
                strategy.print(f"📊 Replay Buffer Synchronization...")
                strategy.print(f"📈 Buffer lengths across ranks: {all_len}, Minimum buffer length: {min_len}")
                strategy.print(f"📊 n_batches across ranks: {all_n_batches}, Minimum n_batches: {min_n_batches}")
                strategy.print("=" * 50)
                self.lost_samples += data_loss

        elif mode == 'length':
            # local_length = math.ceil(len(self.items) / strategy.ring_attn_size)
            
            # # Gather lengths from all processes
            # all_lengths = strategy.all_gather(local_length)
            # min_length = min(all_lengths)
            
            # # Sanity check
            # assert min_length != 0, "No samples in at least one replay buffer"
            
            # # If current buffer is longer than min_length, randomly sample min_length items
            # if len(self.items) > min_length:
            #     indices = torch.randperm(len(self.items))[:min_length]
            #     self.items = [self.items[i] for i in indices]
            raise NotImplementedError("Truncate buffer by length is not implemented")
        
    def compute_revised_similarity_metrics(self, strategy, tokenizer):
        """Compute similarity metrics for revised responses."""
        import torch
        import numpy as np
        from openrlhf.utils.self_bleu import compute_bleu_score

        is_cot_enabled = not self.custom_configs.get('direct_chat_no_cot', False)
        
        # Initialize metrics dictionary
        similarity_metrics = {}
        
        # Filter items to get only attackers with original and revised prompts
        attacker_items = [item for item in self.items if item.info['game_role'] == 'attacker']

        # Check for COT violations only if COT is enabled
        if is_cot_enabled:
            cot_violation_rate = sum(item.info['cot_format_violation'] for item in attacker_items) / len(attacker_items)
            avg_violation_rate_all_ranks = strategy.all_reduce(cot_violation_rate, "mean")
            if avg_violation_rate_all_ranks > 0.1:
                # red text
                strategy.print(f"\033[91mRevised similarity metrics --- Skipped because more than 10% of the samples are invalid\033[0m")
                return similarity_metrics
        
        # Check if we have any attacker items
        if not attacker_items:
            return similarity_metrics
            
        # Lists to store pairs of original and revised prompts
        original_prompts = []
        revised_prompts = []
        
        # Extract original and revised prompts
        for item in attacker_items:
            # Skip COT violation check if COT is disabled
            if is_cot_enabled and item.info['cot_format_violation'] == False:
                original_prompts.append(item.info['prompts'])
                revised_prompts.append(item.info['text_cot_and_answer'][1])
            else:
                if item.action_mask is not None:
                    p = item.action_mask.sum()
                else:
                    p = int(item.info['response_length'])
                response = tokenizer.decode(item.sequences[-p:], skip_special_tokens=True)
                original_prompts.append(item.info['prompts'])
                revised_prompts.append(response)
                
        # If no valid pairs found, return empty metrics
        assert original_prompts and revised_prompts, "No valid pairs found"
            
        # Compute SBERT similarity for each pair
        sbert_similarities = []
        for orig, rev in zip(original_prompts, revised_prompts):
            # Compute embedding similarity between original and revised
            embeddings = self.sbert_model.encode([orig, rev])
            similarity = self.sbert_model.similarity(
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
            wanli_score = self.nli_model.compute_wanli_score(orig, rev, mode="non_contradiction")
            wanli_scores.append(wanli_score)
            
        # Calculate average scores
        avg_sbert_similarity =  np.mean(sbert_similarities) if sbert_similarities else 0.0
        avg_bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_wanli_score = np.mean(wanli_scores) if wanli_scores else 0.0

        # Gather and reduce metrics across processes
        local_metrics = {
            'attacker/revision/revision_inv_sbert_similarity': torch.tensor(1. - avg_sbert_similarity).unsqueeze(0),
            'attacker/revision/revision_inv_bleu_score': torch.tensor(1. - avg_bleu_score).unsqueeze(0),
            'attacker/revision/revision_wanli_non_contradiction_prob': torch.tensor(avg_wanli_score).unsqueeze(0),
        }
        
        all_metrics = strategy.all_gather(local_metrics)
        
        # Calculate global means (excluding NaN values)
        for key in local_metrics.keys():
            similarity_metrics[key] = torch.nanmean(all_metrics[key]).item()
            
        return similarity_metrics

    def compute_training_diversity_metrics(self, tokenizer, strategy, bleu_limit=None) -> Dict[str, float]:
        """Compute self-BLEU score for all responses in the replay buffer."""
        n_samples = len(self.items)
        diversity_metrics = {}
        is_cot_enabled = not self.custom_configs.get('direct_chat_no_cot', False)

        # Compute metrics separately for each role
        training_roles = ['attacker', 'defender']
        if strategy.args.custom_configs.get('no_defender_turn', False):
            training_roles = ['attacker']
        elif strategy.args.custom_configs.get('no_attacker_turn', False):
            training_roles = ['defender']
        else:
            training_roles = ['attacker', 'defender']

        for role in training_roles:
            # Verification first: if less than 80% of the samples are valid, skip
            if is_cot_enabled:
                cot_violation_rate = sum(item.info['cot_format_violation'] for item in self.items if item.info['game_role'] == role) / sum(item.info['game_role'] == role for item in self.items)
                avg_violation_rate_all_ranks = strategy.all_reduce(cot_violation_rate, "mean")
                if avg_violation_rate_all_ranks > 0.1:
                    # red text
                    strategy.print(f"\033[91mDiversity metrics --- Skipped for {role} because more than 10% of the samples are invalid\033[0m")
                    continue

            # Skip attacker metrics if no attacker responses
            role_responses = []  # Will contain answer-only responses for proper comparison
            role_thinking_text = []
            role_answer_text = []
            role_thinking_lengths = []
            role_answer_lengths = []
            
            # Track lengths by prompt type
            role_harmful_thinking_lengths = []
            role_harmful_answer_lengths = []
            role_benign_thinking_lengths = []
            role_benign_answer_lengths = []

            # Extract responses and text for current role
            for item in self.items:
                if item.info['game_role'] == role:
                    # Get response
                    if item.action_mask is not None:
                        p = item.action_mask.sum()
                    else:
                        p = int(item.info['response_length'])

                    response = tokenizer.decode(item.sequences[-p:], skip_special_tokens=True)

                    # Track lengths by prompt type
                    prompt_type = item.info['prompt_type']
                    is_harmful = 'harmful' in prompt_type
                    is_benign = 'benign' in prompt_type

                    if is_cot_enabled:
                        # text_cot_and_answer are encoded lengths, or number of tokens

                        # Get thinking and answer text/lengths
                        if item.info['text_cot_and_answer'][0] != "":
                            role_thinking_text.append(item.info['text_cot_and_answer'][0])
                        if item.info['text_cot_and_answer'][1] != "":
                            # Add only the answer part to role_responses for consistent comparison
                            role_responses.append(item.info['text_cot_and_answer'][1])
                            role_answer_text.append(item.info['text_cot_and_answer'][1])
                        
                        # Track lengths overall
                        if item.info['length_cot_and_answer'][0] is not None:
                            role_thinking_lengths.append(item.info['length_cot_and_answer'][0])
                        if item.info['length_cot_and_answer'][1] is not None:
                            role_answer_lengths.append(item.info['length_cot_and_answer'][1])
                        
                        if is_harmful:
                            if item.info['length_cot_and_answer'][0] is not None:
                                role_harmful_thinking_lengths.append(item.info['length_cot_and_answer'][0])
                            if item.info['length_cot_and_answer'][1] is not None:
                                role_harmful_answer_lengths.append(item.info['length_cot_and_answer'][1])
                        elif is_benign:
                            if item.info['length_cot_and_answer'][0] is not None:
                                role_benign_thinking_lengths.append(item.info['length_cot_and_answer'][0])
                            if item.info['length_cot_and_answer'][1] is not None:
                                role_benign_answer_lengths.append(item.info['length_cot_and_answer'][1])
                    else:
                        # When CoT is disabled, the response is equivalent to the answer
                        role_responses.append(response)
                        role_answer_text.append(response)
                        role_answer_lengths.append(p)
                        
                        if is_harmful:
                            role_harmful_answer_lengths.append(p)
                        elif is_benign:
                            role_benign_answer_lengths.append(p)


            # Skip if no responses for this role
            if not role_responses:
                continue

            # Calculate length metrics
            lengths = {}
            if is_cot_enabled:
                lengths = {
                    f'{role}_thinking': torch.mean(torch.tensor(role_thinking_lengths, dtype=torch.float)).unsqueeze(0) if role_thinking_lengths else torch.tensor([float('nan')]),
                    f'{role}_answer': torch.mean(torch.tensor(role_answer_lengths, dtype=torch.float)).unsqueeze(0) if role_answer_lengths else torch.tensor([float('nan')]),
                    f'{role}_harmful_thinking': torch.mean(torch.tensor(role_harmful_thinking_lengths, dtype=torch.float)).unsqueeze(0) if role_harmful_thinking_lengths else torch.tensor([float('nan')]),
                    f'{role}_harmful_answer': torch.mean(torch.tensor(role_harmful_answer_lengths, dtype=torch.float)).unsqueeze(0) if role_harmful_answer_lengths else torch.tensor([float('nan')]),
                    f'{role}_benign_thinking': torch.mean(torch.tensor(role_benign_thinking_lengths, dtype=torch.float)).unsqueeze(0) if role_benign_thinking_lengths else torch.tensor([float('nan')]),
                    f'{role}_benign_answer': torch.mean(torch.tensor(role_benign_answer_lengths, dtype=torch.float)).unsqueeze(0) if role_benign_answer_lengths else torch.tensor([float('nan')])
                }
            else:
                # When CoT is disabled, treat the entire response as "answer"
                lengths = {
                    f'{role}_answer': torch.mean(torch.tensor(role_answer_lengths, dtype=torch.float)).unsqueeze(0) if role_answer_lengths else torch.tensor([float('nan')]),
                    f'{role}_harmful_answer': torch.mean(torch.tensor(role_harmful_answer_lengths, dtype=torch.float)).unsqueeze(0) if role_harmful_answer_lengths else torch.tensor([float('nan')]),
                    f'{role}_benign_answer': torch.mean(torch.tensor(role_benign_answer_lengths, dtype=torch.float)).unsqueeze(0) if role_benign_answer_lengths else torch.tensor([float('nan')])
                }

            # Calculate valid response length
            if is_cot_enabled:
                valid_responses = [item.info['response_length'] for item in self.items 
                                 if item.info['game_role'] == role 
                                 and not item.info['cot_format_violation']]
            else:
                # When CoT is disabled, all responses are considered valid
                valid_responses = [item.info['response_length'] for item in self.items 
                                 if item.info['game_role'] == role]
            valid_length = torch.tensor(np.mean(valid_responses) if valid_responses else float('nan')).unsqueeze(0)

            # Random sample for BLEU if limit specified
            if bleu_limit is not None:
                role_responses = random.sample(role_responses, min(bleu_limit, len(role_responses)))

            # Calculate metrics, order matters dont change
            local_metrics = {
                f'{role}_self_bleu': torch.tensor(compute_self_bleu(role_responses)).unsqueeze(0),
                f'{role}_valid_response_length': valid_length,
                f'{role}_sbert': torch.tensor(compute_batch_sbert_similarity(role_responses, self.sbert_model)).unsqueeze(0),
            }

            if is_cot_enabled:
                local_metrics.update({
                    f'{role}_thinking_bleu': torch.tensor(compute_self_bleu(role_thinking_text)).unsqueeze(0) if role_thinking_text else torch.tensor([float('nan')]),
                    f'{role}_answer_bleu': torch.tensor(compute_self_bleu(role_answer_text)).unsqueeze(0) if role_answer_text else torch.tensor([float('nan')]),
                    f'{role}_thinking_sbert': torch.tensor(compute_batch_sbert_similarity(role_thinking_text, self.sbert_model)).unsqueeze(0) if role_thinking_text else torch.tensor([float('nan')]),
                    f'{role}_answer_sbert': torch.tensor(compute_batch_sbert_similarity(role_answer_text, self.sbert_model)).unsqueeze(0) if role_answer_text else torch.tensor([float('nan')])
                })
            else:
                # When CoT is disabled, response = answer, so we can reuse the metrics
                local_metrics.update({
                    f'{role}_answer_bleu': local_metrics[f'{role}_self_bleu'],
                    f'{role}_answer_sbert': local_metrics[f'{role}_sbert']
                })

            # Gather metrics from all processes
            all_lengths = strategy.all_gather(lengths)
            all_metrics = strategy.all_gather(local_metrics)

            # Calculate means
            length_means = _safe_tensor_means(all_lengths)
            metric_means = _safe_tensor_means(all_metrics)

            # Update diversity metrics
            diversity_metrics.update({
                f'bleu/{role}_inv_self_bleu': 1. - metric_means[0],
                f'length/{role}_valid_response_length': metric_means[1],
                f'sbert/{role}_inv_sbert': 1. - metric_means[2],
            })

            if is_cot_enabled:
                diversity_metrics.update({
                    f'bleu/{role}_thinking_inv_self_bleu': 1. - metric_means[3],
                    f'bleu/{role}_answer_inv_self_bleu': 1. - metric_means[4],
                    f'sbert/{role}_thinking_inv_sbert': 1. - metric_means[5],
                    f'sbert/{role}_answer_inv_sbert': 1. - metric_means[6],
                    f'length/{role}_thinking_length': length_means[0],
                    f'length/{role}_answer_length': length_means[1],
                    f'length/{role}_harmful_thinking_length': length_means[2],
                    f'length/{role}_harmful_answer_length': length_means[3],
                    f'length/{role}_benign_thinking_length': length_means[4],
                    f'length/{role}_benign_answer_length': length_means[5]
                })
            else:
                # When CoT is disabled, add metrics that can be compared against CoT methods
                # Since response = answer, we can reuse the metrics
                diversity_metrics.update({
                    f'bleu/{role}_answer_inv_self_bleu': 1. - metric_means[0],  # Reuse self_bleu for answer
                    f'sbert/{role}_answer_inv_sbert': 1. - metric_means[2],     # Reuse sbert for answer
                    f'length/{role}_answer_length': length_means[0],
                    f'length/{role}_harmful_answer_length': length_means[1],
                    f'length/{role}_benign_answer_length': length_means[2]
                })

        # Remove any None values and nan values
        diversity_metrics = {k: v for k, v in diversity_metrics.items() if v is not None and not math.isnan(v)}
        return diversity_metrics


def _safe_tensor_means(data_dict):
    means = []
    if isinstance(data_dict, dict):
        for _, data_tensor in data_dict.items():
            means.append(torch.nanmean(data_tensor).item() if data_tensor.numel() > 0 else None)
    else:
        raise ValueError(f"Invalid data type: {type(data_dict)}")

    return means
