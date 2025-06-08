#!/bin/bash
set -x

MODEL_PATH="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"

# # ============ select prefix ============
PREFIX="selfplay_RL_FULL_PTX_SFT_wjbhs"
# PREFIX="selfplay_RL_FULL_PTX_SFT_wjbhs_defender_only"
# PREFIX="selfplay_RL_only"
# PREFIX="no_cot_RL_only"
# PREFIX="defender_only"
# PREFIX="attacker_only"
# PREFIX="no_cot_RL_attacker_only"
# # ============ select prefix ============

# Set the base custom configs
if [[ "$PREFIX" == "selfplay_RL_only" ]]; then
    CUSTOM_CONFIGS='{"max_turns":2,"reward_type":"general_sum","remove_ties":true}'
elif [[ "$PREFIX" == "no_cot_RL_only" ]]; then
    CUSTOM_CONFIGS='{"max_turns":2,"reward_type":"general_sum","remove_ties":true,"direct_chat_no_cot":true}'
elif [[ "$PREFIX" == "defender_only" ]]; then
    CUSTOM_CONFIGS='{"max_turns":2,"reward_type":"general_sum","remove_ties":true,"no_attacker_turn":true}'
elif [[ "$PREFIX" == "selfplay_RL_FULL_PTX_SFT_wjbhs_defender_only" ]]; then
    CUSTOM_CONFIGS='{"max_turns":2,"reward_type":"general_sum","remove_ties":true,"no_attacker_turn":true}'
elif [[ "$PREFIX" == "attacker_only" ]]; then
    CUSTOM_CONFIGS='{"max_turns":2,"reward_type":"general_sum","remove_ties":true,"no_defender_turn":true}'
elif [[ "$PREFIX" == "no_cot_RL_attacker_only" ]]; then
    CUSTOM_CONFIGS='{"max_turns":2,"reward_type":"general_sum","remove_ties":true,"direct_chat_no_cot":true,"no_defender_turn":true}'
else
    CUSTOM_CONFIGS='{"max_turns":2,"reward_type":"general_sum","remove_ties":true}'
fi

RUN_NAME="${PREFIX}_re++_rtg_$(date +%m%dT%H:%M)"
REMOTE_RM_URL="http://0.0.0.0:5000/classify"

WANDB_CONFIG="--wandb_org XXXXXX \
--wandb_run_name $RUN_NAME \
--use_wandb $WANDB_API_KEY \
--wandb_project XXXXXX \
--wandb_max_log 10000 \
--wandb_table_log_interval 1 \
--wandb_table_csv_path checkpoints/$RUN_NAME/run_tables"

# Run the training script directly
python3 -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --remote_rm_url $REMOTE_RM_URL \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.7 \
    --pretrain $MODEL_PATH \
    --save_path checkpoints/${RUN_NAME} \
    --ckpt_path checkpoints/${RUN_NAME}/ckpt \
    --save_steps 100 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --micro_train_batch_size 8  \
    --train_batch_size 32  \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 128 \
    --prompt_data "red_team/data/vanilla_harmful_dataset.jsonl, red_team/data/vanilla_benign_dataset.jsonl" \
    --prompt_data_probs "0.5, 0.5" \
    --eval_data "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl" \
    --sft_data "red_team/data/helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl, red_team/data/vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl" \
    --sft_data_probs "0.5, 0.5" \
    --sft_input_key "vanilla" \
    --sft_output_key "completion" \
    --sft_steps 1 \
    --sft_batches_per_step 1 \
    --max_samples 40000 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --flash_attn \
    --zero_stage 3 \
    --num_episodes 1 \
    --bf16 \
    --seed 8888 \
    --top_p 1.0 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --normalize_reward \
    --packing_samples \
    --gradient_checkpointing \
    --advantage_estimator reinforce \
    --custom_configs $CUSTOM_CONFIGS \
    --actor_loss_coef 1.0 \
    --postfill_cot_loss_coef 1.0 \
    --eval_steps 10 \
    --eval_start_steps 50 \
    --diversity_score_steps 5 \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --stop_at_step_200 \
    $WANDB_CONFIG \