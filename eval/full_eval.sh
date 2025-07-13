#!/bin/bash

# Exit on any error
set -e

# Configuration variables
BASE_CHECKPOINT_PATH="/path/to/checkpoints_home/" # e.g. /mmfs1/home/mickel7/code/selfplay-openrlhf/checkpoints/red_team_game/apr25/

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --eval_home_dir)
      EVAL_HOME_DIR="$2"
      shift 2
      ;;
    --run_chat_eval)
      RUN_CHAT_EVAL="true"
      shift
      ;;
    --chat_template_name)
      CHAT_TEMPLATE_NAME="$2"
      shift 2
      ;;
    --skip_general_eval)
      SKIP_GENERAL_EVAL="true"
      shift
      ;;
    --skip_safety_eval)
      SKIP_SAFETY_EVAL="true"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    exit 1
fi

if [ -z "$EVAL_HOME_DIR" ]; then
    echo "Error: --eval_home_dir is required"
    exit 1
fi

# Add check to ensure at least one evaluation will run
if [ "$SKIP_GENERAL_EVAL" = "true" ] && [ "$SKIP_SAFETY_EVAL" = "true" ] && [ -z "$RUN_CHAT_EVAL" ]; then
    echo "Error: Cannot skip all evaluations. At least one of general eval, safety eval, or chat eval must run."
    exit 1
fi

TIMESTAMP=$(date +%Y%m%dT%H%M)
# Extract a unique model name from the path that preserves hierarchy
MODEL_NAME=$(echo "$MODEL_PATH" | sed "s|^${BASE_CHECKPOINT_PATH}||" | sed 's|/|__|g' | sed 's/[\"<>:|\\?*\[\]]\+/__/g')
RESULTS_DIR=${EVAL_HOME_DIR}/${MODEL_NAME}
mkdir -p ${RESULTS_DIR}

# openai key
if [ "$RUN_CHAT_EVAL" = "true" ]; then
    source ~/.secure/keys/openai
    export OPENAI_API_KEY=$OPENAI_API_KEY
else
    export OPENAI_API_KEY=sk-proj-0123456789abcdef0123456789abcdef
fi

# Initialize conda for bash script
eval "$(conda shell.bash hook)"

# Validate conda environments exist
if ! conda env list | grep -q "lm-eval"; then
    echo "Error: lm-eval conda environment not found"
    exit 1
fi

if ! conda env list | grep -q "olmes"; then
    echo "Error: olmes conda environment not found"
    exit 1
fi

if ! conda env list | grep -q "safety-eval-fork"; then
    echo "Error: safety-eval-fork conda environment not found"
    exit 1
fi

# Add these helper functions at the beginning of the script after the initial checks
print_stage() {
    local text="$1"
    local width=80
    local padding=$(( (width - ${#text}) / 2 ))
    
    echo ""
    echo -e "\033[1;36m╔══$(printf '═%.0s' $(seq 1 $((width-4))))══╗\033[0m"
    echo -e "\033[1;36m║$(printf ' %.0s' $(seq 1 $padding))$text$(printf ' %.0s' $(seq 1 $(($width - ${#text} - $padding))))║\033[0m"
    echo -e "\033[1;36m╚══$(printf '═%.0s' $(seq 1 $((width-4))))══╝\033[0m"
    echo ""
}

print_status() {
    local text="$1"
    echo -e "\033[1;33m➜\033[0m \033[1;32m$text\033[0m"
}

# Record model path and timestamp to a file
{
    echo "Evaluation started at: ${TIMESTAMP}"
    echo "Model path: ${MODEL_PATH}"
} > "${RESULTS_DIR}/eval_info.txt"


# Chat evaluation with olmes (if requested)
# Chat Evaluation is skipped by default due to the cost of running it
if [ -n "$RUN_CHAT_EVAL" ]; then
    print_stage "RUNNING CHAT EVALUATION WITH OLMES"
    cd ~/code/benchmarks/olmes || exit 1
    print_status "Activating olmes environment..."
    conda deactivate && conda activate olmes || exit 1
    
    # Verify OPENAI_API_KEY is set
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY is not set"
        exit 1
    fi
    
    # Add current directory to PYTHONPATH (use absolute path)
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
    
    # TODO: need to install math_verify
    # --task alpaca_eval_v2::sampling_enabled ifeval::sampling_enabled \
    if [ "$CHAT_TEMPLATE_NAME" == "hf" ]; then
        model_args_json="{\"chat_template\": \"\", \"max_length\": 8192}"
    else
        model_args_json="{\"chat_template\": \"${CHAT_TEMPLATE_NAME}\", \"max_length\": 8192}"
    fi

    olmes --model "${MODEL_PATH}" \
        --model-type vllm \
        --use-chat-format true \
        --task alpaca_eval_v2::sampling_enabled ifeval::sampling_enabled \
        --output-dir "${RESULTS_DIR}/chat_eval" \
        --model-args "${model_args_json}" \
        --gpus 4 \
        --limit 1 \
        --batch-size "auto" || { echo "Chat evaluation failed"; exit 1; }
fi

# Wildguard bench
if [ "$SKIP_SAFETY_EVAL" != "true" ]; then
    print_stage "STARTING SAFETY EVALUATION"
    cd ~/code/benchmarks/safety-eval-fork || exit 1
    print_status "Activating safety-eval-fork environment..."
    conda deactivate && conda activate safety-eval-fork || exit 1

    # Add current directory to PYTHONPATH (use absolute path)
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
    mkdir -p "${RESULTS_DIR}/safety_eval"

    python evaluation/run_all_generation_benchmarks.py \
        --model_name_or_path "${MODEL_PATH}" \
        --report_output_path "${RESULTS_DIR}/safety_eval/metrics.json" \
        --save_individual_results_path "${RESULTS_DIR}/safety_eval/all.json" \
        --min_gpus_per_task 1 \
        --model_input_template_path_or_name "${CHAT_TEMPLATE_NAME}" || { echo "Safety evaluation failed"; exit 1; }
fi

# LM-EVAL-HARNESS evaluation (replacing olmes for general evaluation)
if [ "$SKIP_GENERAL_EVAL" != "true" ]; then
    print_stage "STARTING LM-EVAL-HARNESS EVALUATION SUITE"
    cd ~/code/benchmarks/lm-evaluation-harness || exit 1
    print_status "Activating lm-eval environment..."
    conda deactivate && conda activate lm-eval || exit 1

    # Add current directory to PYTHONPATH (use absolute path)
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
    # Define evaluation tasks
    LIKELIHOOD_TASKS="truthfulqa_mc1,gpqa_main_zeroshot,arc_challenge,mmlu"
    GENERATION_TASKS="bbh_cot_fewshot"

    print_stage "RUNNING LIKELIHOOD TASKS EVALUATION"
    print_status "Running HF evaluation for likelihood tasks..."

    # Likelihood tasks
    accelerate launch -m lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks $LIKELIHOOD_TASKS \
        --batch_size auto:8 \
        --output_path "${RESULTS_DIR}/general_eval" \
        --trust_remote_code || { echo "Likelihood evaluation failed"; exit 1; }

    print_stage "RUNNING GENERATION TASKS EVALUATION"
    print_status "Running VLLM evaluation for generation tasks..."

    # Generation tasks
    lm_eval --model vllm  \
        --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
        --tasks $GENERATION_TASKS \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/general_eval" \
        --trust_remote_code || { echo "Generation evaluation failed"; exit 1; }
fi

print_stage "ALL EVALUATIONS COMPLETED SUCCESSFULLY ✨"

# parse results to csv
python ~/code/selfplay-openrlhf/eval/parse_result_to_csv.py "${RESULTS_DIR}"
print_status "Results parsed to csv"
