#!/bin/bash

# Exit on any error
set -e

# Use associative array to map model paths to chat templates
declare -A MODEL_TEMPLATES
MODEL_TEMPLATES=(
    ["/path/to/model/1"]="llama3_cot"
    ["/path/to/model/2"]="llama3_cot"
    ["/path/to/model/3"]=""
)
TIMESTAMP=$(date +%Y%m%dT%H%M)
EVAL_HOME_DIR=~/code/selfplay-openrlhf/eval/results/full_eval_${TIMESTAMP}
mkdir -p ${EVAL_HOME_DIR}

# Print header
echo "Starting full evaluation for all checkpoints..."
echo "EVAL_HOME_DIR: $EVAL_HOME_DIR"
echo "============================================="

# Iterate through each model path and its template
for MODEL_PATH in "${!MODEL_TEMPLATES[@]}"; do
    CHAT_TEMPLATE="${MODEL_TEMPLATES[$MODEL_PATH]}"
    
    echo "Processing checkpoint: $MODEL_PATH with template: $CHAT_TEMPLATE"
    echo "----------------------------------------"
    
    # Run the full evaluation script
    bash eval/full_eval.sh \
        --model_path "$MODEL_PATH" \
        --chat_template_name "$CHAT_TEMPLATE" \
        --eval_home_dir "$EVAL_HOME_DIR" \
        # --skip_safety_eval \ # If you want to skip safety eval (safety-eval)
        # --skip_general_eval \ # If you want to skip general eval (lm-eval)
        # --run_chat_eval # If you want to run chat eval (olmes)
    echo "Completed evaluation for: $MODEL_PATH"
    echo "============================================="
done

echo "All evaluations completed!"