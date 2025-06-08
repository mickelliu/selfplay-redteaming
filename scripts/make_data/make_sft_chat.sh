#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status
set -x  # Print commands for debugging

# Parse command line arguments
temperature=0.6
top_p=0.9
DATASET_PATH=""
GENERATE_OUTPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --cuda_devices)
      CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --temperature)
      temperature="$2"
      shift 2
      ;;
    --top_p)
      top_p="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --model_path MODEL_PATH --cuda_devices CUDA_DEVICES [--temperature TEMP] [--top_p TOP_P]"
      exit 1
      ;;
  esac
done

CUDA_VISIBLE_DEVICES="0,3" python3 -m openrlhf.cli.batch_inference \
       --eval_task generate_vllm \
       --pretrain $MODEL_PATH \
       --max_new_tokens 2048 \
       --prompt_max_len 2048 \
       --dataset $DATASET_PATH \
       --input_key vanilla \
       --max_samples 30000 \
       --seed 8888 \
       --temperature $temperature \
       --top_p $top_p \
       --tp_size 2 \
       --best_of_n 1 \
       --enable_prefix_caching \
       --max_num_seqs 256 \
       --output_path $GENERATE_OUTPUT