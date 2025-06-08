#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status
set -x  # Print commands for debugging

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output_filename)
      OUTPUT_FILENAME="$2"
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
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --model_path MODEL_PATH --cuda_devices CUDA_DEVICES [--temperature TEMP] [--top_p TOP_P]"
      exit 1
      ;;
  esac
done

MODEL_PATH="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
HOME_DIR=""
DATASET_PATH=""
POSTFILL_COT_GENERATE_OUTPUT=""
MAX_SAMPLES=15000

CUDA_VISIBLE_DEVICES="0,1" python3 -m openrlhf.cli.batch_inference \
   --eval_task generate_vllm_postfill_cot \
   --pretrain $MODEL_PATH \
   --prompt_max_len 4096 \
   --max_new_tokens 1024 \
   --temperature 0.6 \
   --top_p 0.9 \
   --input_key vanilla \
   --label_key completion \
   --dataset $DATASET_PATH \
   --max_samples $MAX_SAMPLES \
   --seed 8888 \
   --tp_size 2 \
   --best_of_n 1 \
   --enable_prefix_caching \
   --max_num_seqs 256 \
   --output_path $POSTFILL_COT_GENERATE_OUTPUT