#!/bin/bash
set -x
IP=$(hostname -I | awk '{print $1}')
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TP_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done
# Set defaults if not provided
N_GPUS=${N_GPUS:-4}  # Default to 4 GPUs
TP_SIZE=${TP_SIZE:-1} # Default to tensor parallel size 1

# simple server
python -m openrlhf.cli.serve_wildguard \
    --model_path allenai/wildguard \
    --port 5000 \
    --host $IP \
    --bf16 \
    --num_gpus $N_GPUS \
    --tensor_parallel_size $TP_SIZE \
    --enforce_eager \
    --seed 42 \
    --gpu_memory_utilization 0.9