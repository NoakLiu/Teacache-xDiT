#!/bin/bash
set -x

find_free_port() {
  PORT=29500
  while lsof -i:$PORT &>/dev/null; do
    PORT=$((PORT + 1))
  done
  echo $PORT
}

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="teacache/cogvideox_tecache.py"
# SCRIPT="cogvideox_example.py"
MODEL_ID="./ckpts/THUDM/CogVideoX1.5-5B"
INFERENCE_STEP=50
FREE_PORT=$(find_free_port)

mkdir -p ./results

# CogVideoX specific task args
TASK_ARGS="--height 768 --width 1360 --num_frames 17"
export CUDA_VISIBLE_DEVICES=1
# CogVideoX parallel configuration
N_GPUS=1
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 1"
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS --master_port=${FREE_PORT} ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A little girl is riding a bicycle at high speed. Focused, detailed, realistic." \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG
