#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# ConsisID configuration
SCRIPT="teacache/consisid_teacache.py"
MODEL_ID="BestWishYsh/ConsisID-preview"
INFERENCE_STEP=50

mkdir -p ./results

# ConsisID specific task args
TASK_ARGS="--height 480 --width 720 --num_frames 49"

# ConsisID parallel configuration
N_GPUS=6

PARALLEL_ARGS="--ulysses_degree $N_GPUS --ring_degree 1"
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# ENABLE_TILING="--enable_tiling"
# COMPILE_FLAG="--use_torch_compile"

torchrun --master_port=1234 --nproc_per_node=$N_GPUS $SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel." \
--img_file_path /home/LiaoMingxiang/Workspace/xDiT/imgs/2.png \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG