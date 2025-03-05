# from https://github.com/chengzeyi/ParaAttention/blob/main/examples/run_hunyuan_video.py
import functools
from typing import Any, Dict, Union, Optional
import logging
import time

import torch

from diffusers import DiffusionPipeline, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND
from diffusers.utils import export_to_video
import logging


from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

from xfuser.model_executor.layers.attention_processor import xFuserHunyuanVideoAttnProcessor2_0

assert xFuserHunyuanVideoAttnProcessor2_0 is not None

import numpy as np
import torch.distributed as dist


def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logging.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        assert batch_size % get_classifier_free_guidance_world_size(
        ) == 0, f"Cannot split dim 0 of hidden_states ({batch_size}) into {get_classifier_free_guidance_world_size()} parts."

        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, time_embedding = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states,
                                                      timestep,
                                                      encoder_attention_mask)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1)
        hidden_states = hidden_states.flatten(1, 3)

        hidden_states = torch.chunk(hidden_states,
                                    get_classifier_free_guidance_world_size(),
                                    dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states,
                                    get_sequence_parallel_world_size(),
                                    dim=-2)[get_sequence_parallel_rank()]

        encoder_attention_mask = encoder_attention_mask[0].to(torch.bool)
        encoder_hidden_states_indices = torch.arange(
            encoder_hidden_states.shape[1],
            device=encoder_hidden_states.device)
        encoder_hidden_states_indices = encoder_hidden_states_indices[
            encoder_attention_mask]
        encoder_hidden_states = encoder_hidden_states[
            ..., encoder_hidden_states_indices, :]
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size(
        ) != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        encoder_hidden_states = torch.chunk(
            encoder_hidden_states,
            get_classifier_free_guidance_world_size(),
            dim=0)[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states,
                get_sequence_parallel_world_size(),
                dim=-2)[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        device = hidden_states.device
        # 4. Transformer blocks
        if self.enable_teacache:
            if dist.is_initialized():
                tensor_cnt = torch.tensor(self.cnt, device=device)
                tensor_accum = torch.tensor(self.accumulated_rel_l1_distance, device=device)
                dist.broadcast(tensor_cnt, src=0)
                dist.broadcast(tensor_accum, src=0)
                self.cnt = tensor_cnt.item()
                self.accumulated_rel_l1_distance = tensor_accum.item()

            inp = hidden_states.clone()
            temb_ = time_embedding.clone()
            modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
            
            if self.cnt == 0 or self.cnt == self.num_steps:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [
                    7.33226126e+02,
                    -4.01131952e+02,
                    6.75869174e+01,
                    -3.14987800e+00,
                    9.61237896e-02,
                ]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                
                if dist.is_initialized():
                    tensor_accum = torch.tensor(self.accumulated_rel_l1_distance, device=device)
                    dist.broadcast(tensor_accum, src=0)
                    self.accumulated_rel_l1_distance = tensor_accum.item()

                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                    
            should_calc_tensor = torch.tensor(should_calc, device=hidden_states.device)
            if dist.is_initialized():
                dist.broadcast(should_calc_tensor, src=0)
                
            should_calc = should_calc_tensor.item()

            self.previous_modulated_input = modulated_inp
            self.cnt += 1

            if self.cnt == self.num_steps:
                self.cnt = 0

        if dist.is_initialized():
            dist.barrier()
            
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for block in self.transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                    )

                for block in self.single_transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                    )
            # hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )
        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
        hidden_states = get_cfg_group().all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.reshape(batch_size,
                                              post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, -1, p_t, p, p)

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states, )

        return Transformer2DModelOutput(sample=hidden_states)

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
        block.attn.processor = xFuserHunyuanVideoAttnProcessor2_0()


def timestep_forward(self, timestep, guidance, pooled_projection):
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

    guidance_proj = self.time_proj(guidance)
    guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))  # (N, D)

    time_guidance_emb = timesteps_emb + guidance_emb

    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = time_guidance_emb + pooled_projections

    return conditioning, timesteps_emb


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    print('x'*20)
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )
    print(f'0'*20)
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb = self.time_text_embed(timestep, guidance, pooled_projections)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 3. Attention mask preparation
    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.zeros(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )  # [B, N]

    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

    for i in range(batch_size):
        attention_mask[i, : effective_sequence_length[i]] = True
    # [B, 1, 1, N], for broadcasting across attention heads
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    device = hidden_states.device
    # 4. Transformer blocks
    print(f'1'*20)
    if self.enable_teacache:
        if dist.is_initialized():
            tensor_cnt = torch.tensor(self.cnt, device=device)
            tensor_accum = torch.tensor(self.accumulated_rel_l1_distance, device=device)
            dist.broadcast(tensor_cnt, src=0)
            dist.broadcast(tensor_accum, src=0)
            self.cnt = tensor_cnt.item()
            self.accumulated_rel_l1_distance = tensor_accum.item()

        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)

        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            
            if dist.is_initialized():
                tensor_accum = torch.tensor(self.accumulated_rel_l1_distance, device=device)
                dist.broadcast(tensor_accum, src=0)
                self.accumulated_rel_l1_distance = tensor_accum.item()

        if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
            should_calc = False
        else:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
                
        should_calc_tensor = torch.tensor(should_calc, device=hidden_states.device)
        if dist.is_initialized():
            dist.broadcast(should_calc_tensor, src=0)
        should_calc = should_calc_tensor.item()

        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        tensor_cnt = torch.tensor(self.cnt, device=device)
        if dist.is_initialized():
            dist.broadcast(tensor_cnt, src=0)

        if self.cnt == self.num_steps:
            self.cnt = 0   
            
    if dist.is_initialized():
        dist.barrier()
    print(f'2'*20)
    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
            print(f'3'*20)
        else:
            print(f'4'*20)
            ori_hidden_states = hidden_states.clone()
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )
            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            self.previous_residual = hidden_states - ori_hidden_states
            print(f'5'*20)
    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )
    print(f'6'*20)
    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)



def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for HunyuanVideo"

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        revision="refs/pr/18",
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        transformer=transformer,
        torch_dtype=torch.float16,
        revision="refs/pr/18",
    )

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_video_input_parameters(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )
    

    parallelize_transformer(pipe)
    
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling(
            # Make it runnable on GPUs with 48GB memory
            # tile_sample_min_height=128,
            # tile_sample_stride_height=96,
            # tile_sample_min_width=128,
            # tile_sample_stride_width=96,
            # tile_sample_min_num_frames=32,
            # tile_sample_stride_num_frames=24,
        )

    if args.enable_slicing:
        pipe.vae.enable_slicing()


    parameter_peak_memory = torch.cuda.max_memory_allocated(
        device=f"cuda:{local_rank}")

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer,
                                         mode="max-autotune-no-cudagraphs")

        # one step to warmup the torch compiler
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            num_frames=input_config.num_frames,
            prompt=input_config.prompt,
            num_inference_steps=1,
            generator=torch.Generator(device="cuda").manual_seed(
                input_config.seed),
        ).frames[0]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = input_config.num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = 0.15 # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.forward = teacache_forward
    pipe.transformer.time_text_embed.__class__.forward = timestep_forward
    
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(
            input_config.seed),
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"results/hunyuan_video_{parallel_info}_{resolution}.mp4"
        export_to_video(output, output_filename, fps=15)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9} GB"
        )
    get_runtime_state().destory_distributed_env()


# mkdir -p results && torchrun --nproc_per_node=2 examples/hunyuan_video_usp_example.py --model tencent/HunyuanVideo --ulysses_degree 2 --num_inference_steps 30 --warmup_steps 0 --prompt "A cat walks on the grass, realistic" --height 320 --width 512 --num_frames 61 --enable_tiling --enable_model_cpu_offload
# mkdir -p results && torchrun --nproc_per_node=2 examples/hunyuan_video_usp_example.py --model tencent/HunyuanVideo --ulysses_degree 2 --num_inference_steps 30 --warmup_steps 0 --prompt "A cat walks on the grass, realistic" --height 544 --width 960 --num_frames 129 --enable_tiling --enable_model_cpu_offload
if __name__ == "__main__":
    main()
