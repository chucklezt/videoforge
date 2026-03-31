  # VideoForge                         
                                                                                               
  Video generation training pipeline for AMD RX 6800 XT (16GB, RDNA2) on Ubuntu Server with
  ROCm.                                                                                        
       
  ## Specs                                                                                     
  All project specifications are in `specs/`. Read these before starting any work:
  - `specs/00-PROJECT-OVERVIEW.md` - Architecture, tech stack, goals              
  - `specs/01-HARDWARE-AND-ENVIRONMENT.md` - ROCm setup, PyTorch, validation                   
  - `specs/02-DATA-PIPELINE.md` - Video ingestion, scene detection, clip extraction
  - `specs/03-CAPTIONING-PIPELINE.md` - Auto-captioning with Qwen2-VL                          
  - `specs/04-TRAINING-PIPELINE.md` - LoRA fine-tuning Wan 2.1 1.3B                            
  - `specs/05-INFERENCE-PIPELINE.md` - Script-to-video generation                              
  - `specs/06-MODEL-SELECTION.md` - Model comparison and rationale                             
  - `specs/07-PROJECT-STRUCTURE.md` - Directory layout, CLI, configs                           
  - `specs/08-LIMITATIONS-AND-WORKAROUNDS.md` - VRAM strategies, fallbacks                     
  - `specs/setup-videoforge.sh` - Automated Ubuntu Server setup script                         
                                                                                               
  ## Key Constraints                                                                           
  - 16GB VRAM ceiling -- use quantization, gradient checkpointing, CPU offloading              
  - AMD ROCm (RDNA2/gfx1030) -- no xformers, no CUDA kernels, use SDPA                         
  - HSA_OVERRIDE_GFX_VERSION=10.3.0 required everywhere                                        
  - ROCm already installed and working (llama.cpp, Open WebUI confirmed)

  ## Inference Constraints (validated through experiments 1-9)
  - Inference requires `enable_sequential_cpu_offload()` + `enable_attention_slicing("max")`
  - `enable_model_cpu_offload()` alone is NOT sufficient -- OOMs on SDPA attention
  - VAE must be loaded in float32; pipeline in float16
  - Wan expects 4k+1 frame counts: 17, 33, 49, 65, 81, 97, 113, 129
  - VRAM ceiling by resolution (at 60 steps, sequential offload):
    - 480x480: max 33 frames (2.1s), ~82% VRAM
    - 320x320: max 81 frames (5.1s), ~75% VRAM
    - 320x240: max 129 frames (8.1s), 93.3% VRAM (1.1GB headroom)
    - 320x240 at 150 frames: OOM
  - Prompt engineering matters more than step count for quality
  - `ftfy` package required (not in default diffusers deps)
  - Generator must be on CPU: `torch.Generator(device="cpu")` -- ROCm unreliable with cuda generator                       

