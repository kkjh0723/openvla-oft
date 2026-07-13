#!/usr/bin/env bash
#SBATCH --job-name=oft-finetune-deltaI
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --gpus=16
#SBATCH --cpus-per-gpu=6
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Finetune OpenVLA-OFT *ΔI* (delta-image) with LoRA r=32 + L1 regression on the
# all_v8 12-suite LIBERO-dynamic mixture. BYTE-IDENTICAL recipe to
# finetune_2frame.sbatch; the only change is the CONTENT of the third image:
# instead of the raw t-2 primary frame, the model gets the materialized ΔI base
# channel (image_delta_primary, current step) => input [primary_t, wrist_t, ΔI_t].
# Cross-architecture transfer of the GR00T ΔI TIR (Isaac-GR00T 6b780c2).
#
# Requires the ΔI dataset built by the converter's --delta mode:
#   dynamic_all_v8_deltaI_no_noops  (12-suite mixture of *_deltaI_no_noops).
# Color-jitter is disabled on the ΔI view; its crop is shared with primary via
# the base-view seed-sharing in obs_transforms.augment (geometry aligned).
#
# Usage:
#   sbatch scripts/slurm/finetune_deltaI.sbatch
#   # or explicit:
#   SUITE=dynamic_all_v8 DATASET_NAME=dynamic_all_v8_deltaI_no_noops \
#     sbatch scripts/slurm/finetune_deltaI.sbatch
#
# Differences vs finetune_2frame.sbatch:
#   - DATASET_NAME defaults to ${SUITE}_deltaI_no_noops (the ΔI dataset)
#   - --use_delta_image True (instead of --use_temporal_image); window stays 1
#   - RUN_NAME defaults to openvla-oft-${SUITE}_deltaI (no checkpoint collision)
# Everything else (16 GPU, BATCH_SIZE=2, global batch 32, LR, steps, LoRA r=32,
# proprio, image_aug) matches the 2-frame recipe so the comparison is controlled.

set -x -e

REPO=/home/jinhyung_kim/projects/async-groot/openvla-oft
cd "$REPO"
mkdir -p slurm_logs output

source .venv/bin/activate

rm -rf "$HOME/.cache/huggingface/modules/transformers_modules/47a0ec7fc4ec123775a391911046cf33cf9ed83f/__pycache__"

SUITE="${SUITE:-dynamic_all_v8}"
if [ -z "${DATA_ROOT:-}" ]; then
    case "$SUITE" in
        dynamic_*) DATA_ROOT=/home/jinhyung_kim/projects/async-groot/datasets/dynamic_libero_rlds ;;
        libero_*)  DATA_ROOT=/home/jinhyung_kim/projects/async-groot/datasets/libero_standard/modified_libero_rlds ;;
        *) echo "ERROR: cannot infer DATA_ROOT from SUITE=$SUITE"; exit 1 ;;
    esac
fi
DATASET_NAME="${DATASET_NAME:-${SUITE}_deltaI_no_noops}"
# Distinct run name so the ΔI checkpoints never collide with the 2-frame run.
RUN_NAME="${RUN_NAME:-openvla-oft-${SUITE}_deltaI}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/shared/jinhyung/checkpoints/libero-dynamic/OpenVLA-OFT}"

MAX_STEPS="${MAX_STEPS:-150005}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
# Same per-GPU batch as the 2-frame run. ΔI has the IDENTICAL token budget (still
# 3 images), so if the 3-image forward OOMs at batch=2 on A100-40GB, drop to
# BATCH_SIZE=1 and scale to 32 GPUs (2 nodes) to preserve global batch 32.
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-5e-4}"
LORA_RANK="${LORA_RANK:-32}"
# 3 images: [primary_t, wrist_t, ΔI_t].
NUM_IMAGES="${NUM_IMAGES:-3}"
USE_PROPRIO="${USE_PROPRIO:-True}"
USE_DELTA_IMAGE="${USE_DELTA_IMAGE:-True}"
WANDB_ENTITY="${WANDB_ENTITY:-lgairesearch}"
WANDB_PROJECT="${WANDB_PROJECT:-libero-vla-comparison}"

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_P2P_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC="${NPROC:-16}"
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" \
    vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$CKPT_ROOT/$RUN_NAME" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input "$NUM_IMAGES" \
    --use_delta_image "$USE_DELTA_IMAGE" \
    --use_proprio "$USE_PROPRIO" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --num_steps_before_decay "$NUM_STEPS_BEFORE_DECAY" \
    --max_steps "$MAX_STEPS" \
    --save_freq "$SAVE_FREQ" \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank "$LORA_RANK" \
    --grad_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --merge_lora_during_training False \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id_override "$RUN_NAME"
