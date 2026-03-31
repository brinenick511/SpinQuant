# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

MODEL="${HOME}/models/meta-llama/Llama-3.2-1B-Instruct"
MODEL="${HOME}/models/meta-llama/Llama-3.2-3B-Instruct"
# MODEL="${HOME}/models/meta-llama/Llama-3.1-8B-Instruct"
BITS=$1
LR=1.5

CUDA_VISIBLE_DEVICES=0,5 torchrun --nnodes=1 --nproc_per_node=2 --master_port=29510 optimize_rotation.py \
--input_model "${MODEL}"  \
--output_rotation_path "outputs/l3w${BITS}/rot" \
--output_dir "outputs/l3w${BITS}/" \
--logging_dir "outputs/logs/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 4 \
--logging_steps 1 \
--learning_rate "${LR}" \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits "${BITS}" \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--w_asym \
--nsamples 128 \

sleep 1s
