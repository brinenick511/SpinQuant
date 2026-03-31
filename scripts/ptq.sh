# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

MODEL="${HOME}/models/meta-llama/Llama-3.2-1B-Instruct"
MODEL="${HOME}/models/meta-llama/Llama-3.2-3B-Instruct"
# MODEL="${HOME}/models/meta-llama/Llama-3.1-8B-Instruct"
BITS=$1
# LR=1.5

CUDA_VISIBLE_DEVICES=0,5 torchrun --nnodes=1 --nproc_per_node=2 --master_port=29511 ptq.py \
--input_model "${MODEL}"  \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 2 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits "${BITS}" \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--w_asym \
--nsamples 128 \
--rotate \
--optimized_rotation_path "outputs/w${BITS}/rot/R.bin" \
--gptq_batch_size 8 \
--save_qmodel_path "outputs/l1-w${BITS}-node2-bs4-step100.pt"

sleep 1s
