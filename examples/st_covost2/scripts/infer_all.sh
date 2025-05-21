export MASTER_ADDR=localhost
# export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=12348
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5



#  GPU num
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
elif command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=1  # 默认值
fi

echo "GPU number: $gpu_count"

# get the current script's directory
current_script=$(readlink -f "$0")
current_dir=$(dirname "$current_script")
code=$(realpath "$current_dir/../../../../LLM-SRT")
echo "Code path: ${code}"
cd ${code}
source=wmt24_all
mode=srt
validnum=-2

peft=true
if [ "$peft" = "true" ]; then
    freeze_llm="false"
else
    freeze_llm="true"
fi


checkpoint_dir=${code}/models/llm-srt/qwen2.5-3b.pt
encoder_path_hf=/data_a100/models/whisper-large-v3
llm_path=/data_a100/models/Qwen2.5-3B
llm_name=$(basename "$llm_path")

val_data_path=${code}/data/fleurs/wavs/test.jsonl


# decode_log file path
val_data_dir=$(dirname "${val_data_path}")
val_data_basename=$(basename "${val_data_path}" .jsonl)
decode_log="${code}/${val_data_basename}_${mode}_${llm_name}.jsonl"


# max_epoch=$(ls -d ${checkpoint_dir}/asr_epoch_*_step_* | sed -n 's/.*asr_epoch_\([0-9]*\)_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)
# max_step=$(ls -d ${checkpoint_dir}/asr_epoch_${max_epoch}_step_* | sed -n 's/.*asr_epoch_[0-9]*_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)

# final_path="${checkpoint_dir}/asr_epoch_${max_epoch}_step_${max_step}"
# ckpt_name=$final_path/model.pt

ckpt_name=$checkpoint_dir

echo "find .pt file: $ckpt_name"



echo "Decode log saved to: ${decode_log}"
if [ "$gpu_count" -gt 1 ]; then
    enable_ddp=true
else
    enable_ddp=false
fi

# Inference
torchrun \
    --nnodes 1 \
    --nproc_per_node ${gpu_count} \
    --master_port=29508 \
    ${code}/examples/st_covost2/inference_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++fsdp_config.pure_bf16=true \
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=2048 \
    ++model_config.query_len=80 \
    ++model_config.encoder_name=whisper \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_path_hf=$encoder_path_hf \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_projector=q-former \
    ++dataset_config.dataset=st_dataset \
    ++dataset_config.file=examples/st_covost2/dataset/srt_dataset.py:get_speech_dataset \
    ++dataset_config.val_data_path=$val_data_path \
    ++dataset_config.input_type=mel \
    ++dataset_config.fix_length_audio=80 \
    ++dataset_config.mel_size=128 \
    ++dataset_config.inference_mode=true \
    ++dataset_config.source=$source \
    ++dataset_config.mode=$mode \
    ++dataset_config.validnum=$validnum \
    ++train_config.model_name=asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=$freeze_llm \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=8 \
    ++train_config.enable_ddp=$enable_ddp \
    ++train_config.num_workers_dataloader=16 \
    ++train_config.enable_ddp=$enable_ddp \
    ++log_config.decode_log=$decode_log \
    ++ckpt_path=$ckpt_name \
    ++train_config.use_peft=${peft} \
