export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
# export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1,2,3,4
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

echo "GPU number: $gpu_count"

current_script=$(readlink -f "$0")
current_dir=$(dirname "$current_script")
code=$(realpath "$current_dir/../../../../SLAM-LLM")
cd ${code}
source=en_x

checkpoint_dir=${code}/models/output/asr-3B-encoder-15lang-lora-2
output_dir=${code}/models/output/asr-covost2-en_zh

encoder_path_hf=${code}/models/whisper-large-v3
llm_path=${code}/models/Qwen2.5-3B



max_epoch=$(ls -d ${checkpoint_dir}/asr_epoch_*_step_* | sed -n 's/.*asr_epoch_\([0-9]*\)_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)
max_step=$(ls -d ${checkpoint_dir}/asr_epoch_${max_epoch}_step_* | sed -n 's/.*asr_epoch_[0-9]*_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)

# 构建最终的路径
final_path="${checkpoint_dir}/asr_epoch_${max_epoch}_step_${max_step}"
ckpt_name=$final_path/model.pt

# 打印找到的 ckpt 文件
echo "找到的最新 .pt 文件为: $ckpt_name"




hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=Qwen \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2048 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_path_hf=$encoder_path_hf \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=q-former \
++model_config.query_len=80 \
++dataset_config.dataset=st_dataset \
++dataset_config.file=examples/st_covost2/dataset/hf_covost2.py:get_speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128  \
++dataset_config.fix_length_audio=80 \
++dataset_config.source=$source \
++train_config.model_name=asr \
++train_config.num_epochs=10 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.gradient_accumulation_steps=1 \
++train_config.warmup_steps=500 \
++train_config.total_steps=100000 \
++train_config.lr=1e-5 \
++train_config.batch_size_training=24 \
++train_config.val_batch_size=48 \
++train_config.num_workers_dataloader=16 \
++train_config.output_dir=$output_dir \
++metric=acc \
++train_config.use_fp16=false \
++ckpt_path=$ckpt_name \
"



torchrun \
    --nnodes 1 \
    --nproc_per_node ${gpu_count} \
    --master_port=29504 \
    ${code}/examples/st_covost2/finetune_asr.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++fsdp_config.pure_bf16=true \
    ++log_config.use_wandb=false \
    ++log_config.wandb_project_name=srt \
    ++log_config.wandb_exp_name=${source} \
    ++train_config.validation_interval=500 \
    ++log_config.wandb_exp_name=all \
    ++train_config.use_peft=true \
    $hydra_args
fi
        