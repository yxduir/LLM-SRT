export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=1,2,3,4

# 设置 GPU 数量
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
elif command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=1  # 默认值
fi

echo "GPU number: $gpu_count"

# 获取脚本路径
current_script=$(readlink -f "$0")
current_dir=$(dirname "$current_script")
code=$(realpath "$current_dir/../../../../")

echo "Code path: ${code}"
cd ${code}/SLAM-LLM

# 设置路径

checkpoint_dir=${code}/data/output/mmt-3B-3
output_dir=${code}/data/output/asr-3B-encoder-15lang-srt-3
encoder_path_hf=${code}/models/whisper-large-v3
llm_path=${code}/models/Qwen2.5-3B
speech_encoder_path=${code}/models/whisper-large-v3  # 根据实际情况设置路径

# 数据路径
train_data_path=${code}/SLAM-LLM/examples/st_covost2/mmt-all/fleurs_data/wavs/train_500.jsonl
val_data_path=${code}/SLAM-LLM/examples/st_covost2/mmt-all/fleurs_data/wavs/filter_test.jsonl

train_data_path=${code}/data/common/19/train.jsonl
val_data_path=${code}/SLAM-LLM/examples/st_covost2/mmt-all/fleurs/en/mmt_test.jsonl


# 语言列表
SOURCES=('eng' 'deu' 'fra' 'spa' 'por' 'ita' 'nld' 'rus' 'jpn' 'kor' 'vie' 'ind' 'tha' 'zho' 'yue')
TARGETS=('eng' 'deu' 'fra' 'spa' 'por' 'ita' 'nld' 'rus' 'jpn' 'kor' 'vie' 'ind' 'tha' 'zho' 'yue')

SOURCES=('eng')
TARGETS=('eng' 'deu' 'fra' 'rus' 'jpn' 'zho' )

# 获取最新的 checkpoint 路径
max_epoch=$(ls -d ${checkpoint_dir}/asr_epoch_*_step_* 2>/dev/null | sed -n 's/.*asr_epoch_\([0-9]*\)_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)
max_step=$(ls -d ${checkpoint_dir}/asr_epoch_${max_epoch}_step_* 2>/dev/null | sed -n 's/.*asr_epoch_[0-9]*_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)

if [ -z "$max_epoch" ] || [ -z "$max_step" ]; then
    echo "Error: No checkpoint found in ${checkpoint_dir}"
    exit 1
fi

final_path="${checkpoint_dir}/asr_epoch_${max_epoch}_step_${max_step}"
ckpt_path=$final_path/model.pt

# 遍历所有源语言和目标语言组合
for src in "${SOURCES[@]}"; do
  for tgt in "${TARGETS[@]}"; do
    # 跳过源语言和目标语言相同的情况
    if [ "$src" == "$tgt" ]; then
      continue
    fi

    source=fleurs_${src}_${tgt}
    echo "Processing source language: $src, target language: $tgt"
    echo "Source language is $source"

    # 设置 decode log 路径
    decode_log=$final_path/${source}.jsonl
    echo "Decode log saved to: ${decode_log}"

    # 如果 decode_log 文件已存在，则跳过当前语言组合
    if [ -f "$decode_log" ]; then
      echo "Decode log already exists. Skipping ${source}."
      continue
    fi

    # 执行推理任务
    torchrun \
        --nnodes 1 \
        --nproc_per_node ${gpu_count} \
        --master_port=29503 \
        ${code}/SLAM-LLM/examples/st_covost2/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++fsdp_config.pure_bf16=true \
        ++model_config.llm_name="Qwen2-7B" \
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
        ++dataset_config.file=examples/st_covost2/dataset/st_dataset.py:get_speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.fix_length_audio=80 \
        ++dataset_config.mel_size=128 \
        ++dataset_config.inference_mode=true \
        ++dataset_config.source=$source \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=16 \
        ++train_config.num_workers_dataloader=16 \
        ++log_config.decode_log=$decode_log \
        ++ckpt_path=$ckpt_path
  done
done