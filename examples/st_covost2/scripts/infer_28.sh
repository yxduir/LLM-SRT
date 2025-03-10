export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,3,4

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
code=$(realpath "$current_dir/../../../../SLAM-LLM")
echo "Code path: ${code}"
cd ${code}

# 设置路径
beam=5
mode=srt
validnum=-2

peft=true
# 根据 peft 的值设置 freeze_llm 的相反值
if [ "$peft" = "true" ]; then
    freeze_llm="false"
else
    freeze_llm="true"
fi
# /mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/models/output/asr-3B-encoder-15lang-lora-2
# /mgData3/zhaozhiyuan/vits/hit/code/data/qwen/asr-Qwen2.5_7B-3
checkpoint_dir=${code}/models/output/asr-7B-mi-28lang-srt-lora-2
# checkpoint_dir=/mgData3/zhaozhiyuan/vits/hit/code/data/qwen/srt-6-12-main-7

output_dir=${code}/models/output/qwen2.5-srt-15lang

encoder_path_hf=${code}/models/whisper-large-v3
llm_path=${code}/models/GemmaX2-28-9B-v0.1

train_data_path=${code}/data/fleurs/wavs/train_300_30.jsonl
val_data_path=${code}/data/fleurs/wavs/test_30.jsonl


# 语言列表
SOURCES=('eng' 'deu' 'fra' 'spa' 'por' 'ita' 'nld' 'rus' 'jpn' 'kor' 'vie' 'ind' 'tha' 'zho' 'ces' 'pol' 'ara' 'fas' 'heb' 'tur' 'msa' 'lao' 'mya' 'khm' 'tgl' 'hin' 'ben' 'urd')

# 获取最新的 checkpoint 路径
max_epoch=$(ls -d ${checkpoint_dir}/asr_epoch_*_step_* 2>/dev/null | sed -n 's/.*asr_epoch_\([0-9]*\)_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)
max_step=$(ls -d ${checkpoint_dir}/asr_epoch_${max_epoch}_step_* 2>/dev/null | sed -n 's/.*asr_epoch_[0-9]*_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)

if [ -z "$max_epoch" ] || [ -z "$max_step" ]; then
    echo "Error: No checkpoint found in ${checkpoint_dir}"
    exit 1
fi

final_path="${checkpoint_dir}/asr_epoch_${max_epoch}_step_${max_step}"
ckpt_name=$final_path/model.pt
echo "找到的最新 .pt 文件为: $ckpt_name"

# 遍历所有源语言和目标语言组合
for src in "${SOURCES[@]}"; do

  source=fleurs_${src}
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
      ${code}/examples/st_covost2/inference_asr_batch.py \
      --config-path "conf" \
      --config-name "prompt.yaml" \
      ++train_config.enable_fsdp=false \
      ++train_config.enable_ddp=true \
      ++fsdp_config.pure_bf16=true \
      ++model_config.llm_name="Qwen2.5-7B" \
      ++model_config.llm_path=$llm_path \
      ++model_config.llm_dim=3584 \
      ++model_config.query_len=80 \
      ++model_config.encoder_name=whisper \
      ++model_config.encoder_projector_ds_rate=5 \
      ++model_config.encoder_path=$speech_encoder_path \
      ++model_config.encoder_path_hf=$encoder_path_hf \
      ++model_config.encoder_dim=1280 \
      ++model_config.encoder_projector=q-former \
      ++model_config.beam=$beam \
      ++dataset_config.dataset=st_dataset \
      ++dataset_config.file=examples/st_covost2/dataset/st_dataset.py:get_speech_dataset \
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
      ++train_config.val_batch_size=32 \
      ++train_config.num_workers_dataloader=16 \
      ++log_config.decode_log=$decode_log \
      ++ckpt_path=$ckpt_name \
      ++train_config.use_peft=${peft} 
  done
done