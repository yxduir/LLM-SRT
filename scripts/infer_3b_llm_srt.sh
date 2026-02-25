export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

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
code=$(realpath "$current_dir/../SLAM-LLM")
echo "Code path: ${code}"
cd ${code}

# 设置路径
beam=5

validnum=-1

encoder_path_hf=${code}/../models/whisper-large-v3
ckpt_name=${code}/../models/llm-srt/llm_srt_3b.pt
llm_path=${code}/../models/Qwen2.5-3B


val_data_path=${code}/../data/fleurs_eng_test_s2tt/fleurs_eng_test_s2tt.jsonl
file_name=$(basename "$val_data_path")



echo "${val_data_path}"
mode=srt
source=engcmn


peft=false
freeze_llm="true"
encoder_projector=q-former
query_len=80
encoder_projector_ds_rate=5
fix_length_audio=80
use_api=false
llm_dim=2048






echo "找到的最新 .pt 文件为: $ckpt_name"
# 设置 decode log 路径
decode_log=${code}/../output/${mode}_${source}_${encoder_projector}_${file_name}
echo "Decode log saved to: ${decode_log}"

# --- 执行推理任务 ---
echo "开始执行推理任务..."



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
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.query_len=$query_len \
    ++model_config.encoder_name=whisper \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_path_hf=$encoder_path_hf \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_projector=$encoder_projector \
    ++model_config.beam=$beam \
    ++dataset_config.dataset=st_dataset \
    ++dataset_config.file=${code}/examples/st_covost2/dataset/st_dataset.py:get_speech_dataset \
    ++dataset_config.val_data_path=$val_data_path \
    ++dataset_config.input_type=mel \
    ++dataset_config.fix_length_audio=$fix_length_audio \
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
    ++train_config.val_batch_size=64 \
    ++train_config.num_workers_dataloader=32 \
    ++log_config.decode_log=$decode_log \
    ++ckpt_path=$ckpt_name \
    ++train_config.use_api=${use_api} \
    ++train_config.use_peft=${peft} 
done
