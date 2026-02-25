# 定义模型列表
models=(
    "openai/whisper-large-v3"
    "ModelSpace/GemmaX2-28-9B-v0.1"
    "yxdu/mcat-large"
    "yxdu/mcat-small"
    "google/gemma-3-27b-it"
    "Unbabel/wmt22-comet-da"
)

# 循环下载
for model_config in "${models[@]}"; do
    model_dir=${model_config##*/}
    
    echo "--------------------------------------------------"
    echo "正在开始下载: $model_config -> 目录: $model_dir"
    echo "--------------------------------------------------"
    
    hf download "$model_config" --local-dir "$model_dir"
    
    if [ $? -eq 0 ]; then
        echo "✅ 下载完成: $model_config"
    else
        echo "❌ 下载失败: $model_config"
    fi
done