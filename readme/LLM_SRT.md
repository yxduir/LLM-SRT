## Download Model 
Encoder | Adapter | LLM 
|---|---|---
[whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/llm-srt) | [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) 
```
cd models/

hf download yxdu/llm-srt --local-dir llm-srt
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
hf download Qwen/Qwen2.5-3B --local-dir Qwen2.5-3B

cd ..
```

## Download Data
```
cd data/

bash demo_llm_srt.sh

cd ..
```

## Infer Demo
This is a script for the fleurs dataset from English (eng) to Chinese (cmn).
```
bash scripts/infer_3b_llm_srt.sh
```