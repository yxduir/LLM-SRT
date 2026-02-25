## Download Model 
Encoder | Adapter | LLM 
|---|---|---
[whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/smt) | [GemmaX2](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1) 
```
cd models/

hf download yxdu/smt --local-dir smt
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
hf download ModelSpace/GemmaX2-28-2B-v0.1 --local-dir GemmaX2-28-2B-v0.1

cd ..
```

## Download Data
```
cd data/

bash multi30k_smt.sh

cd ..
```

## Infer Demo
This is a script for the Multi30k dataset from English (eng) to Chinese (cmn).
```
bash scripts/infer_9b_smt.sh
```