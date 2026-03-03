# Install
```
pip install torch transformers datasets tqdm sacrebleu
```

## Demo 
```
import torch, json
from tqdm import tqdm
from transformers import AutoModel
from datasets import load_dataset
from sacrebleu.metrics import BLEU

# --- 配置与加载 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
m_path, d_name, cfg = "yxdu/smt-9b-hf", "yxdu/multi30k_tts_test", "test_2016_flickr"
model = AutoModel.from_pretrained(m_path, trust_remote_code=True).to(device, torch.bfloat16).eval()
ds = load_dataset(d_name, split=cfg)
bleu = BLEU(tokenize="13a")
batch_size = 64
res_map, all_data = {}, []

# --- 推理 ---
with torch.inference_mode():
    for i in tqdm(range(0, len(ds), batch_size), desc="Inference"):
        b = ds[i : i + batch_size]
        # 拼接 ASR 和 Prompt
        prompts = [a + p for a, p in zip(b["asr"], b["prompt"])]
        # 批量翻译
        outs = model.translate_batch(b["audio"], prompts, max_new_tokens=200)
        tqdm.write(f"\n[Batch {i//batch_size+1} | Sample 0]\nInput: {prompts[0]}\nOutput: {outs[0]}\n" + "-"*30)
        
        for j, out in enumerate(outs):
            # 记录数据
            item = {k: b[k][j] for k in ["id", "asr", "s2tt", "prompt", "source"]}
            item["response"] = out
            all_data.append(item)
            # 按语种对分类用于计算 BLEU
            p = item["prompt"]
            pair = (p[2:5], p[9:12])
            res_map.setdefault(pair, [[], []])
            res_map[pair][0].append(out)
            res_map[pair][1].append(item["s2tt"])

# --- 保存与评估 ---
with open("results.jsonl", "w", encoding="utf-8") as f:
    for d in all_data: f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"\n{'Pair':<12} | {'BLEU':<6} | {'Count'}\n" + "-"*30)
for (s, t), (hyps, refs) in res_map.items():
    score = bleu.corpus_score(hyps, [refs]).score
    print(f"{s}->{t:<7} | {score:<6.2f} | {len(hyps)}")
```