import os
import json
import csv
from collections import defaultdict
from multiprocessing import Process, Queue, current_process
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import torch
from sacrebleu.metrics import BLEU
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from comet import load_from_checkpoint
import jiwer  # New requirement


# -------------------- Parameters & Setup ----------------------

file_path = "../output/srt_engcmn_q-former_fleurs_test_eng.jsonl"
file_path = "../output/smt_all_q-former_test_2016_flickr.jsonl"
gpus = [0]
batch_size = 64

normalizer = BasicTextNormalizer()

# Languages requiring Character Error Rate (no spaces)
CER_LANGS = {'jpn', 'kor', 'tha', 'cmn', 'yue'} 

language_9  = ["cmn","eng","ara","ind","tur","tha","jpn","kor","vie"]
language_11  = ["cmn","eng","ara","ind","tur","tha","jpn","kor","vie","khm","lao","mya"]
language_28 = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'cmn']
language_70 = ['afr', 'amh', 'ara', 'asm', 'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 'ces', 'cmn', 'cym', 'dan', 'deu', 'ell', 'eng', 'est', 'fas', 'fin', 'fra', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hun', 'hye', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kan', 'kat', 'kaz', 'khm', 'kir', 'kor', 'lao', 'lav', 'lit', 'mal', 'mkd', 'msa', 'mya', 'nld', 'nob', 'npi', 'pan', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'ukr', 'urd', 'uzb', 'vie', 'yue']
src_langs = language_9
tgt_langs = language_28

# Updated metrics list
test_metrics = ["idx", "iso3", "spbleu", "comet", "wer_cer"]

# -------------------- Data Loading ----------------------
lang_groups = defaultdict(lambda: defaultdict(lambda: {
    "asr_gt": [], "asr_re": [], "st_gt": [], "st_re": []
}))

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        gt = data.get("gt","")
        prompt = data.get("prompt","")
        response = data.get("response","") or data.get("s2tt","")
        
        if "Error" in response: continue

        src_lang = prompt.split("|>")[0].split("<|")[-1]
        tgt_lang = prompt.split("<|")[-1].split("|>")[0]
        if src_lang == tgt_lang or src_lang not in src_langs: continue

        prompt_tag = f"<|{src_lang}|><|{tgt_lang}|>"
        split_res = response.split(prompt_tag)
        split_gt = gt.split(prompt_tag)

        asr_gt = split_gt[0] if len(split_gt)==2 else gt
        st_gt = split_gt[1] if len(split_gt)==2 else gt
        asr_re = split_res[0] if len(split_res)==2 else response
        st_re = split_res[1] if len(split_res)==2 else response.split("|>")[-1]

        lang_groups[src_lang][tgt_lang]["asr_gt"].append(asr_gt.strip())
        lang_groups[src_lang][tgt_lang]["asr_re"].append(asr_re.strip())
        lang_groups[src_lang][tgt_lang]["st_gt"].append(st_gt.strip())
        lang_groups[src_lang][tgt_lang]["st_re"].append(st_re.strip())

tasks = []
for src_lang in sorted(lang_groups.keys()):
    for tgt_lang in sorted(lang_groups[src_lang].keys()):
        d = lang_groups[src_lang][tgt_lang]
        tasks.append((src_lang, tgt_lang, d["asr_gt"], d["asr_re"], d["st_re"], d["st_gt"]))

chunk_size = len(tasks) // len(gpus)
task_chunks = [tasks[i*chunk_size:(i+1)*chunk_size] for i in range(len(gpus)-1)]
task_chunks.append(tasks[(len(gpus)-1)*chunk_size:])

# -------------------- Worker Function ----------------------
def worker(gpu, task_chunk, output_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda:0")
    comet_model = load_from_checkpoint("../models/wmt22-comet-da/checkpoints/model.ckpt").half().to(device)

    results = []
    for src_lang, tgt_lang, asr_gt, asr_re, st_re, st_gt in task_chunk:
        # 1. Calculate ASR metric (WER or CER)
        # Normalize and filter out empty strings to avoid jiwer errors
        norm_gt = [normalizer(text) for text in asr_gt]
        norm_re = [normalizer(text) for text in asr_re]
        
        if src_lang in CER_LANGS:
            # CER: treat every character as a word by joining with spaces or using cer function
            asr_err = jiwer.cer(norm_gt, norm_re)
        else:
            # WER: standard word error rate
            asr_err = jiwer.wer(norm_gt, norm_re)

        # 2. BLEU
        cer_lang = ["tha", "jpn", "kor", "zho", "yue", "cmn","lao","mya"]
        if tgt_lang in cer_lang:
            bleu = BLEU(tokenize="char")
        else:
            bleu = BLEU(tokenize="13a")
        bleu_score = bleu.corpus_score(st_re, [st_gt]).score

        # 2. spBLEU
        spbleu = BLEU(tokenize="flores200")
        spbleu_score = spbleu.corpus_score(st_re, [st_gt]).score

        # 3. COMET
        comet_data = [{'src': s, 'mt': p, 'ref': r} for s,p,r in zip(asr_gt, st_re, st_gt)]
        comet_score = comet_model.predict(comet_data, batch_size=batch_size, devices=[0])['system_score']*100

        res = {
            "iso3": f"{src_lang}_{tgt_lang}",
            "bleu": round(bleu_score, 2),
            "spbleu": round(spbleu_score, 2),
            "comet": round(comet_score, 2),
            "wer_cer": round(asr_err * 100, 2) # Represented as %
        }
        results.append(res)
        print(f"[GPU{gpu}] {src_lang}->{tgt_lang} | BLEU: {res['bleu']} | spBLEU: {res['spbleu']} | COMET: {res['comet']} | ASR Err: {res['wer_cer']}%")
    output_queue.put(results)

# -------------------- Main Process ----------------------
if __name__ == "__main__":
    output_queue = Queue()
    processes = []

    for i, gpu in enumerate(gpus):
        p = Process(target=worker, args=(gpu, task_chunks[i], output_queue))
        p.start()
        processes.append(p)

    all_results = []
    for _ in processes:
        all_results.extend(output_queue.get())
    for p in processes:
        p.join()

    # Write CSV & Compute Averages
    output_csv = f"./csv/{file_path.split('/')[-1].split('.')[0]}_evaluated.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(test_metrics)
        
        totals = {"bleu":0, "spbleu": 0, "comet": 0, "asr": 0}
        for idx, r in enumerate(all_results):
            writer.writerow([idx+1, r["iso3"],r["bleu"], r["spbleu"], r["comet"], r["wer_cer"]])
            totals["bleu"] += r["bleu"]
            totals["spbleu"] += r["spbleu"]
            totals["comet"] += r["comet"]
            totals["asr"] += r["wer_cer"]

    n = len(all_results)
    if n > 0:
        print(f"\n✅ Results saved to {output_csv}")
        print(f"📊 Global Averages:")
        print(f"   - BLEU: {totals['bleu']/n:.2f}")
        print(f"   - spBLEU: {totals['spbleu']/n:.2f}")
        print(f"   - COMET:  {totals['comet']/n:.2f}")
        print(f"   - ASR Error (WER/CER): {totals['asr']/n:.2f}%")