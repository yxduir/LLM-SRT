import json
from collections import defaultdict
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sacrebleu
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER
import os 
import csv
from comet import download_model, load_from_checkpoint
import os
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()
import torch
import openpyxl
from comet import download_model, load_from_checkpoint
import torch
import argparse
import nltk
from nltk.translate import meteor_score
from nltk import word_tokenize

device_num = 0

src_langs = ['ara','arz','cmn','zsm', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho','yue','oci','mon','khk','yue']
tgt_langs = ['ara','zsm', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho','yue']

file_path = "../fleurs_enzh_srt_Qwen2.5-3B.jsonl"
# file_path = "../test_srt_Qwen2.5-3B.jsonl"



test_metrics_all = ["idx","iso3","iso2","resource","bleu","spbleu","comet","meteor","xcomet","cometwiki","wer","cer"]

test_metrics = ["idx","iso3","iso2","bleu","spbleu","comet","xcomet","cometwiki"]

test_metrics = ["idx","iso3","iso2","bleu","spbleu","comet","meteor"]
test_metrics = ["idx","iso3","iso2","bleu","spbleu","comet"]



ISO3_TO_ISO2_MAPPING = {
    'ara': 'ar',  # Arabic
    'arz': 'ar',  # Arabic (Egypt)
    'ben': 'bn',  # Bengali
    'ces': 'cs',  # Czech
    'deu': 'de',  # German
    'eng': 'en',  # English
    'spa': 'es',  # Spanish
    'fas': 'fa',  # Persian
    'pes': 'fa',  # Persian
    'fra': 'fr',  # French
    'heb': 'he',  # Hebrew
    'hin': 'hi',  # Hindi
    'ind': 'id',  # Indonesian
    'ita': 'it',  # Italian
    'jpn': 'ja',  # Japanese
    'khm': 'km',  # Khmer
    'kor': 'ko',  # Korean
    'lao': 'lo',  # Lao
    'msa': 'ms',  # Malay
    'zsm': 'ms',  # Malay
    'mya': 'my',  # Burmese
    'nld': 'nl',  # Dutch
    'pol': 'pl',  # Polish
    'por': 'pt',  # Portuguese
    'rus': 'ru',  # Russian
    'tha': 'th',  # Thai
    'tgl': 'tl',  # Tagalog
    'tur': 'tr',  # Turkish
    'urd': 'ur',  # Urdu
    'vie': 'vi',  # Vietnamese
    'zho': 'zh',  # Chinese
    'cmn': 'zh',  # Mandarin Chinese
    'yue': 'ye',  # Cantonese
    'ceb': 'ce',  # Cebuan
    'oci': 'oc',  # Occitan
    'mon': 'mn',  # Mongolian
    'khk': 'mn',  # Mongolian (Khalkha)
}


ISO3_TO_LOW = {
    'ara': '1',  # Arabic
    'arz': '1',  # Arabic (Egypt)
    'ben': '2',  # Bengali
    'ces': '1',  # Czech
    'deu': '1',  # German
    'eng': '1',  # English
    'spa': '1',  # Spanish
    'fas': '1',  # Persian
    'pes': '1',  # Persian
    'fra': '1',  # French
    'heb': '2',  # Hebrew
    'hin': '1',  # Hindi
    'ind': '2',  # Indonesian
    'ita': '1',  # Italian
    'jpn': '1',  # Japanese
    'khm': '3',  # Khmer
    'kor': '1',  # Korean
    'lao': '3',  # Lao
    'msa': '2',  # Malay
    'zsm': '2',  # Malay
    'mya': '3',  # Burmese
    'nld': '1',  # Dutch
    'pol': '1',  # Polish
    'por': '1',  # Portuguese
    'rus': '1',  # Russian
    'tha': '2',  # Thai
    'tgl': '2',  # Tagalog
    'tur': '1',  # Turkish
    'urd': '2',  # Urdu
    'vie': '1',  # Vietnamese
    'zho': '1',  # Chinese
    'cmn': '1',  # Mandarin Chinese
    'yue': '1',  # Cantonese 
    'ceb': '1',  # Cebuan 
    'oci': '3',  # Occitan
    'mon': '3',  # Mongolian
    'khk': '3',  # Mongolian (Khalkha)
}


if "wer" in test_metrics:
    wer = load("wer")

if "cer" in test_metrics:
    cer = load("cer")

if "comet" in test_metrics:
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path).half()

if "xcomet" in test_metrics:
    xmodel_path = download_model("Unbabel/XCOMET-XXL")
    xcomet_model = load_from_checkpoint(xmodel_path).half()

if "cometwiki" in test_metrics:
    cometwikiz_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    cometwiki_model = load_from_checkpoint(cometwikiz_model_path).half()


lang_groups = defaultdict(lambda: defaultdict(lambda: {"asr_gt": [], "asr_re": [], "st_gt": [], "st_re": []}))

count = 0
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        gt = data.get("gt", "")
        prompt = data.get("prompt","")
        response = data.get("response", "")
        

        src_lang = gt.split("|>")[0].split("<|")[-1]
        
        tgt_lang = gt.split("<|")[-1].split("|>")[0]

        if src_lang == tgt_lang or src_lang not in src_langs or tgt_lang not in tgt_langs:
            continue
        
        prompt = f"<|{src_lang}|><|{tgt_lang}|>"
        split_responses = response.split(prompt)
        if len(split_responses) == 2:
            asr_re, st_re = split_responses
        else:
            # continue
            print(count,response)
            print(count,gt)
            count +=1 

            asr_re = response
            st_re = response.split("|>")[-1] if "|>" in response else response
            if len(st_re)==0:
                st_re = response

        lang_groups[src_lang][tgt_lang]["asr_gt"].append(gt.split(prompt)[0])
        lang_groups[src_lang][tgt_lang]["asr_re"].append(asr_re)
        lang_groups[src_lang][tgt_lang]["st_gt"].append(gt.split(prompt)[1])
        lang_groups[src_lang][tgt_lang]["st_re"].append(st_re)

results = {}
idx = 1
for src_lang in sorted(lang_groups.keys()):  
    tgt_lang_data = lang_groups[src_lang]
    for tgt_lang in sorted(tgt_lang_data.keys()): 
        data = tgt_lang_data[tgt_lang]
        
        iso3 = f"{src_lang}_{tgt_lang}"
        iso2 = f"{ISO3_TO_ISO2_MAPPING[src_lang]}_{ISO3_TO_ISO2_MAPPING[tgt_lang]}"
        sources = [s.strip() for s in data["asr_gt"]]
        asr_predictions = [a.strip() for a in data["asr_re"]]
        predictions = [p.strip() for p in data["st_re"]]
        references = [r.strip() for r in data["st_gt"]]

        cer_lang = ["tha", "jpn", "kor", "zho", "yue", "cmn","lao","mya"]

        if src_lang in cer_lang and "cer" in test_metrics:
            normalized_predictions = [normalizer(pred) for pred in asr_predictions]
            normalized_references = [normalizer(ref) for ref in sources]
            cer_score = cer.compute(predictions=normalized_predictions, references=normalized_references)*100
        else:
            cer_score = 0

        if src_lang not in cer_lang and "wer" in test_metrics:
            normalized_predictions = [normalizer(pred) for pred in asr_predictions]
            normalized_references = [normalizer(ref) for ref in sources]
            wer_score = wer.compute(predictions=normalized_predictions, references=normalized_references)*100
        else:
            wer_score = 0
        
        if "bleu" in test_metrics:
            tokenize_method = "char" if tgt_lang in ["tha", "jpn", "kor", "zho", "yue", "cmn","lao","mya"] else "13a"
            bleu = BLEU(tokenize=tokenize_method)
            bleu_score = bleu.corpus_score(predictions, [references]).score
        else:
            bleu_score = 0
        
        if "spbleu" in test_metrics:
            spbleu = BLEU(tokenize='flores200')
            spbleu_score = spbleu.corpus_score(predictions, [references]).score
        else:
            spbleu_score = 0

        if "comet" in test_metrics:
            comet_data = [{'src': s, 'mt': p, 'ref': r} for s, p, r in zip(sources, predictions, references)]
            comet_score = comet_model.predict(comet_data, batch_size=512,devices=[device_num])['system_score']*100
        else:
            comet_score = 0

        if "xcomet" in test_metrics:
            comet_data = [{'src': s, 'mt': p} for s, p in zip(sources, predictions)]
            xcomet_score = xcomet_model.predict(comet_data, batch_size=16,devices=[device_num])['system_score']*100
        else:
            xcomet_score = 0
        
        if "cometwiki" in test_metrics:
            comet_data = [{'src': s, 'mt': p} for s, p in zip(sources, predictions)]
            cometwiki_score = cometwiki_model.predict(comet_data, batch_size=16,devices=[device_num])['system_score']*100
        else:
            cometwiki_score = 0

        if "meteor" in test_metrics:
            meteor_lang = {
                "tha": "th",  
                "jpn": "ja", 
                "kor": "ko",  
                "zho": "zh",  
                "yue": "zh",  
                "cmn": "zh",  
                "lao": "lo",  
                "mya": "my",  
            }.get(tgt_lang, "english")  
            
            
            
            meteor_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = word_tokenize(pred, language=meteor_lang)
                ref_tokens = word_tokenize(ref, language=meteor_lang)
                
                score = meteor_score.meteor_score([ref_tokens], pred_tokens) * 100
                meteor_scores.append(score)
            
            meteor_value = sum(meteor_scores) / len(meteor_scores)
        else:
            meteor_value = 0
        results[idx] = {
            "idx": idx,
            "iso3":iso3,
            "iso2":iso2,
            "resource":ISO3_TO_LOW[tgt_lang],
            "bleu": round(bleu_score, 2),
            "spbleu": round(spbleu_score, 2),
            "comet": round(comet_score, 2),
            "meteor": round(meteor_value , 2),
            "xcomet": round(xcomet_score, 2),
            "cometwiki": round(cometwiki_score, 2),
            "wer": round(wer_score, 2),
            "cer": round(cer_score, 2),
        }
        print(results[idx])
        idx +=1


output_xlsx = file_path.split("/")[-1].split(".")[0] + ".xlsx"
wb = openpyxl.Workbook()
ws = wb.active
ws.append(test_metrics_all)

for key, scores in results.items():
    ws.append([scores["idx"],scores["iso3"],scores["iso2"],scores["resource"], scores["bleu"], scores["spbleu"], scores["comet"], scores["meteor"],scores["xcomet"],scores["cometwiki"],scores["wer"],scores["cer"]])

wb.save(output_xlsx)
print(f"result saved in {output_xlsx}")