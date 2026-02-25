import json
import os
from vllm import LLM, SamplingParams
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()
import torch
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from transformers import AutoProcessor, AutoTokenizer,LogitsProcessor
from vllm.sampling_params import BeamSearchParams
import csv
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

src_lang = ['eng']

# src_lang = ['eng']
language_9 = ["cmn","eng","ara","ind","tur","tha","jpn","kor","vie"]


tgt_lang = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho']
language_28 = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'cmn']

src_lang = language_9
tgt_lang = language_28

input_file = "/mgData2/yxdu/data/fleurs_all/data/srt_test.jsonl"



model_vllm = "../models/GemmaX2-28-9B-v0.1"


tokenizer = AutoTokenizer.from_pretrained(model_vllm)
mode = "mt"



import pycountry
def clean_language_name(name):
    """使用正则表达式移除语言名称中所有圆括号及其内容，并去除多余的首尾空格。"""
    # 替换匹配到的模式（例如 '(1453-)', '(Kenya)'）为空字符串
    cleaned_name = re.sub(r'\s*\(.*\)', '', name).strip()
    return cleaned_name

# 获取所有语言，并对名称进行清洗
LANGUAGE_MAPPING = {
    lang.alpha_3: clean_language_name(lang.name)
    for lang in pycountry.languages
    if hasattr(lang, 'alpha_3')
}



def Prompt_template(query, src_language, trg_language):
    instruction = f'Translate the following sentences from {src_language} to {trg_language}.'
    prompt = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt

class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, prompt_tokens_ids: tuple, past_tokens_ids: tuple, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        :ref: https://github.com/vllm-project/vllm/blob/911c8eb0000b1f9d1fef99ac9e209f83d801bd0a/vllm/model_executor/layers/logits_processor.py#L186
        """
        # score: [B, vocab_size]
        # input_ids: [B, cur_len]
        input_ids = prompt_tokens_ids + past_tokens_ids
        if len(input_ids) < self.ngram_size:
            return scores

        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        num_batch_hypotheses = scores.shape[0]
        input_ids = torch.LongTensor(input_ids).reshape(num_batch_hypotheses, -1)
        cur_len = input_ids.shape[-1]
        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed

def process_jsonl(input_file, output_file,llm,model_vllm):
# 读取所有行并预处理
    batch_size=4096*4096
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = []
        for line in infile:
            # 提取 prompt[2:5] 部分，并检查是否是 "eng" 或 "zho"
            data = json.loads(line)
            prompt = data.get("prompt")

            if tgt_lang == "all":
                if prompt[2:5] in src_lang:
                    lines.append(line)
            elif prompt[2:5] in src_lang and prompt[9:12] in tgt_lang and src_lang!=tgt_lang:
                lines.append(line)
        count = 1

        sampling_params = SamplingParams(max_tokens=512, top_k=1, top_p=1,temperature=0,n=1)

        results = []  # 用于存储所有处理结果

        # 分批处理
        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i:i + batch_size]
            batch_data = [json.loads(line) for line in batch_lines]
            
            # 准备批量输入
            templates = []
            src_texts = []
            sources = []
            for data in batch_data:
                gt = data.get('gt', '')
                prompt = data.get('prompt', '')
                # source = data.get('source', '')
                
                # 提取语言和原文
                source_lang, target_lang = prompt[2:5], prompt[9:12]

                src_text = gt.split(prompt)[0]

                # Gemma3翻译指令
                translation_prompt = (
                    f"Only return in {LANGUAGE_MAPPING[target_lang]}. Translate this from {LANGUAGE_MAPPING[source_lang]} to {LANGUAGE_MAPPING[target_lang]}:\n"
                    f"{LANGUAGE_MAPPING[source_lang]}: {src_text}\n"
                    f"{LANGUAGE_MAPPING[target_lang]}:"
                )
                
                
                # GemmaX2-28-9B构造翻译指令
                if "GemmaX2-28-9B" in model_vllm:
                    translation_prompt = (
                        f"""Translate this from {LANGUAGE_MAPPING[source_lang]} to {LANGUAGE_MAPPING[target_lang]}:\n
                        {LANGUAGE_MAPPING[source_lang]}: {src_text}\n
                        {LANGUAGE_MAPPING[target_lang]}:"""
                    )
                if "gemma-3" in model_vllm:
                    translation_prompt = (f"""<bos><start_of_turn>user
                        You are a skilled translator who only outputs translations without any explanations or repetitions.\n
                        Translate the following sentences from {source_lang} to {target_lang}: {src_text}<end_of_turn>
                        <start_of_turn>model""")
                
                # llama 翻译指令
                if "llama" in model_vllm:
                    translation_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    You are a helpful translator, only return the translation result, without other message.<|eot_id|><|start_header_id|>user<|end_header_id|>
                    Translate this from {LANGUAGE_MAPPING[source_lang]} to {LANGUAGE_MAPPING[target_lang]}{src_text}\n:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

                #  #qwen 翻译指令Í
                if "Qwen2.5" in model_vllm:
                    translation_prompt = f"""<|im_start|>system
                        You are a skilled translator who only outputs translations without any explanations or additional information.<|im_end|>
                        <|im_start|>user
                        Translate the following {LANGUAGE_MAPPING[source_lang]} sentence into {LANGUAGE_MAPPING[target_lang]}: {src_text}<|im_end|>
                        <|im_start|>assistant"""
                if "Qwen3" in model_vllm:
                    if "Qwen3-Next" in model_vllm:
                        translation_prompt = f"""
                            <|im_start|>user
                            You are a skilled translator who only outputs translations without any explanations or additional information.
                            Translate the following {LANGUAGE_MAPPING[source_lang]} sentence into {LANGUAGE_MAPPING[target_lang]}: {src_text}.<|im_end|>
                            <|im_start|>assistant\n"""
                    else:
                        translation_prompt = f"""
                            <|im_start|>user
                            You are a skilled translator who only outputs translations without any explanations or additional information.
                            Translate the following {LANGUAGE_MAPPING[source_lang]} sentence into {LANGUAGE_MAPPING[target_lang]}: {src_text}.<|im_end|>
                            <|im_start|>assistant
                            <think>

                            </think>
                            """

                if "LLaMAX3" in model_vllm:
                    translation_prompt = Prompt_template(src_text, LANGUAGE_MAPPING[source_lang], LANGUAGE_MAPPING[target_lang])
     
                
                templates.append(translation_prompt)
                src_texts.append(src_text)
                # sources.append(source)
            
            # 批量推理
            outputs = llm.generate(templates, sampling_params)
            
            # 处理输出并写入文件
            for j, output in enumerate(outputs):
                if "Qwen3" in model_vllm:
                    translated_text = output.outputs[0].text.split("</think>")[-1].replace("\n","")
                if "gpt-oss" in model_vllm:
                    translated_text = output.outputs[0].text.split("assistantfinal")[-1].replace("\n","")
                else:
                    translated_text = output.outputs[0].text.replace("\n","")
                original_data = batch_data[j]
                
                new_data = {
                    "gt": original_data["gt"],
                    "prompt": original_data["prompt"],
                    # "source": sources[j],
                    "response": f"{src_texts[j]}{original_data['prompt']}{translated_text}"
                }
                
                results.append(new_data)
                print(f"Processed {count}: {translated_text}")  # 打印前50字符避免刷屏
                count += 1
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = gpu_count if gpu_count > 0 else 1



    llm = LLM(model_vllm, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=tensor_parallel_size)
    # 修改 process_jsonl 函数的调用部分


    # 使用 os.path 模块处理路径
    base_name = os.path.splitext(os.path.basename(input_file))[0]  # 获取文件名（不含路径和扩展名）
    output_file = f"{base_name}_vlm_{mode}_{model_vllm.split('/')[-1]}.jsonl"  # 输出文件路径

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    process_jsonl(input_file, output_file,llm,model_vllm)



