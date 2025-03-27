#### conda activate audio

#### 以下为模型代码 ####

from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from transformers import PreTrainedModel,WhisperModel,AutoModelForCausalLM, AutoTokenizer,AutoTokenizer,AutoConfig,get_linear_schedule_with_warmup, set_seed
import torch.nn as nn
from typing import List, Optional
from slam_llm.utils.metric import compute_accuracy
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft import PeftConfig, get_peft_model
from peft import LoraConfig, PeftType  # 确保导入 PeftType
import whisper
import os

from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from tqdm import tqdm
import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size, print_model_size
from peft import PeftModel, PeftConfig
from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy
from transformers import SeamlessM4Tv2ForSpeechToText,SeamlessM4Tv2ForTextToText
import logging

from transformers import StoppingCriteria, StoppingCriteriaList

class EncoderProjectorQFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_dim = 1280
        self.llm_dim = 3584
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = 8
        

        self.query_len = 80
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)


        if self.llm_dim <= 1536:
            self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)
        elif self.llm_dim <= 2560:
            self.linear1 = nn.Linear(configuration.hidden_size, 1536)  
            self.relu = nn.ReLU()  
            self.linear2 = nn.Linear(1536, self.llm_dim) 
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)  
        else:
            self.linear1 = nn.Linear(configuration.hidden_size, 2560) 
            self.relu = nn.ReLU()  
            self.linear2 = nn.Linear(2560, self.llm_dim)  
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)  
        
       

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        if self.llm_dim <= 1536:
            query_proj = self.norm(self.linear(query_output.last_hidden_state))
        else:
            x = self.linear1(query_output.last_hidden_state) 
            x = self.relu(x)  
            x = self.linear2(x) 
            query_proj = self.norm(x)   

        
        return query_proj

class CustomSLM(PreTrainedModel):
    def __init__(self, config, ckpt_path=None):
        super().__init__(config)
        self.encoder = WhisperModel.from_pretrained("/mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/models/whisper-large-v3",torch_dtype=torch.float16).encoder
        self.llm = AutoModelForCausalLM.from_pretrained("/mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/models/GemmaX2-28-9B-v0.1",torch_dtype=torch.float16)
        peft_config = LoraConfig(
            peft_type=PeftType.LORA,  # 直接使用枚举值 PeftType.LORA，不需要 <...> 符号
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            target_modules=["q_proj", "v_proj"],  # 使用列表而非集合（set）
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            # 其他参数若不需要自定义，可以省略（库会使用默认值）
        )        
        self.llm = get_peft_model(self.llm, peft_config)
        self.encoder_projector = EncoderProjectorQFormer()
        self.tokenizer = AutoTokenizer.from_pretrained("/mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/models/GemmaX2-28-9B-v0.1")

        if ckpt_path is not None:
            print("loading model checkpoint from: {}".format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(ckpt_dict, strict=False)  # 

    @torch.no_grad()
    def inference(
        self,
        wav_path=None,
        prompt=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        # inference for asr model

        if torch.cuda.is_available():
            # 强制使用卡1（需确保物理卡1存在）
            device = torch.device("cuda:5")
            # 如果卡1不存在，此处会抛出 RuntimeError
        else:
            device = torch.device("cpu")
        if os.path.exists(wav_path):  # Audio-Text QA

            audio_raw = whisper.load_audio(wav_path)
            audio_raw = whisper.pad_or_trim(audio_raw)

            # 生成梅尔频谱并调整维度
            mel_size = 128
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
            # audio_mel = audio_mel.T  # (time_steps, n_mels)
            audio_mel = audio_mel.unsqueeze(0)  # (1, time_steps, n_mels)
            audio_mel = audio_mel.to(device).half()
            encoder_outs = self.encoder(audio_mel).last_hidden_state

            audio_mel_post_mask = torch.ones(
                encoder_outs.size()[:-1], dtype=torch.long
            ).to(encoder_outs.device)
            encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        prompt = prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds[None, :, :]), dim=1
        )  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )

        # generate
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model_outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=kwargs.get("max_new_tokens", 300),
                num_beams=kwargs.get("num_beams", 5),
                do_sample=kwargs.get("do_sample", False),
                min_length=kwargs.get("min_new_tokens", 10),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=kwargs.get("repetition_penalty", 1),
                length_penalty=kwargs.get("length_penalty", 1.0),
                temperature=kwargs.get("temperature", 1.0),
                no_repeat_ngram_size=5,
                early_stopping=True,
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return model_outputs
    
config = AutoConfig.from_pretrained("/mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/models/whisper-large-v3")  # CustomSLM继承了HF的父类，要求必须有一个config，无实际作用
ckpt_path="/mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/models/output/asr-7B-mi-28lang-srt-lora-2/asr_epoch_1_step_9000/model.pt"
model = CustomSLM(config,ckpt_path=ckpt_path)
print(model)
# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 强制使用卡1（需确保物理卡1存在）
    device = torch.device("cuda:5")
    # 如果卡1不存在，此处会抛出 RuntimeError
else:
    device = torch.device("cpu")
model.half().to(device)
model.eval()

#### 以下为服务器代码 ####
    
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from split_audio.split_srt_jsonl import split
import json
from pathlib import Path


app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

upload_path = "/mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/examples/st_covost2/upload/"


@app.post("/translateAudio")
async def translateAudio(source: str = Form(...), target: str = Form(...), username: str = Form(...), audio: UploadFile = File(...)):
    wav_path = "upload/" + str(username) + "_" + audio.filename

    # 保存音频文件
    with open(wav_path, "wb") as buffer:
        buffer.write(await audio.read())

    prompt = f"<|{source}|><|{target}|>"

    print("get translateAudio requesst")
    print("wav_path: ", wav_path)
    print("prompt: ", prompt)

    model_outputs = model.inference(wav_path, prompt)
    output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
    
    print("output_text: ", output_text)
    
    if prompt in output_text[0]:
        [source_text, translation_text] = output_text[0].split(prompt)
    else:
        source_text, translation_text = "", ""
        print(f"output_text is invalid, expecting {prompt} in it")

    return {"message": "success", "filename": wav_path, "source_text": source_text, "translation_text": translation_text}


@app.post("/translateAudioPath")
async def translateAudioPath(
        wav_path: str = Form(...), 
        source: str = Form(...), 
        target: str = Form(...)
    ):
    prompt = f"<|{source}|><|{target}|>"

    wav_path = Path(upload_path) / wav_path
    print("get translateAudioPath requesst")
    print("wav_path: ", wav_path)
    print("prompt: ", prompt)

    model_outputs = model.inference(wav_path, prompt)
    output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
    
    print("output_text: ", output_text)
    
    if prompt in output_text[0]:
        [source_text, translation_text] = output_text[0].split(prompt)
    else:
        source_text, translation_text = "", ""
        print(f"output_text is invalid, expecting {prompt} in it")

    return {"message": "success", "filename": wav_path, "source_text": source_text, "translation_text": translation_text}


@app.post("/splitAudio")
async def splitAudio(source: str = Form(...), target: str = Form(...), username: str = Form(...), audio: UploadFile = File(...)):
    filename = str(username) + "_" + audio.filename
    wav_path = upload_path + filename
    prompt = f"<|{source}|><|{target}|>"

    # 保存音频文件
    with open(wav_path, "wb") as buffer:
        buffer.write(await audio.read())

    # 分割音频
    (name, output_jsonl_path) = split(prompt, wav_path, upload_path)
    print(output_jsonl_path)

    # 读取JSONL文件内容
    jsonl_data = []
    try:
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    jsonl_data.append(json.loads(line))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error reading JSONL file: {str(e)}"}
        )

    return {
        "message": "success",
        "data": jsonl_data,  # 返回解析后的JSON对象数组
        "name": name
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9055)