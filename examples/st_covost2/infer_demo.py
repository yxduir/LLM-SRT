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
import os
import logging
from tqdm import tqdm
import os
import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import json
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
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        visual_mask = kwargs.get("visual_mask", None)


        # for text encoder
        instruct_ids = kwargs.get("instruct_ids", None)
        instruct_mask = kwargs.get("instruct_mask", None)

        modality_mask = kwargs.get("modality_mask", None)


        
        encoder_outs = None
        if audio_mel is not None or audio is not None or visual is not None:
            encoder_outs = self.encoder(audio_mel.permute(0, 2, 1)).last_hidden_state # bs*seq*dim
           

           
            encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            

        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                encoder_outs_pad[
                    i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                ] = encoder_outs[i][:modality_lengths[i]]
            
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        # print(inputs_embeds.shape)
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,)
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(input=model_outputs.logits, dim=-1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)

                
        return model_outputs, acc
    
    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                beam: Optional[int] = 1,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        
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

while True:
    print("=====================================")
    jsonl_path = input("Your JSONL Path:\n")
    output_folder = os.path.dirname(jsonl_path)
    # 准备输出文件路径（在原路径基础上添加"_processed"后缀）
    output_path = os.path.splitext(jsonl_path)[0] + "_translation.jsonl"

    # 读取原始JSONL文件内容
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        original_lines = [json.loads(line) for line in f]



    dataset_config = {
        "inference_mode": True,
        "val_data_path": jsonl_path,
        "file": "/mgData3/zhaozhiyuan/vits/hit/code/SLAM-LLM/examples/st_covost2/dataset/st_dataset.py:get_speech_dataset"
    }

    dataset_test = get_preprocessed_dataset(
        model.tokenizer,
        dataset_config,
        split="test",
    )
    test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=8,
            pin_memory=True,
			shuffle=False,
            batch_size=4,
			drop_last=False,
			collate_fn=dataset_test.collator
    )

    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write("")

    processed_data = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        
        for key in batch.keys():
            batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
        model_outputs = model.generate(**batch)
        output_texts = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)

        for idx, (key, audio_path, prompt, output_text, target) in enumerate(zip(
            batch["keys"], 
            batch["audio_paths"],
            batch["prompts"], 
            output_texts, 
            batch["targets"]
        )):
            # 获取对应的原始数据
            original_entry = original_lines[step * test_dataloader.batch_size + idx]

            new_entry = {**original_entry, "response": output_text, "audio_path": audio_path}            
            processed_data.append(new_entry)

            if prompt in output_text:
                [source_text, translation_text] = output_text.split(prompt)
            else:
                source_text, translation_text = "", ""
                print(output_text)
                print(f"output_text is invalid, expecting {prompt} in it")
            
            print(key,source_text,translation_text)
    
        # 写入处理后的JSONL文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in processed_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 创建完成标记文件
    complete_file = os.path.join(output_folder, f"translation_complete")
    open(complete_file, 'w').close()
    print(f"\nProcessing complete. Marker file created: {complete_file}")


