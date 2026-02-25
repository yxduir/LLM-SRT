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
import packaging
from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size, print_model_size
from peft import PeftModel, PeftConfig
from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy
from transformers import SeamlessM4Tv2ForSpeechToText,SeamlessM4Tv2ForTextToText
import logging
import requests
import json
import base64
import io
from typing import Optional, List
from transformers import StoppingCriteria, StoppingCriteriaList
import concurrent.futures # 用于并行加速 API 请求
import io
import base64
import torch
import numpy as np
import aiohttp
import asyncio
from typing import List, Optional
logger = logging.getLogger(__name__)



def model_factory(train_config, model_config, **kwargs):
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)



    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    model = slam_model(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )
    

    ckpt_path = kwargs.get("ckpt_path", None) #FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        
        # 加载模型
        # 保存原始的键字典
        original_keys = set(model.state_dict().keys())
        
        # 获取加载的键字典
        ckpt_dict = {k: v for k,v in torch.load(ckpt_path).items()}           # 修正代码
        ckpt_keys = set(ckpt_dict.keys())
        load_result = model.load_state_dict(ckpt_dict, strict=False)

        # 计算加载成功的键 (既在模型又在检查点中的键)
        loaded_keys = original_keys & ckpt_keys

        print("\n=== Checkpoint Loading Summary ===")
        print(f"Model layers: {len(original_keys)} layers")
        print(f"Checkpoint layers: {len(ckpt_keys)} layers")
        print(f"Successfully loaded: {len(loaded_keys)} layers")

        print(f"Missing in checkpoint: {len(load_result.missing_keys)} layers")
        print(f"Unexpected in checkpoint: {len(load_result.unexpected_keys)} layers")

        # 打印Unexpected的键 (前几个示例)
        if load_result.unexpected_keys:
            print("\nUnexpected loaded layers:")
            for key in list(load_result.unexpected_keys)[:5]: # 只打印5示例
                print(f"(\t- {key})")

        # 打印Missing的键 (前几个示例)
        if load_result.missing_keys:
            print("\nMissing loaded layers:")
            for key in list(load_result.missing_keys)[:5]: # 只打印5示例
                print(f"(\t- {key})")

        # 打印加载成功的键 (前几个示例)
        if loaded_keys:
            print("\nSample successfully loaded layers:")
            for key in sorted(loaded_keys)[:999]: # 只打印999示例
                print(f"(\t- {key})")

    print_model_size(model, train_config, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    # print(model)
    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if "vallex" in model_config.llm_name.lower():
        return None  
    elif "mupt" in model_config.llm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path,
                                            trust_remote_code=True,
                                            use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path,trust_remote_code=True)
        

        # tokenizer.pad_token_id = tokenizer.eos_token_id
        print("tokenizer.pad_token_id:",tokenizer.pad_token_id)
        print("tokenizer.eos_token_id:",tokenizer.eos_token_id)

        # 打印对应的 token 文本
        if tokenizer.pad_token_id is not None:
            pad_token = tokenizer.decode(tokenizer.pad_token_id)
            print("tokenizer.pad_token:", pad_token)
        else:
            print("tokenizer.pad_token: None (未设置)")

        if tokenizer.eos_token_id is not None:
            eos_token = tokenizer.decode(tokenizer.eos_token_id)
            print("tokenizer.eos_token:", eos_token)
        else:
            print("tokenizer.eos_token: None (未设置)")

    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_list = model_config.encoder_name.split(",") if model_config.encoder_name else []
    if len(encoder_list) == 0:
        return None
    if len(encoder_list) == 1:
        encoder_name = encoder_list[0]
        if encoder_name == "whisper" or encoder_name == "qwen-audio":
            from slam_llm.models.encoder import WhisperWrappedEncoder
            encoder = WhisperWrappedEncoder.load(model_config)
        if encoder_name == "beats": 
            from slam_llm.models.encoder import BEATsEncoder
            encoder = BEATsEncoder.load(model_config)
        if encoder_name == "eat":
            from slam_llm.models.encoder import EATEncoder
            encoder = EATEncoder.load(model_config)
        if encoder_name == "SpatialAST":
            from slam_llm.models.encoder import SpatialASTEncoder
            encoder = SpatialASTEncoder.load(model_config)
        if encoder_name == "wavlm":
            from slam_llm.models.encoder import WavLMEncoder
            encoder = WavLMEncoder.load(model_config)
        if encoder_name == "av_hubert":
            from slam_llm.models.encoder import AVHubertEncoder
            encoder = AVHubertEncoder.load(model_config)
        if encoder_name == "hubert":
            from slam_llm.models.encoder import HubertEncoder
            encoder = HubertEncoder.load(model_config)
        if encoder_name == "musicfm":
            from slam_llm.models.encoder import MusicFMEncoder
            encoder = MusicFMEncoder.load(model_config)

        if "llama" in encoder_name.lower():
            from slam_llm.models.encoder import HfTextEncoder
            encoder = HfTextEncoder.load(model_config)
    print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()
    print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder

def setup_llm(train_config, model_config, **kwargs):
    # from pkg_resources import packaging
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.llm_path,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
        # use_cache=use_cache,
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
        # attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    # print(model)
    

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    
        
    # if kwargs.get("peft_ckpt", None): # (FIX:MZY):reload will get wrong results when decoding
    #     logger.info("loading peft_ckpt from: {}".format(kwargs.get("peft_ckpt")))
    #     model = PeftModel.from_pretrained(model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True)
    #     model.print_trainable_parameters()
    if train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        print(peft_config)
        # NOTE: 此时 LLM Base Model 已经被 PEFT 自动冻结 (requires_grad=False)
        #       只有 LoRA 适配器是可训练的 (requires_grad=True)。
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # 2. 处理手动冻结 (非 PEFT 场景)
    # 只有在没有使用 PEFT 且明确要求冻结 LLM 时，才手动冻结。
    elif train_config.freeze_llm: # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()

    # print(model)
    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model

def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from slam_llm.models.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from slam_llm.models.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from slam_llm.models.projector import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(model_config)
    elif model_config.encoder_projector == "qqm":
        from slam_llm.models.projector import EncoderProjectorQFormerPool
        encoder_projector = EncoderProjectorQFormerPool(model_config)
    else:
        return None
    print_module_size(encoder_projector, model_config.encoder_projector, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return encoder_projector


class slam_model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        tokenizer, 
        train_config, 
        model_config, 
        **kwargs
    ):
        super().__init__()
        # modality encoder 
        self.encoder = encoder


        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer

        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if train_config.get("enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)



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
            if self.train_config.freeze_encoder: # freeze encoder
                self.encoder.eval()
            if self.model_config.encoder_path_hf is not None:
                encoder_outs = self.encoder(audio_mel.permute(0, 2, 1)).last_hidden_state # bs*seq*dim
                # print("encoder_outs",encoder_outs.shape)
            elif self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(audio_mel, audio_mel_mask) # bs*seq*dim
            if self.model_config.encoder_name == "eat":
                encoder_outs = self.encoder.model.extract_features(audio_mel.unsqueeze(dim=1), padding_mask = None, mask=False, remove_extra_tokens = False)['x']
            if self.model_config.encoder_name == "SpatialAST":
                encoder_outs = self.encoder(audio) # output: [bs, seq_len=3+512, dim=768]
            if self.model_config.encoder_name == "wavlm":
                encoder_outs = self.encoder.extract_features(audio, 1 - audio_mask) #(FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
            if self.model_config.encoder_name == "hubert":
                results = self.encoder(source = audio, padding_mask = 1-audio_mask)
                if self.model_config.encoder_type == "pretrain":
                    encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
                if self.model_config.encoder_type == "finetune":
                    encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                    encoder_outs = encoder_outs.transpose(0, 1)
            if self.model_config.encoder_name == "av_hubert":
                results = self.encoder(source={'video':visual, 'audio':audio}, padding_mask=visual_mask) # bs*seq*dim  
                encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                encoder_outs = encoder_outs.transpose(0, 1)
                audio_mel_post_mask = (~audio_mel_post_mask).float()
            if self.model_config.encoder_name == 'musicfm':
                encoder_outs = self.encoder.extract_features(audio, padding_mask = None) # MusicFM doesn't support padding mask 
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio

            if self.model_config.encoder_projector == "q-former" or self.model_config.encoder_projector == "qqm":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
            if self.model_config.encoder_projector == "cov1d-linear": 
                encoder_outs = self.encoder_projector(encoder_outs) 

        
        if instruct_ids is not None:
            if self.encoder is not None:
                encoder_outs = self.encoder(input_ids=instruct_ids, attention_mask=instruct_mask).last_hidden_state

            if self.model_config.encoder_projector == "q-former" or self.model_config.encoder_projector == "qqm":
                encoder_outs = self.encoder_projector(encoder_outs, instruct_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
        

        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                # Check for the language_model attribute which is present in some models
                if hasattr(self.llm.model, "language_model") and hasattr(self.llm.model.language_model, "embed_tokens"):
                    inputs_embeds = self.llm.model.language_model.embed_tokens(input_ids)
                elif hasattr(self.llm.model, "embed_tokens"):
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
                beam: Optional[int] = 5,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        # 1. 获取 Embeddings (保持 bf16 精度)
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
        # print("inputs_embeds shape:", inputs_embeds.shape)
        # print("attention_mask shape:", attention_mask.shape)
        # print(attention_mask)

        # --- 方案 A：API 访问 (支持 Batch & 返回格式对齐) ---
        use_api = kwargs.get("use_api", False)
        if use_api:
            vllm_url = "http://localhost:8901/v1/completions"
            llm_path = self.model_config.get("llm_path")
            model_name = os.path.basename(llm_path.rstrip('/'))
            batch_size = inputs_embeds.shape[0]
            prompts = kwargs.get("prompts", [""] * batch_size)
            
            def process_single_embed(i, must_mode=False, max_retries=5):
                # 1. 提取单条并保持 bf16

                start_idx = attention_mask[i].nonzero(as_tuple=True)[0][0]
                embed_tensor = inputs_embeds[i, start_idx:, :].detach().cpu().to(torch.bfloat16)
                prompt = prompts[i]
                
                # 2. 序列化
                buff = io.BytesIO()
                torch.save(embed_tensor, buff)
                buff.seek(0)
                base64_str = base64.b64encode(buff.read()).decode("utf-8")

                payload = {
                    "model": model_name,
                    "prompt_embeds": base64_str,
                    "max_tokens": 300,
                    "min_tokens": 10,
                    "temperature": 0, # 
                    "top_p": 1,
                }

                # 3. 重试逻辑
                # 预设不同尝试次数对应的温度
                temp_map = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

                attempts = 0
                last_error = ""

                while attempts < max_retries:
                    try:
                        # 根据当前尝试次数更新 payload 中的温度
                        if attempts > 0:
                            payload["temperature"] = temp_map.get(attempts, 1.0)

                        response = requests.post(vllm_url, json=payload, timeout=120)
                        
                        if response.status_code == 200:
                            result_text = response.json()["choices"][0]["text"]
                            
                            # 检查是否满足 must 条件
                            if not must_mode or (prompt in result_text):
                                return result_text
                            else:
                                last_error = "Prompt not in response (must_mode check failed)"
                                # 这种情况下也计入重试
                        else:
                            last_error = f"API Error (Status {response.status_code}): {response.text}"
                            
                    except Exception as e:
                        last_error = f"Request Exception: {str(e)}"
                    
                    # --- 统一重试逻辑 ---
                    attempts += 1
                    
                print(f"Attempt {attempts} failed: {last_error}.")               
                return f"Error: Failed to include prompt after {max_retries} retries."

            # 使用线程池并行发送 Batch 请求，提高效率
            print(f"正在通过 API 并行处理 Batch (Size: {batch_size})...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                batch_texts = list(executor.map(process_single_embed, range(batch_size)))

            # 关键步骤：将 Text 转回 Token IDs，确保与本地 generate 输出格式一致
            # 我们手动添加 EOS token 模拟 generate 的行为
            encoded_outputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids.to(inputs_embeds.device)
            
            return encoded_outputs

        # --- 方案 B：本地 LLM 推理 ---
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 400),
            num_beams=kwargs.get("num_beams", beam),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_new_tokens", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            no_repeat_ngram_size=5,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return model_outputs