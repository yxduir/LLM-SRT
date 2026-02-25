import os.path as osp
import random
import json, yaml
import copy
import os
import numpy as np
from scipy import signal
import soundfile as sf
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset
import whisper
from slam_llm.utils.compute_utils import calculate_output_length_1d


class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.mode = dataset_config.get("mode", "srt")
        
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", "")
        self.bf16 = dataset_config.get("bf16", True)
        self.fp16 = dataset_config.get("fp16", False)
        self.mel_size = dataset_config.get("mel_size", 128) # 80 for whisper large v1 and v2, 128 for large v3
        self.source = dataset_config.get("source", "eng")

        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", 80)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.validnum = dataset_config.get("validnum", -2)
        self.input_type = dataset_config.get("input_type", "mel")
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 
        self.data_dir = os.path.dirname(dataset_config.get("val_data_path"))+"/"
        print(self.data_dir)

        src_lang = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho']
        # src = self.source.split("_")[-1]
        # src_lang = [src]
        # src_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho","yue"]
        # src_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho"]
        # src_lang = ['zho']
        # src_lang = ['eng',"zho","jpn","kor"]
        # src_lang = ['spa']
        # src_lang = ['zho']
        src_lang = ['eng']







        # src_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho","yue"]



        
        # eng_Lant
        tgt_lang = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho']
        
        tgt_lang = ['ara','zsm', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho']

        
        
        # tgt_lang = ["pes","tur","hin","tgl","arb","zsm","ces"]
        # tgt_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho","yue"]

        # tgt_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho","yue"]

        # tgt_lang = ['deu', 'fra', 'rus', 'jpn', "zho", "eng"]

        # tgt_lang = ['zho']
        # tgt_lang = ['eng']



        # tgt_lang = ['jpn']

        # tgt_lang = ['jpn', "zho","yue"]
        # tgt_lang = ["eng"]


        # 设置随机种子，确保结果可复现
        random_seed = 42  # 可以替换为任意整数
        random.seed(random_seed)

        
        self.data_list = []
        self.count = 0

        if split == "train":
            with open(dataset_config.get("train_data_path"), encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data_source = data_dict["source"]
                    if self.source==data_source:
                        self.data_list.append(data_dict)
                    elif self.source == "all":
                        self.data_list.append(data_dict)
                    elif  data_source.split("_")[-2] in src_lang and data_source.split("_")[-1] in tgt_lang:
                        self.data_list.append(data_dict)
            # 打乱数据顺序
            random.shuffle(self.data_list)          
        else:
            with open(dataset_config.get("val_data_path"), encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data_source = data_dict["source"]
                    if self.source == data_source:
                        self.data_list.append(data_dict)
                    elif self.source == "all":
                        self.data_list.append(data_dict)
                    elif  data_source.split("_")[-2] in src_lang and data_source.split("_")[-1] in tgt_lang:
                        self.data_list.append(data_dict)
                if self.validnum == -1:
                    random.shuffle(self.data_list)
                    # if len(self.data_list)>50000:
                    #     self.data_list=self.data_list[:50000]
                elif self.validnum == -2:
                    pass
                else:
                    self.data_list = random.sample(self.data_list, self.validnum)
               


        # 截取前 1000 条数据
        self.printed = False  # 标志位，控制print只执行一次
        print(split,len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]

        audio_path = data_dict.get("audio","")
        if not audio_path.startswith('/'):
            audio_path = self.data_dir + audio_path
        


        
        prompt = data_dict.get("prompt")
        target = data_dict.get("gt")
        source = data_dict.get("source")

        if self.mode == "smt":
            prompt = target.split(prompt)[0]+prompt
            if self.validnum ==-1:
                target = target.split(prompt)[1]
        elif self.mode == "asr":
            prompt = prompt[:7]
            target = target.split(prompt)[0]
        elif self.mode == "asrmmt":
            prompt = data_dict.get("asr").split(prompt)[0]+prompt
        
        if not self.printed:  
            print(prompt)
            print(target)
            self.printed = True 


        key = data_dict.get("key", str(index))

        audio_raw = whisper.load_audio(audio_path)
        # audio_raw, sr = librosa.load(audio_path, sr=None)  # sr=None ensures we get the original sample rate
        # Resample audio to 16000 Hz if the sample rate is different
        # if sr != 16000:
        #     audio_raw = librosa.resample(audio_raw, orig_sr=sr, target_sr=16000)
        #     sr = 16000  # Update the sample rate to 16000

        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320 # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)

        
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)


        if self.inference_mode:
            audio_mel = audio_mel.to(torch.float16)

        
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "audio_path":audio_path,
                "key": key,
                "target": target,
                "audio_path":audio_path,
                "prompt_id":prompt_ids,
                "prompt":prompt,
                "source":source,
                "prompt_length": prompt_length,
            }
        
        if self.bf16:
            audio_mel = audio_mel.to(torch.bfloat16)
        answer = self.answer_template.format(target)
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.

        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64)

        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]



        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples] #[120, 48, 82, 42]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in samples]  #[0, 0, 0, 0]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])


        if self.input_type == "raw":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]
            audio_paths = [s['audio_path'] for s in samples]
            prompts = [s['prompt'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
                "audio_paths": audio_paths,
                "prompts": prompts,
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask
        }




def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
