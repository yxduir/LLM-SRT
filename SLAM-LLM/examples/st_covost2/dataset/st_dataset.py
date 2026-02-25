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


iso3_to_iso2_map = {
    "afr": "af", "amh": "am", "ara": "ar", "asm": "as", "ast": "ast",
    "azj": "az", "bel": "be", "bul": "bg", "ben": "bn", "bos": "bs",
    "cat": "ca", "ceb": "ceb", "ckb": "ckb", "cmn": "zh", "ces": "cs",
    "cym": "cy", "dan": "da", "deu": "de", "ell": "el", "eng": "en",
    "spa": "es", "est": "et", "fas": "fa", "ful": "ff", "fin": "fi",
    "tgl": "tl", "fra": "fr", "gle": "ga", "glg": "gl", "guj": "gu",
    "hau": "ha", "heb": "he", "hin": "hi", "hrv": "hr", "hun": "hu",
    "hye": "hy", "ind": "id", "ibo": "ig", "isl": "is", "ita": "it",
    "jpn": "ja", "jav": "jv", "kat": "ka", "kam": "kam", "kea": "kea",
    "kaz": "kk", "khm": "km", "kan": "kn", "kor": "ko", "kir": "ky",
    "ltz": "lb", "lug": "lg", "lin": "ln", "lao": "lo", "lit": "lt",
    "luo": "luo", "lav": "lv", "mri": "mi", "mkd": "mk", "mal": "ml",
    "mon": "mn", "mar": "mr", "msa": "ms", "mlt": "mt", "mya": "my",
    "nob": "nb", "npi": "ne", "nld": "nl", "nso": "nso", "nya": "ny",
    "oci": "oc", "orm": "om", "ory": "or", "pan": "pa", "pol": "pl",
    "pus": "ps", "por": "pt", "ron": "ro", "rus": "ru", "snd": "sd",
    "slk": "sk", "slv": "sl", "sna": "sn", "som": "so", "srp": "sr",
    "swe": "sv", "swh": "sw", "tam": "ta", "tel": "te", "tgk": "tg",
    "tha": "th", "tur": "tr", "ukr": "uk", "umb": "umb", "urd": "ur",
    "uzb": "uz", "vie": "vi", "wol": "wo", "xho": "xh", "yor": "yo",
    "yue": "yue", "zul": "zu"
}

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
        self.lang_code = dataset_config.get("lang_code", "eng")
        print("lang_code:", self.lang_code)

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
        src_lang = ['eng',"zho","jpn","kor"]
        # src_lang = ['spa']
        src_langs = ['eng']
        # src_lang = ['jpn', 'kor']






        # src_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho","yue"]



        
        # eng_Lant
        tgt_langs = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'cmn','yue']
        # tgt_lang = ["pes","tur","hin","tgl","arb","zsm","ces"]
        # tgt_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho","yue"]

        # tgt_lang = ['eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'rus', 'jpn', 'kor', 'vie', 'ind','tha',"zho","yue"]

        # tgt_lang = ['deu', 'fra', 'rus', 'jpn', "zho", "eng"]

        # tgt_lang = ['zho']



        # tgt_lang = ['jpn']

        # tgt_lang = ['jpn', "zho","yue"]
        # tgt_lang = ["eng"]
        langs_must = ["cmn","eng","ara","ind","tur","tam","swh","tha","jpn","kor","hun","vie"]
        langs_must = ["cmn","eng"]

        langs_not = [
                    "amh",
                    "asm",
                    "azj",
                    "ckb",
                    "ful",
                    "gle",
                    "hau",
                    "ibo",
                    "kam",
                    "kea",
                    "khm",
                    "kir",
                    "lin",
                    "lug",
                    "luo",
                    "mon",
                    "mri",
                    "mya",
                    "nso",
                    "nya",
                    "orm",
                    "ory",
                    "pus",
                    "sna",
                    "snd",
                    "som",
                    "tgk",
                    "umb",
                    "wol",
                    "xho",
                    "yor",
                    "zul"
                    ]
        language_1 =  ['eng']
        language_28 = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'cmn']
        language_12 = ["cmn","eng","ara","ind","tur","tam","swh","tha","jpn","kor","hun","vie"]
        language_ab = ["ara","cmn","ind","jpn","khm","kor","lao","mya","tha","tur","vie","eng"]
        language_9 = ["cmn","eng","ara","ind","tur","tha","jpn","kor","vie"]
        language_50 = ["ara","bul","ben","cat","ces","dan","deu","ell","spa","eng","est","fin","fil","fra","guj","heb","hin","hrv","hun","ind","isl","ita","jpn","kan","kor","lit","lav","mal","mar","nld","nor","pan","pol","por","ron","rus","slk","slv","srp","swe","swa","tam","tel","tha","tur","ukr","urd","vie","cmn","yue","zul"]
        language_67 = ['afr', 'ara', 'ast', 'bel', 'ben', 'bos', 'bul', 'cat', 'ceb', 'ces', 'cmn', 'cym', 'dan', 'deu', 'ell', 'est', 'fas', 'fin', 'fra', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hun', 'hye', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kan', 'kat', 'kaz', 'kor', 'lao', 'lav', 'lit', 'ltz', 'mal', 'mar', 'mkd', 'mlt', 'msa', 'nld', 'nob', 'npi', 'oci', 'pan', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'ukr', 'urd', 'uzb', 'vie', 'yue', 'eng']
        language_71 = ['afr', 'amh', 'ara', 'asm', 'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 'ces', 'cmn', 'cym', 'dan', 'deu', 'ell', 'eng', 'est', 'fas', 'fin', 'fra', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hun', 'hye', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kan', 'kat', 'kaz', 'khm', 'kir', 'kor', 'lao', 'lav', 'lit', 'mal', 'mkd', 'mon', 'msa', 'mya', 'nld', 'nob', 'npi', 'pan', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'ukr', 'urd', 'uzb', 'vie', 'yue']
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
                    src_lang = data_source.split("_")[-2]
                    tgt_lang = data_source.split("_")[-1]
                    if self.source == data_source:
                        self.data_list.append(data_dict)
                    elif self.source == "all":
                        if src_lang not in langs_not and tgt_lang not in langs_not:
                            self.data_list.append(data_dict)
                    elif self.source == "must":
                        if src_lang not in langs_not and tgt_lang not in langs_not:
                            if src_lang in langs_must or tgt_lang in langs_must:
                                self.data_list.append(data_dict)
                    elif self.source == "other":
                        if src_lang not in langs_not and tgt_lang not in langs_not:
                            if src_lang not in langs_must and tgt_lang not in langs_must:
                                self.data_list.append(data_dict)
                    elif self.source == "lang":
                        if src_lang == self.lang_code:
                            self.data_list.append(data_dict)
                    elif self.source == "lang_must":
                        if src_lang == self.lang_code or tgt_lang == self.lang_code:
                            if src_lang in langs_must or tgt_lang in langs_must:
                                self.data_list.append(data_dict)
                    elif self.source == "1228":
                        src_langs = language_12
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "0928":
                        src_langs = language_9
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "0128":
                        src_langs = language_1
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "2828":
                        src_langs = language_28
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "0101":
                        src_langs = language_1
                        tgt_langs = language_1
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "5050":
                        src_langs = language_50
                        tgt_langs = language_50
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "eng28":
                        src_langs = ["eng"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "cmn28":
                        src_langs = ["cmn"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "jpn28":
                        src_langs = ["jpn"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "kor28":
                        src_langs = ["kor"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                print(f"src_langs: {src_langs}")
                print(f"tgt_langs: {tgt_langs}")
            # 打乱数据顺序
            random.shuffle(self.data_list)          
        else:
            with open(dataset_config.get("val_data_path"), encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data_source = data_dict["source"]
                    src_lang = data_source.split("_")[-2]
                    tgt_lang = data_source.split("_")[-1]
                    if self.source == data_source:
                        self.data_list.append(data_dict)
                    elif self.source == "all":
                            self.data_list.append(data_dict)
                    elif self.source == "must":
                        if src_lang not in langs_not and tgt_lang not in langs_not:
                            if src_lang in langs_must or tgt_lang in langs_must:
                                self.data_list.append(data_dict)
                    elif self.source == "other":
                        if src_lang not in langs_not and tgt_lang not in langs_not:
                            if src_lang not in langs_must and tgt_lang not in langs_must:
                                self.data_list.append(data_dict)
                    elif self.source == "lang":
                        if src_lang == self.lang_code:
                            self.data_list.append(data_dict)
                    elif self.source == "lang_must":
                        if src_lang == self.lang_code or tgt_lang == self.lang_code:
                            if src_lang in langs_must or tgt_lang in langs_must:
                                self.data_list.append(data_dict)
                    elif self.source == "1228":
                        src_langs = language_12
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "0928":
                        src_langs = language_9
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "0128":
                        src_langs = language_1
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "2828":
                        src_langs = language_28
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "1271":
                        if src_lang in language_12 and  tgt_lang in language_71:
                            if src_lang in language_9 and tgt_lang in language_67:
                                pass
                            else:
                                self.data_list.append(data_dict)
                    elif self.source == "6771":
                        if src_lang in language_71 and  tgt_lang in language_71:
                            if src_lang in language_12:
                                pass
                            elif src_lang in language_67 and tgt_lang in language_67:
                                pass
                            else:
                                self.data_list.append(data_dict)
                    elif self.source == "0101":
                        src_langs = language_1
                        tgt_langs = language_1
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "5050":
                        src_langs = language_50
                        tgt_langs = language_50
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "engdeu":
                        src_langs = ["eng"]
                        tgt_langs = ["deu"]
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "engcmn":
                        src_langs = ["eng"]
                        tgt_langs = ["cmn"]
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "eng28":
                        src_langs = ["eng"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "eng50":
                        src_langs = ["eng"]
                        tgt_langs = language_50
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "engab":
                        src_langs = ["eng"]
                        tgt_langs = language_ab
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "cmn28":
                        src_langs = ["cmn"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "jpn28":
                        src_langs = ["jpn"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                    elif self.source == "kor28":
                        src_langs = ["kor"]
                        tgt_langs = language_28
                        if src_lang in src_langs and  tgt_lang in tgt_langs:
                            self.data_list.append(data_dict)
                random.shuffle(self.data_list)  
                if self.validnum == -1:
                    random.shuffle(self.data_list)
                    if len(self.data_list)>1000:
                        self.data_list=self.data_list[:1000]
                elif self.validnum == -2:
                    pass
                else:
                    self.data_list = random.sample(self.data_list, self.validnum)
               


        self.printed = False  # 标志位，控制print只执行一次
        print(split,len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]

        audio_path = data_dict.get("audio")
        if not audio_path.startswith('/'):
            audio_path = self.data_dir + audio_path
        



        
        prompt = data_dict.get("prompt")
        target = data_dict.get("gt")
        source = data_dict.get("source")

        if self.mode == "smt":
            prompt = target.split(prompt)[0]+prompt
            target = target.split(prompt)[1]
        if self.mode == "smtsrt":
            if random.random() < 0.5:  # 50%概率触发
                prompt = target.split(prompt)[0] + prompt
                target = target.split(prompt)[1]
        if self.mode == "st":
            prompt = prompt
            target = target.split(prompt)[1]
        elif self.mode == "asr":
            prompt = prompt[:7]
            prompt_lang = prompt[2:5]
            iso_2_map = iso3_to_iso2_map[prompt_lang]
            prompt = "<"+iso_2_map+">"
            target = target.split(prompt)[0]
        # elif self.mode == "srt":
        #     src_lang = prompt[2:5]
        #     tgt_lang = prompt[9:12]
        #     prompt = "<" + iso3_to_iso2_map[src_lang] + "><" + iso3_to_iso2_map[tgt_lang] + ">"
        #     target = target.replace(f"<|{src_lang}|>",f"<{iso3_to_iso2_map[src_lang]}>").replace(f"<|{tgt_lang}|>",f"<{iso3_to_iso2_map[tgt_lang]}>")
            
        
        if not self.printed:  # 如果没有打印过，则打印一次
            print(prompt)
            print(target)
            self.printed = True  # 设置标志位，表示已经打印过了


        key = data_dict.get("key", str(index))

        audio_raw = whisper.load_audio(audio_path)


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
            audio_mel = audio_mel.to(torch.bfloat16)

        
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
