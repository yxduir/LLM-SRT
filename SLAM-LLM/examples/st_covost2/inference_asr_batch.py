import hydra
import logging
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional
from asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig
# import fire
import random
import torch
import logging
import sacrebleu
# import argparse
import itertools
import json
import time
from slam_llm.models.slam_model import slam_model

# config
# from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
# from llama_recipes.configs import model_config as MODEL_CONFIG
# from llama_recipes.configs import log_config as LOG_CONFIG
from slam_llm.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    get_policies
)
from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
import os
import logging
from tqdm import tqdm
from model.slam_model_st import model_factory
from transformers import  AutoTokenizer,AutoConfig,AutoModel

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from slam_llm.utils.model_utils import get_custom_model_factory

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def Inference(kwargs: DictConfig):

    # Update the configuration for the training and sharding process
    train_config, fsdp_config, model_config, log_config, dataset_config,ckpt_path = kwargs.train_config, \
                                                                          kwargs.fsdp_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config, \
                                                                          kwargs.ckpt_path 

    OmegaConf.set_struct(kwargs,False)
    del kwargs["train_config"]
    del kwargs["fsdp_config"]
    del kwargs["model_config"]
    del kwargs["log_config"]
    del kwargs["dataset_config"]
    OmegaConf.set_struct(kwargs,True)


    # Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 

    logger.addHandler(file_handler)


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)


    # --- 分布式设置 ---
    if train_config.enable_fsdp or train_config.enable_ddp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # 非分布式模式下的默认值
        local_rank = 0
        rank = 0
        world_size = 1


    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
            
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        logger.info("train_config: {}".format(train_config))
        logger.info("fsdp_config: {}".format(fsdp_config))
        logger.info("model_config: {}".format(model_config))
        logger.info("log_config: {}".format(log_config))

    beam = model_config["beam"]
    use_api = train_config["use_api"]
    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
            

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # model.to(torch.bfloat16)
    model.to(torch.bfloat16)

    dataset_config["fp16"]=False
    model.to(device)
    model.eval()
    tokenizer.padding_side = 'left'


    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

    test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            sampler=InferenceSampler(len(dataset_test)),
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            shuffle=False,
            batch_size=train_config.val_batch_size,
            drop_last=False,
            prefetch_factor=10,
            persistent_workers=False,
            collate_fn=dataset_test.collator
        )

    
    # ----------------------------------------------------
    # OOM 修复部分：移除列表积累，改为实时写入磁盘
    # ----------------------------------------------------
    
    # 定义当前 Rank 的临时文件名
    rank_results_file_path = f"{log_config.decode_log}.rank_{rank}"
    
    # 确保在开始推理前清空临时文件，以防残留
    if os.path.exists(rank_results_file_path):
        os.remove(rank_results_file_path)
    
    logger.info(f"Rank {rank}: 开始推理，结果将写入临时文件 {rank_results_file_path}")
    
    # 替换原来的列表积累

    with open(rank_results_file_path, 'a', encoding='utf-8') as f:
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Rank {rank} Inferring"):

                for key in batch.keys():
                    batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]

                model_outputs = model.generate(**batch,beam=beam,use_api=use_api, **kwargs)

                output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)

                # 实时写入磁盘
                batch_results = []
                source = "srt" # 在循环外部定义，减少重复赋值
                
                for key, audio_path ,prompt,text, target in zip(batch["keys"],batch["audio_paths"],batch["prompts"], output_text, batch["targets"]):    
                    
                    # 打印到控制台/日志
                    print(key,"pred: ",text)
                    print(key,"gold: ",target)

                    # 将结果字典添加到当前批次列表中
                    result = {
                        'gt': target,
                        'response': text,
                        'source': source,
                        "audio_path": audio_path,
                        "prompt": prompt,
                        "rank": rank # 可选：添加 Rank 信息
                    }
                    batch_results.append(result)

                # 将当前批次结果立即写入 Rank 文件
                
                for res in batch_results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # ----------------- 推理循环结束 -----------------
            
    # 同步：确保所有 Rank 都完成了文件写入
    logger.info(f"Rank {rank}: 推理完成，等待所有 Rank 同步...")
    torch.distributed.barrier()

    
    # ----------------------------------------------------
    # OOM 修复部分： Rank 0 合并文件
    # ----------------------------------------------------

    if rank == 0:
        logger.info("Rank 0: 所有 Rank 推理完成，开始合并结果文件...")
        
        final_results_file = log_config.decode_log
        
        # 清空或创建最终的合并文件
        try:
            with open(final_results_file, 'w', encoding='utf-8') as outfile:
                
                # 遍历所有 Rank 的结果文件
                for i in range(world_size):
                    temp_file_path = f"{log_config.decode_log}.rank_{i}"
                    
                    if os.path.exists(temp_file_path):
                        with open(temp_file_path, 'r', encoding='utf-8') as infile:
                            # 逐行复制内容
                            for line in infile:
                                outfile.write(line)
                        
                        # 成功合并后删除临时文件
                        os.remove(temp_file_path)
                    else:
                        logger.warning(f"Rank 0: 未找到 Rank {i} 的结果文件：{temp_file_path}")
                        
            logger.info(f"Rank 0: 最终结果合并完成，写入 {final_results_file}")
            
        except Exception as e:
            logger.error(f"Rank 0: 结果合并过程中发生致命错误: {e}")

    # 最终同步，确保 Rank 0 的文件操作完成
    torch.distributed.barrier()


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: str = field(
        default="output/model.pt", metadata={"help": "The path to projector checkpoint"}
    )
    peft_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to peft checkpoint, should be a directory including adapter_config.json"
        },
    )


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    # kwargs = to_plain_list(cfg)
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if cfg.get("debug", False):
        import pdb

        pdb.set_trace()

    Inference(cfg)


if __name__ == "__main__":
    main_hydra()