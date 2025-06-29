
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




	if train_config.enable_fsdp or train_config.enable_ddp:
		setup()
		local_rank = int(os.environ["LOCAL_RANK"])
		rank = int(os.environ["RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
	else:
		local_rank = 0
		rank = 0
		world_size = 1
	print("local_rank: ",local_rank)
	print("rank: ",rank)
	print("world_size: ",world_size)


	if torch.distributed.is_initialized():
		torch.cuda.set_device(local_rank)
		clear_gpu_cache(local_rank)
		setup_environ_flags(rank)
            
	if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
		logger.info("train_config: {}".format(train_config))
		logger.info("fsdp_config: {}".format(fsdp_config))
		logger.info("model_config: {}".format(model_config))
		logger.info("log_config: {}".format(log_config))

	model_factory = get_custom_model_factory(model_config, logger)
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
			

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.
	# model.to(torch.bfloat16)
	model.to(torch.float16)

	dataset_config["fp16"]=True
	model.to(device)
	model.eval()
	tokenizer.padding_side = 'left'


	

	dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
	if world_size > 1:
		test_sampler = InferenceSampler(len(dataset_test))
	else:
		from torch.utils.data import SequentialSampler
		test_sampler = SequentialSampler(dataset_test)

	test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
			sampler=test_sampler,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
			shuffle=False,
            batch_size=train_config.val_batch_size,
			drop_last=False,
			prefetch_factor=10,
            persistent_workers=False,
			collate_fn=dataset_test.collator
        )

	gts = []
	sources = []
	rets = []
	audio_paths = []
	prompts = []
	
	for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

		for key in batch.keys():
			batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]

		model_outputs = model.generate(**batch)

		# print(model_outputs)
		output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)

		for key, audio_path ,prompt,text, target in zip(batch["keys"],batch["audio_paths"],batch["prompts"], output_text, batch["targets"]):	
			# print("Prediction:  ",key,text)
			# print("Ground Truth:",key,target)
			print(key,"pred: ",text)
			print(key,"gold: ",target)

			source = "eng"

			audio_paths.append(audio_path)
			rets.append(text)
			gts.append(target)
			sources.append(source)
			prompts.append(prompt)

	if world_size > 1:
		torch.distributed.barrier()
		merged_gts = [None for _ in range(world_size)]
		torch.distributed.all_gather_object(merged_gts, gts)
		merged_gts = [None for _ in range(world_size)]
		merged_sources = [None for _ in range(world_size)]
		merged_responses = [None for _ in range(world_size)]
		merged_audio_paths = [None for _ in range(world_size)]
		merged_prompts = [None for _ in range(world_size)]
		torch.distributed.all_gather_object(merged_gts, gts)
		torch.distributed.all_gather_object(merged_sources, sources)
		torch.distributed.all_gather_object(merged_responses, rets)
		torch.distributed.all_gather_object(merged_audio_paths, audio_paths)
		torch.distributed.all_gather_object(merged_prompts, prompts)

		merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
		merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
		merged_responses = [_ for _ in itertools.chain.from_iterable(merged_responses)]
		merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
		merged_prompts = [_ for _ in itertools.chain.from_iterable(merged_prompts)]
	else:
		merged_gts = gts
		merged_responses = rets
		merged_sources = sources
		merged_audio_paths = audio_paths
		merged_prompts = prompts


	

	if world_size > 1:
		if torch.distributed.get_rank() == 0:
			results_file = log_config.decode_log
			with open(results_file, 'w') as f:
				for gt, response, source, audio_path, prompt in zip(
					merged_gts, merged_responses, merged_sources, merged_audio_paths, merged_prompts
				):
					result = {
						'gt': gt,
						'response': response,
						'source': source,
						"audio_path": audio_path,
						"prompt": prompt,
					}
					f.write(json.dumps(result, ensure_ascii=False) + '\n')
			print(f"Results saved to: {results_file}") 
		torch.distributed.barrier()
	else:
		results_file = log_config.decode_log
		with open(results_file, 'w') as f:
			for gt, response, source, audio_path, prompt in zip(
				merged_gts, merged_responses, merged_sources, merged_audio_paths, merged_prompts
			):
				result = {
					'gt': gt,
					'response': response,
					'source': source,
					"audio_path": audio_path,
					"prompt": prompt,
				}
				f.write(json.dumps(result, ensure_ascii=False) + '\n')
		print(f"Results saved to: {results_file}")  


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
