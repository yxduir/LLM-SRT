from typing import List, Optional
import torch
import argparse
import evaluate
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Audio
from accelerate import Accelerator
from transformers import PreTrainedModel,WhisperModel,AutoModelForCausalLM, AutoTokenizer,AutoTokenizer,AutoConfig,get_linear_schedule_with_warmup, set_seed
import torch.nn as nn
from huggingface_hub import hf_hub_download
import copy
from tqdm import tqdm
import whisper
import os
from collections import OrderedDict
import itertools
from examples.st_covost2.dataset.st_dataset import get_speech_dataset
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from examples.st_covost2.model.slm_model import CustomSLM
from slam_llm.utils.config_utils import get_dataloader_kwargs
import sys
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

sys.path.append('/mgData3/zhaozhiyuan/vits/hit/SLAM-LLM')


def training_function(config, args):
    gradient_accumulation_steps= 32

    if args.with_tracking:
        accelerator = Accelerator(mixed_precision=args.mixed_precision,gradient_accumulation_steps=gradient_accumulation_steps, log_with="all",project_dir=args.project_dir)
        accelerator.init_trackers(project_name="cot",init_kwargs={"wandb": {"entity": "yxduir","name": "st-zh"}})
    else:
        accelerator = Accelerator(mixed_precision=args.mixed_precision,gradient_accumulation_steps=gradient_accumulation_steps)
    

    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    tokenizer = AutoTokenizer.from_pretrained("/mgData3/zhaozhiyuan/vits/hit/speech/models/Qwen2.5-32B")
    
    set_seed(seed)
    dataset_config=  {'train_data_path': '/mgData3/zhaozhiyuan/vits/hit/SLAM-LLM/examples/st_covost2/mmt-all/enzh-mix/common_train.jsonl', 'val_data_path': '/mgData3/zhaozhiyuan/vits/hit/SLAM-LLM/examples/st_covost2/mmt-all/enzh-mix/common_test.jsonl', 'train_split': 'train', 'test_split': 'test', 'source': 'all'}
    dataset_train = get_speech_dataset(tokenizer=tokenizer,dataset_config=dataset_config,split="train")
    dataset_val = get_speech_dataset(tokenizer=tokenizer,dataset_config=dataset_config,split="val",)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=dataset_val.collator
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset_val.collator
    )


    if args.resume_from_checkpoint == None or args.resume_from_checkpoint=="":
        #自动下载训练好的q-former模型
        model_repo = "yxdu/cotst"
        model_filename = "model.pt"
        ckpt_path = hf_hub_download(repo_id=model_repo, filename=model_filename, local_dir="./")
    else:
        ckpt_path=args.resume_from_checkpoint

    config = AutoConfig.from_pretrained("/mgData3/zhaozhiyuan/vits/hit/speech/models/whisper-large-v3")  # CustomSLM继承了HF的父类，要求必须有一个config，无实际作用
    print("load ckpt: "+ ckpt_path)
    model = CustomSLM(config,ckpt_path=ckpt_path)

    # 冻结 LLM (Qwen2-7B)和冻结 encoder (Whisper)
    for param in model.llm.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    model = model.to(accelerator.device,dtype=torch.bfloat16)

    # 获取所有需要梯度计算的参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # 优化器只更新可训练的参数
    optimizer = AdamW(params=trainable_params, lr=lr)

    total_length = len(train_dataloader)//gradient_accumulation_steps
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=1000,num_training_steps=total_length * num_epochs,)

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)


    # Now we train the model
    max_eval_ave_acc = 0
    for epoch in range(num_epochs):
        model.train()
        
        train_total_acc = 0
        total_loss = 0
        train_progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in train_progress_bar:
            with accelerator.accumulate(model):
                outputs,*rest = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.detach().float()
                accuracy = rest[0] if rest else -1
                train_total_acc += accuracy
                train_ave_acc = train_total_acc/(step+1)
                if step % 100 == 0:
                    print("train_ave_acc: "+str(train_ave_acc))

                if step>0 and step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    model.eval()
                    all_accuracies = []  # 用于存储当前所有进程的准确率

                    eval_progress_bar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
                    for eval_step, eval_batch in eval_progress_bar:
                        with torch.no_grad():
                            outputs, *rest = model(**eval_batch)
                        
                        accuracy = rest[0] if rest else -1
                        all_accuracies.append(accuracy)  # 收集当前进程的准确率


                    all_accuracies = accelerator.gather(all_accuracies)  # 收集所有进程上的准确率
                    all_accuracies = [accuracy.tolist() if isinstance(accuracy, torch.Tensor) else accuracy for accuracy in all_accuracies]

                    # 只有在主进程上汇总准确率
                    if accelerator.is_main_process:
                        # 扁平化二维列表为一维列表
                        flat_list = [item for sublist in all_accuracies for item in sublist]
                        print(flat_list)
                        eval_ave_acc = sum(flat_list) / len(flat_list)
                        # 计算全局平均准确率
                        print(f"Global Eval Average Accuracy: {eval_ave_acc:.4f}")

                        #只保留最好的模型
                        if  eval_ave_acc > max_eval_ave_acc:
                            max_eval_ave_acc = eval_ave_acc
                            if args.output_dir is not None:
                                output_dir = f"epoch_{epoch}_step_{step}"
                                output_dir = os.path.join(args.output_dir, output_dir)
                                os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
                            def save_trainable_params(model, save_path):
                                # 获取可训练的部分
                                model = model.module
                                cpu_state = model.state_dict()
                                state_dict = OrderedDict()
                                for name, param in model.named_parameters():
                                    if param.requires_grad:
                                        state_dict[name] = cpu_state[name]

                                # 保存模型参数
                                torch.save(state_dict, save_path)
                            # 保存可训练参数的函数
                            save_trainable_params(model, os.path.join(output_dir, "model.pt"))
                    
                        if args.with_tracking:
                            accelerator.log({"eval_ave_acc": eval_ave_acc,"eval_best_accuracy": max_eval_ave_acc},step=(epoch * len(train_dataloader) + step),)
                            accelerator.log({"train_ave_acc": train_ave_acc,"train_ave_loss": total_loss.item() / (step+1),},step=(epoch * len(train_dataloader) + step),)
                    accelerator.wait_for_everyone()
                    model.train()

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--mixed_precision",type=str,default="bf16",choices=["no", "fp16", "bf16", "fp8"])
    parser.add_argument("--with_tracking",action="store_true",default=False,help="Whether to load in all available experiment trackers from the environment and use them for logging.",)
    parser.add_argument("--project_dir",type=str,default="logs",help="Location on where to store experiment tracking logs` and relevent project information",)
    parser.add_argument("--output_dir",type=str,default="./output/",help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",)
    parser.add_argument("--checkpointing_steps",type=int,default=2000,help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",)
    parser.add_argument("--resume_from_checkpoint",type=str,default="/mgData3/zhaozhiyuan/vits/hit/speech/data/qwen/asr-32B-enzh-2/asr_epoch_1_step_2000/model.pt",help="If the training should continue from a checkpoint folder.",)
    args = parser.parse_args()

    #默认评估的batch_size是两倍训练batch_size
    config = {"lr": 1e-4, "num_epochs": 5, "seed": 42, "batch_size":2}
    training_function(config, args)


if __name__ == "__main__":
    main()
