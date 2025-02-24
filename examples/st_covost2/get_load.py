def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (batch_size, Lmax).
        pad_targets (LongTensor): Target label tensors (batch_size, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float() #(FIX:MZY):return torch.Tensor type


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    datasets = load_dataset("yxdu/covost2_en_x")
    # 对 'test' 数据集进行随机打乱，并选择前 1000 条数据
    # datasets['test'] = datasets['test'].shuffle(seed=42).select(range(100))

    datasets = datasets.cast_column("audio", Audio(sampling_rate=16000))
    print(datasets)

    # 根据混合精度设置填充倍数
    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None

    def collate_fn(examples):
        # 用于批量处理每个样本
        batch_input_ids = []
        batch_labels_ids = []
        batch_example_mask = []
        batch_audio_mel = []

        prompt = "<|zh|>"    #对于语音翻译任务，prompt为<|zh|>，表示生成中文的文本翻译
        prompt_ids = tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        audio_length = 80   #80 for self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        # 初始化 max_length 为 0
        max_length = 0
        n_mels=128  #128 for whisper-large-v3
        
        for example in examples:
            # 构造目标和输入
            target = example["en"] + prompt + example["zh"]  #生成的目标为，类似：Sure, see you there!<|zh|>当然，不见不散！
            example_input = prompt + target

            audio_raw = whisper.pad_or_trim(example["audio"]["array"])# 此处padding到30s
            audio_raw = torch.tensor(audio_raw, dtype=torch.float32)  
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=n_mels).permute(1, 0)
            audio_mel = audio_mel.to(torch.bfloat16)
            
            # 使用分词器编码输入
            example_ids = tokenizer.encode(example_input)
            example_ids.append(tokenizer.eos_token_id)
            example_ids = torch.tensor(example_ids, dtype=torch.int64)

            # 拼接音频数据和文本数据（[audio, prompt, answer, eos]）
            example_ids = torch.cat((audio_pseudo, example_ids))  # 确保audio_pseudo已定义且形状正确
            labels_ids = copy.deepcopy(example_ids)  # 复制以处理标签

            # 对填充部分进行掩码处理（例如：[-1, -1, answer, eos]）
            labels_ids[:audio_length + prompt_length] = -1  # 将填充部分标记为-1
            
            # 生成注意力掩码和标签掩码
            example_mask = example_ids.ge(-1)  # 注意力掩码
            label_mask = labels_ids.ge(0)  # 标签掩码

            # 应用掩码（将填充位置设置为0，标签位置设置为-100）
            example_ids[~example_mask] = 0  # 将输入中的填充位置设置为0
            labels_ids[~label_mask] = -100  # 将标签中的填充位置设置为-100，忽略这些部分的损失计算

            max_length = max(max_length, example_ids.size(0))  # 正确更新最大长度
            
            # 收集批量中的各项数据
            batch_input_ids.append(example_ids)
            batch_labels_ids.append(labels_ids)
            batch_example_mask.append(example_mask)
            batch_audio_mel.append(audio_mel)

        # 如果设置了 pad_to_multiple_of，则调整最大长度为其倍数
        if pad_to_multiple_of:
            max_length = (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

        # 批量填充到相同长度，右填充
        def pad_sequence(sequences, pad_value):
            padded = []
            for seq in sequences:
                padding_length = max_length - seq.size(0)
                if padding_length > 0:
                    pad = torch.full((padding_length,), pad_value, dtype=seq.dtype)
                    padded.append(torch.cat([seq, pad]))
                else:
                    padded.append(seq)
            return torch.stack(padded)
        
        input_ids = pad_sequence(batch_input_ids, pad_value=tokenizer.pad_token_id)  # 填充值为 tokenizer.pad_token_id
        labels = pad_sequence(batch_labels_ids, pad_value=-100)  # 填充值为 -100
        attention_mask = pad_sequence(batch_example_mask, pad_value=False)  # 填充值为 False
        audio_mel = torch.stack(batch_audio_mel)  # audio_mel 不需要填充,都padding到30s
        audio_mel_post_mask = torch.ones(len(examples),1500) #全为1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio_mel": audio_mel,
            "audio_length": audio_length,
            "audio_mel_post_mask":audio_mel_post_mask
        }
    
    # Instantiate dataloaders.
    train_dataloader = DataLoader(datasets["train"], pin_memory=True,shuffle=True, collate_fn=collate_fn, batch_size=batch_size,prefetch_factor=1000, drop_last=True,num_workers=8,persistent_workers=True)
    eval_dataloader = DataLoader(datasets["test"],shuffle=False,pin_memory=True,collate_fn=collate_fn,batch_size=(batch_size*3),prefetch_factor=1000,num_workers=8,drop_last=(accelerator.mixed_precision == "fp8"),persistent_workers=True)

    return train_dataloader, eval_dataloader