from transformers import GPT2Tokenizer, AutoModelForCausalLM

# 加载基于 BPE 的分词器，例如 GPT-2 的分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# 测试分词器和模型是否正常运行
text = "This is an example sentence for testing the BPE-based tokenizer."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

# 解码生成的结果
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)