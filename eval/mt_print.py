from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/translategemma-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")

source_lang_code = "cs"
target_lang_code = "de-DE"
source_text = "V nejhorším případě i k prasknutí čočky."

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": source_lang_code,
                "target_lang_code": target_lang_code,
                "text": source_text,
            }
        ],
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("=== model input ids ===")
print(model_inputs.input_ids)
print("=== model input tokens ===")
print(tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0]))
print("=== model input decoded (with special tokens) ===")
print(tokenizer.decode(model_inputs.input_ids[0], skip_special_tokens=False))

generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=False)
print("=== generated ids (all) ===")
print(generated_ids)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
print("=== output ids (new tokens) ===")
print(output_ids)
print("=== output tokens (new tokens) ===")
print(tokenizer.convert_ids_to_tokens(output_ids))

outputs = tokenizer.decode(output_ids, skip_special_tokens=False)

print("response:", outputs)
