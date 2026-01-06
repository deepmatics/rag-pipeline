
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("flowaicom/Flow-Judge-v0.1", trust_remote_code=False, local_files_only = True)
model = AutoModelForCausalLM.from_pretrained("flowaicom/Flow-Judge-v0.1", trust_remote_code=False, local_files_only = True)
messages = [
    {"role": "user", "content": "What is your purpose??"},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))