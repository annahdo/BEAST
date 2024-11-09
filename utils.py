import torch

def generate(model, tokenizer, text, max_new_tokens=5, do_sample=False, temperature=None, top_p=None):
    text = list(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    _, input_length = inputs["input_ids"].shape
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, 
                             top_p=top_p, pad_token_id=tokenizer.eos_token_id, output_scores=True)
    answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    return outputs, answers


def tokenize(tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    return inputs

def get_logits(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs