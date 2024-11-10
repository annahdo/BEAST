import torch
import torch.nn.functional as F
from einops import repeat, rearrange

def prepare_input(chat_format, sens):        
    # assert only one user-assistant dialog
    # empty list of size len(sens)
    formated_sens = [0] * len(sens)
    for i in range(len(sens)):
        formated_sens[i] = "{}{}{}{}".format(chat_format['sep'][0], chat_format['user'][0], sens[i].strip(" "), chat_format['assistant'][0])

    return formated_sens

def sample_top_p(probs, p, return_tokens=0):
    """
    Masks out the bottom (1-p) fraction from token probabilities,
    and returns the next_token / all probability indices.
    Params:
        probs: softmax logit values
        p: top_p
        return_tokens: no. of tokens returned
    Return:
        next_token: set of next tokens
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))    
    next_token = torch.multinomial(probs_sort, num_samples=max(1, return_tokens))
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate(model, tokenizer, text, max_new_tokens=5, do_sample=False, temperature=None, top_p=1, top_k=None):
    text = list(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    _, input_length = inputs["input_ids"].shape
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, 
                             top_p=top_p, top_k=top_k, pad_token_id=tokenizer.eos_token_id)
    answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    return answers


def get_logits(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

def get_target_tokens(tokenizer, targets, lookahead_length):
    
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    target_tokens = []
    target_tokens = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
    # reduce to lookahead_length, delete bos token
    target_tokens['input_ids'] = target_tokens['input_ids'][:, 1:1+lookahead_length]
    target_tokens['attention_mask'] = target_tokens['attention_mask'][:, 1:1+lookahead_length]
    # TODO check if bos token should be deleted
    tokenizer.padding_side = original_padding_side
    return target_tokens

def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def calc_perplexity(total_tokens, model, batch_size=16):
    # TODO: total_tokens are mostly the same sentence at the beginning. it might make sense to calculate hidden states for the beginning of the sentence only once
    # calculate perplexity in batches
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    perplexities = []
    for input_ids_batch, input_attention_mask_batch in zip(batchify(total_tokens['input_ids'], batch_size), batchify(total_tokens['attention_mask'], 16)):

        batch = {'input_ids': input_ids_batch, 'attention_mask': input_attention_mask_batch}
        logits = model(**batch, return_dict=True).logits
        # exclude last token from perplexity calculation
        logits = logits[:, :-1].contiguous() 
        labels = batch['input_ids'][:, 1:].contiguous()
        attention_mask = batch['attention_mask'][:, 1:].contiguous()
        # Compute loss for each token (not averaged)
        per_token_loss = cross_entropy(
            rearrange(logits, 'b t c -> (b t) c'),
            rearrange(labels, 'b t -> (b t)')
            )
        per_token_loss = rearrange(per_token_loss, '(b t) -> b t', b=batch['input_ids'].size(0))

        # Mask padding tokens
        per_token_loss = per_token_loss * attention_mask  # Zero-out loss for padding

        # Sum the loss over the sequence
        seq_loss = per_token_loss.sum(dim=-1)/(attention_mask.sum(dim=-1) + 1e-8)
        perplexities.extend(torch.exp(seq_loss).detach().cpu())

    return torch.tensor(perplexities)

def generate_and_extend(model, tokenizer, inputs, max_new_tokens=1, do_sample=True, top_p=1, top_k=None, temperature=1.0, k=15):
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, 
                         pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True, top_k=top_k, top_p=top_p)

    logits = outputs.scores[0]
    probs = torch.softmax(logits, dim=-1)

    curr_tokens = sample_top_p(probs, top_p, return_tokens=k)

    # add the new tokens to the input
    # Repeat input_ids and rearrange curr_tokens to match dimensions
    input_ids_expanded = repeat(inputs['input_ids'], 'b t -> b r t', r=k) 
    attention_mask_expanded = repeat(inputs['attention_mask'], 'b t -> b r t', r=k)  
    curr_tokens_expanded = rearrange(curr_tokens, 'b r -> b r 1')  
    curr_attentions_expanded = torch.ones_like(curr_tokens_expanded) 

    # Concatenate along the last dimension
    input_ids_expanded = torch.cat([input_ids_expanded, curr_tokens_expanded], dim=-1)  
    attention_mask_expanded = torch.cat([attention_mask_expanded, curr_attentions_expanded], dim=-1)  

    # Reshape to desired shape
    input_ids_expanded = rearrange(input_ids_expanded, 'b r t -> (b r) t')  
    attention_mask_expanded = rearrange(attention_mask_expanded, 'b r t -> (b r) t') 

    return {'input_ids': input_ids_expanded, 'attention_mask': attention_mask_expanded}
    

def choose_best_candidate(inputs, perplexity, best_k, out_of, device):

    perplexity = rearrange(perplexity, '(b r) -> b r', r=out_of)
    # indices corresponding to the lowest perplexity
    best_k_indices = torch.topk(perplexity, best_k, dim=1, largest=False).indices
    inputs['input_ids'] = rearrange(inputs['input_ids'], '(b r) t -> b r t', r=out_of)
    inputs['attention_mask'] = rearrange(inputs['attention_mask'], '(b r) t -> b r t', r=out_of)

    # apply k1_lowest_indices
    token_dim = inputs['input_ids'].shape[-1]
    expanded_indices = repeat(best_k_indices, 'b k -> b k d', d=token_dim)    
    selected_input_ids = torch.gather(inputs['input_ids'], dim=1, index=expanded_indices.to(device))
    selected_attention_mask = torch.gather(inputs['attention_mask'], dim=1, index=expanded_indices.to(device))

    # reshape
    inputs['input_ids'] = rearrange(selected_input_ids, 'b r t -> (b r) t')
    inputs['attention_mask'] = rearrange(selected_attention_mask, 'b r t -> (b r) t')

    return inputs