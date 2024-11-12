import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from tqdm import tqdm
import copy
import wandb
import re

def prepare_input(chat_format, sens):        
    # assert only one user-assistant dialog
    # empty list of size len(sens)
    formated_sens = [0] * len(sens)
    for i in range(len(sens)):
        formated_sens[i] = "{}{}{}{}".format(chat_format['system'], chat_format['user'], sens[i].strip(" "), chat_format['assistant'])

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
    target_tokens = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    # reduce to lookahead_length, delete bos token
    target_tokens['input_ids'] = target_tokens['input_ids'][:, :lookahead_length]
    target_tokens['attention_mask'] = target_tokens['attention_mask'][:, :lookahead_length]
    # TODO check if bos token should be deleted
    tokenizer.padding_side = original_padding_side
    return target_tokens

def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def calc_perplexity(input_tokens, total_tokens, model, batch_size=16):
    # TODO: total_tokens are mostly the same sentence at the beginning. it might make sense to calculate hidden states for the beginning of the sentence only once
    # calculate perplexity in batches
    perplexities = []

    sl_input_tokens = input_tokens['input_ids'].shape[1]
    
    for input_ids_batch, input_attention_mask_batch in zip(batchify(total_tokens['input_ids'], batch_size), batchify(total_tokens['attention_mask'], batch_size)):


        num_samples, sl_total_tokens = input_ids_batch.shape
        batch = {'input_ids': input_ids_batch, 'attention_mask': input_attention_mask_batch}
        logits = model(**batch, return_dict=True).logits
        # exclude last token from perplexity calculation
        softmax = torch.nn.Softmax(dim=-1)

        # Calculate log probabilities in a vectorized manner
        log_probs = -torch.log(softmax(logits))
        target_positions = torch.arange(sl_input_tokens, sl_total_tokens)
        logs = log_probs[torch.arange(num_samples)[:, None], target_positions - 1, input_ids_batch[:, target_positions]].sum(dim=1)
        perp = torch.exp(logs / (sl_total_tokens - sl_input_tokens))

        perplexities.extend(perp.detach().cpu())

    return torch.tensor(perplexities)

def generate_and_extend(model, tokenizer, inputs, max_new_tokens=1, do_sample=True, top_p=1., top_k=None, temperature=1.0, k=15):
    # TODO could maybe replace by just forward pass
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, 
                         pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True, top_k=top_k, top_p=top_p)

    logits = outputs.scores[0]
    probs = torch.softmax(logits, dim=-1) 

    # make sure not to append special tokens 
    # TODO an alternative workaround would be not hardcode the attention mask to ones but set attention for special tokens to 0
    special_ids = tokenizer.all_special_ids  # List of all special token IDs
    valid_probs = probs.clone()
    valid_probs[:, special_ids] = 0  # Set probabilities of special tokens to 0
    valid_probs /= valid_probs.sum(dim=-1, keepdim=True)  # Renormalize probabilities

    # Sample valid tokens
    # TODO do top_p_sampling if desired
    # adv_tokens = sample_top_p(valid_probs, top_p, return_tokens=k)
    adv_tokens = torch.multinomial(valid_probs, num_samples=k, replacement=False)


    # add the new tokens to the input
    # Repeat input_ids and rearrange adv_tokens to match dimensions
    input_ids_expanded = repeat(inputs['input_ids'], 'b t -> b r t', r=k) 
    attention_mask_expanded = repeat(inputs['attention_mask'], 'b t -> b r t', r=k)  
    adv_tokens_expanded = rearrange(adv_tokens, 'b r -> b r 1')  
    adv_attentions_expanded = torch.ones_like(adv_tokens_expanded) 

    # Concatenate along the last dimension
    input_ids_expanded = torch.cat([input_ids_expanded, adv_tokens_expanded], dim=-1)  
    attention_mask_expanded = torch.cat([attention_mask_expanded, adv_attentions_expanded], dim=-1)  

    # Reshape to desired shape
    input_ids_expanded = rearrange(input_ids_expanded, 'b r t -> (b r) t')  
    attention_mask_expanded = rearrange(attention_mask_expanded, 'b r t -> (b r) t') 

    return {'input_ids': input_ids_expanded, 'attention_mask': attention_mask_expanded}
    

def choose_best_candidate(inputs, perplexity, best_k, out_of, device):

    perplexity = rearrange(perplexity, '(b r) -> b r', r=out_of)
    # indices corresponding to the lowest perplexity
    best_k_perplexity, best_k_indices = torch.topk(perplexity, best_k, dim=1, largest=False)

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

def extend_tokens(inputs, assistant_tokens_expanded, target_tokens_expanded):            
    total_tokens = {}        
    total_tokens['input_ids'] = torch.cat([inputs['input_ids'], assistant_tokens_expanded, target_tokens_expanded], dim=-1)
    total_tokens['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(assistant_tokens_expanded), torch.ones_like(target_tokens_expanded)], dim=-1)
    
    # add assistant token to inputs
    inputs_with_assistant = {}
    inputs_with_assistant['input_ids'] = torch.cat([inputs['input_ids'], assistant_tokens_expanded], dim=-1)
    inputs_with_assistant['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(assistant_tokens_expanded)], dim=-1)

    return total_tokens, inputs_with_assistant
    



def attack_BEAST(tokenizer, model, prompts, targets, assistant_string, lookahead_length, num_adv_tokens, do_sample, temperature, top_p, top_k, k1, k2, batch_size):
    device = model.device

    all_attack_sentences = []
    all_attack_tokens = torch.empty((0, num_adv_tokens), dtype=torch.long)

    for batch_prompts, batch_targets in zip(batchify(prompts, batch_size), batchify(targets, batch_size)):
        # print which batch is being processed
        print(f"Processing batch {len(all_attack_sentences)//batch_size + 1} of {(len(prompts) + batch_size - 1) // batch_size}")
        # target tokens repeated
        target_tokens = get_target_tokens(tokenizer, batch_targets, lookahead_length).to(device)
        target_tokens_expanded = repeat(target_tokens['input_ids'], 'b t -> b r t', r=k1*k2)
        target_tokens_expanded = rearrange(target_tokens_expanded, 'b r t -> (b r) t')

        # model specific "ASSISTANT:" token repeated
        assistant_token = torch.tensor(tokenizer.encode(assistant_string, add_special_tokens=False)).to(device)
        assistant_tokens_expanded = repeat(assistant_token, 't -> b t', b=target_tokens_expanded.shape[0])

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # generate k1 beams
        inputs = generate_and_extend(model, tokenizer, inputs, max_new_tokens=1, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, k=k1)
        
        for _ in tqdm(range(1, num_adv_tokens), desc="Adversarial Token Generation"):
            
            # generate k2 candidates for each beam
            inputs = generate_and_extend(model, tokenizer, inputs, max_new_tokens=1, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, k=k2)
            
            # create total_tokens: input_tokens (with adv_tokens) + assistant_tokens + target_tokens
            total_tokens, inputs_with_assistant = extend_tokens(inputs, assistant_tokens_expanded, target_tokens_expanded)
            
            # calculate perplexity
            perplexity = calc_perplexity(inputs_with_assistant, total_tokens, model, batch_size=batch_size)

            # choose the best candidate
            inputs = choose_best_candidate(inputs, perplexity, best_k=k1, out_of=k1*k2, device=device)

        
        # create total_tokens: input_tokens (with adv_tokens) + assistant_tokens + target_tokens
        total_tokens = choose_best_candidate(copy.deepcopy(total_tokens), perplexity, best_k=k1, out_of=k1*k2, device=device)
        inputs_with_assistant = choose_best_candidate(copy.deepcopy(inputs_with_assistant), perplexity, best_k=k1, out_of=k1*k2, device=device)
        perplexity = calc_perplexity(inputs_with_assistant, total_tokens, model, batch_size=batch_size)
        # choose best out of k1 beams
        final_inputs = choose_best_candidate(inputs, perplexity, best_k=1, out_of=k1, device=device)

        # extract pure adversarial tokens
        final_attack_tokens = final_inputs['input_ids'][:, -num_adv_tokens:].detach().cpu()
        all_attack_tokens = torch.cat([all_attack_tokens, final_attack_tokens], dim=0)

        final_attack_sentences = tokenizer.batch_decode(final_inputs['input_ids'], skip_special_tokens=True)
        all_attack_sentences.extend(final_attack_sentences)

    return all_attack_sentences, all_attack_tokens


def evaluate_BEAST(model, tokenizer, adv_prompts, refusal_strings, batch_size, num_trials, do_sample, temperature, num_gen_tokens):
    successfull_attacks = []
    all_responses = []
    all_refusals = []

    for prompt_batch in tqdm(batchify(adv_prompts, batch_size), desc="Processing Batches"):
        
        refusals_batch = torch.zeros(len(prompt_batch), num_trials)
        batch_responses = [[] for _ in range(len(prompt_batch))]  # Temporary list to store responses for the current batch

        # the original paper tests for 5 different generations
        for i in range(num_trials):
            # Generate attacked prompts
            responses_batch = generate(model, tokenizer, prompt_batch, max_new_tokens=num_gen_tokens, do_sample=do_sample, temperature=temperature)
            # check if the response contains refusal strings
                    # Check if each response contains any of the refusal strings
            contains_refusal = torch.tensor([
                any(refusal in response for refusal in refusal_strings) 
                for response in responses_batch
            ], dtype=torch.float32)
            refusals_batch[:, i] = contains_refusal
            # Collect responses for this trial
            for idx, response in enumerate(responses_batch):
                batch_responses[idx].append(response)  # Append trial response for each prompt

        all_refusals.extend(refusals_batch.tolist())
        # save the successful attacks (no refusals in any of the generations
        successfull_attacks.extend((refusals_batch.sum(dim=1)==0).tolist())
        all_responses.extend(batch_responses)

    # calculate success rate
    success_rate = sum(successfull_attacks) / len(successfull_attacks)

    return success_rate, all_responses, all_refusals, successfull_attacks


def logging(adv_prompts_file, successfull_attacks, adv_prompts, adv_strings, num_trials, all_responses, all_refusals, success_rate):
    # Initialize W&B run
    run_name = adv_prompts_file.split('/')[1].replace('_formatted_attacked_prompts.pkl', '')
    wandb.init(project="Evaluate_BEAST", name=run_name)

    # Create a W&B table for detailed tracking

    response_cols = []
    for i in range(num_trials):
        response_cols.append(f"Response {i+1}")
        response_cols.append(f"Response {i+1} contains refusal")

    table = wandb.Table(columns=
        ["Success", "Full adversarial Prompt", "Adversarial string found by BEAST"] + response_cols)

    for success, prompt, adv_string, responses, refusals in zip(
        successfull_attacks, adv_prompts, adv_strings, all_responses, all_refusals
    ):
        row = [bool(success), prompt, adv_string]  # Start with non-trial data
        for i in range(num_trials):
            # Add response and refusal status for each trial
            row.append(responses[i] if i < len(responses) else None)
            row.append(bool(refusals[i]) if i < len(refusals) else None)
        table.add_data(*row)



    # extract parameters from file name
    # Define a regex pattern to match the parameters
    pattern = (
        r".*/(?P<model_name>[^/]+)_topk_(?P<top_k>None|\d+)_topp_(?P<top_p>[\d.]+)_"
        r"k1_(?P<k1>\d+)_k2_(?P<k2>\d+)_temp_(?P<temperature>[\d.]+)_"
        r"num_adv_tokens_(?P<num_adv_tokens>\d+)_lookahead_(?P<lookahead_length>\d+)"
        r".*\.pkl$"
    )

    # Match the pattern against the file name
    match = re.match(pattern, adv_prompts_file)

    params = None
    # Extract the parameters into a dictionary
    if match:
        params = match.groupdict()
        # Convert numeric values to integers or floats as appropriate
        params['model_name'] = params['model_name']
        params["top_k"] = None if params["top_k"] == "None" else int(params["top_k"])
        params["top_p"] = None if params["top_p"] == "None" else float(params["top_p"])
        params["k1"] = int(params["k1"])
        params["k2"] = int(params["k2"])
        params["temperature"] = None if params["temperature"] == "None" else float(params["temperature"])
        params["num_adv_tokens"] = int(params["num_adv_tokens"])
        params["lookahead_length"] = int(params["lookahead_length"])
        print(params)
    else:
        print("No match found.")

    params['num_trials'] = num_trials
    params['num_samples'] = len(adv_prompts)
    # Log the table and success rate
    wandb.log({"Adversarial Results": table})
    # Log parameters
    wandb.config.update(params)

    # Log the success rate as a scalar
    wandb.run.summary["Success Rate"] = success_rate

    for p in params:
        wandb.run.summary[p] = params[p]

    summary_note = (
        f"This experiment tests adversarial prompts with the {params['model_name']} model.\n"
        f"Success rate: {success_rate}\n"
        f"Number of samples tested: {params['num_samples']}\n"
        "Key parameters:\n"
        f"\t num_trials: {params['num_trials']}\n"
        f"\t temperature: {params['temperature']}\n"
        f"\t num_adv_tokens: {params['num_adv_tokens']}\n"
        f"\t lookahead_length: {params['lookahead_length']}\n"
    )

    # Log the summary note
    wandb.run.notes = summary_note

    # Finish W&B run
    wandb.finish()
