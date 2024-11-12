import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import wandb
import pickle
import os
import re

from utils import evaluate_BEAST, logging
from constants import refusal_strings

def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False,
        low_cpu_mem_usage=True
    ).to(device).eval()
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id


    # load adv prompts
    with open(args.adv_prompts_file, "rb") as f:
        adv_prompts = pickle.load(f)
    with open(args.adv_tokens_file, "rb") as f:
        adv_tokens = torch.tensor(pickle.load(f)).to(torch.long)

    adv_strings = tokenizer.batch_decode(adv_tokens, skip_special_tokens=True)


    do_sample = False if args.temperature == 0 or args.temperature==None else True
    temperature = None if args.temperature == 0 else args.temperature

    success_rate, all_responses, all_refusals, successfull_attacks = evaluate_BEAST(model, tokenizer, adv_prompts, refusal_strings, 
                                                               args.batch_size, args.num_trials, do_sample, temperature, args.num_gen_tokens)

    print(f"Success rate: {success_rate}")


    logging(args.adv_prompts_file, successfull_attacks, adv_prompts, 
            adv_strings, args.num_trials, all_responses, all_refusals, success_rate)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line application for adversarial prompt evaluation.")

    # Model and tokenizer arguments
    parser.add_argument("--model_name", type=str, default="lmsys/vicuna-7b-v1.5", help="Name of the model to load.")
    
    # Data arguments
    parser.add_argument("--adv_prompts_file", type=str, default="data/vicuna-7b-v1.5_topk_None_topp_1.0_k1_15_k2_15_temp_1.0_num_adv_tokens_20_lookahead_10_formatted_attacked_prompts.pkl", help="Path to the pkl file with adversarial prompts.")
    parser.add_argument("--adv_tokens_file", type=str, default="data/vicuna-7b-v1.5_topk_None_topp_1.0_k1_15_k2_15_temp_1.0_num_adv_tokens_20_lookahead_10_attack_tokens.pkl", help="Path to the pkl file with adversarial tokens.")

    # generation arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    parser.add_argument("--num_gen_tokens", type=int, default=50, help="Number of tokens to generate for model response.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. 0 means no sampling.")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of generation trials per adversarial prompt.")

    args = parser.parse_args()
    main(args)


