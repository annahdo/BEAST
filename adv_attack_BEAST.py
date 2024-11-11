import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
import importlib
import pickle

from utils import prepare_input, attack_BEAST

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

    chat_format = {
        'user': "USER: ",
        'assistant': " ASSISTANT:",
        'system': "A chat between a curious user and an artificial intelligence assistant. "
                  "The assistant gives helpful, detailed, and polite answers to the user's questions."
    }
    
    # Load data
    data = pd.read_csv(args.data_file)
    prompts = list(data[args.prompt_tag])
    targets = list(data[args.target_tag])
    if args.num_data_points is not None:
        prompts = prompts[:args.num_data_points]
        targets = targets[:args.num_data_points]

    # Generate attacked prompts
    chat_format_attack = deepcopy(chat_format)
    chat_format_attack['assistant'] = ""
    formatted_prompts = prepare_input(chat_format_attack, prompts)
    assistant_string = chat_format['assistant']

    do_sample = False if args.temperature == 0 else True
    temperature = None if args.temperature == 0 else args.temperature
    attacked_prompts, attack_tokens = attack_BEAST(
        tokenizer, model, formatted_prompts, targets, assistant_string, args.lookahead_length, args.num_adv_tokens,
        do_sample, temperature, args.top_p, args.top_k, args.k1, args.k2, args.batch_size
    )
    
    chat_format_attacked = deepcopy(chat_format)
    chat_format_attacked['user'] = ""
    chat_format_attacked['system'] = ""

    formatted_attacked_prompts = prepare_input(chat_format_attacked, attacked_prompts)

    # automatic_filename_generation
    file_name = f"data/{args.model_name.split('/')[-1]}_topk_{args.top_k}_topp_{args.top_p}_k1_{args.k1}_k2_{args.k2}_temp_{args.temperature}_num_adv_tokens_{args.num_adv_tokens}_lookahead_{args.lookahead_length}"

    # Save outputs
    with open(file_name+'_formatted_attacked_prompts.pkl', 'wb') as file:
        pickle.dump(formatted_attacked_prompts, file)

    with open(file_name + '_attack_tokens.pkl', 'wb') as file:
        pickle.dump(attack_tokens, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line application for adversarial prompt generation using BEAST.")

    # Model and tokenizer arguments
    parser.add_argument("--model_name", type=str, default="lmsys/vicuna-7b-v1.5", help="Name of the model to load.")
    
    # Data arguments
    parser.add_argument("--data_file", type=str, default="data/harmful_behaviors.csv", help="Path to the CSV data file.")
    parser.add_argument("--prompt_tag", type=str, default="goal", help="Column name for prompts in the CSV file.")
    parser.add_argument("--target_tag", type=str, default="target", help="Column name for targets in the CSV file.")
    parser.add_argument("--num_data_points", type=int, default=None, help="Number of data points to consider.")
    
    # Attack parameters
    parser.add_argument("--top_k", type=int, default=None, help="Top k tokens to consider.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p cumulative probability.")
    parser.add_argument("--k1", type=int, default=15, help="Number of candidate beams in beam search.")
    parser.add_argument("--k2", type=int, default=15, help="Number of candidates per candidate evaluated.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. 0 means no sampling.")
    parser.add_argument("--num_adv_tokens", type=int, default=40, help="Number of adversarial tokens to add.")
    parser.add_argument("--lookahead_length", type=int, default=10, help="Number of target tokens used for objective.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    
    args = parser.parse_args()
    main(args)
