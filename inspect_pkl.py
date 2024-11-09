import pickle as pkl
from transformers import AutoTokenizer

# Specify the file path to your log file
file_path = "data/vicuna_k1=15_k2=15_length=40_0_1_ngram=1.pkl"

# Load the file
with open(file_path, "rb") as file:
    log = pkl.load(file)

# Initialize the tokenizer for decoding
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

# Iterate over the log and decode results
for index, (output, _, _) in log.items():
    prompt_tokens = output[0][0][-1]
    prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    print(f"Prompt {index}:\n{prompt}\n")
