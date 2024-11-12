#!/bin/bash

# Set the directory containing your files
DATA_DIR="data"

# Define the list of potential model names
MODELS=(
    "EleutherAI/pythia-70m"
    "EleutherAI/pythia-410m"
    "lmsys/vicuna-7b-v1.5"
    "EleutherAI/pythia-6.9b"
    "EleutherAI/pythia-1b"
)

# Iterate over each model
for MODEL in "${MODELS[@]}"; do
    # Extract the short model name (everything after the last /)
    SHORT_MODEL_NAME=$(echo "$MODEL" | awk -F/ '{print $NF}')

    # Find all files in the directory that match the required patterns
    PROMPT_FILE=$(find "$DATA_DIR" -type f -name "${SHORT_MODEL_NAME}*_formatted_attacked_prompts.pkl")
    TOKEN_FILE=$(find "$DATA_DIR" -type f -name "${SHORT_MODEL_NAME}*_attack_tokens.pkl")

    # Check if both files exist
    if [[ -n "$PROMPT_FILE" && -n "$TOKEN_FILE" ]]; then
        # Call the evaluation program with the appropriate arguments
        python eval_BEAST.py \
            --model_name "$MODEL" \
            --adv_prompts_file "$PROMPT_FILE" \
            --adv_tokens_file "$TOKEN_FILE" \
            --batch_size 16 \
            --num_gen_tokens 50 \
            --temperature 1.0 \
            --num_trials 5
    else
        echo "Skipping $MODEL: Required files not found."
    fi
done
