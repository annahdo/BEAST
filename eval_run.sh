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
    
    # Build the expected file paths
    PROMPT_FILE="$DATA_DIR/${SHORT_MODEL_NAME}_topk_None_topp_1.0_k1_15_k2_15_temp_1.0_num_adv_tokens_20_lookahead_10_formatted_attacked_prompts.pkl"
    TOKEN_FILE="$DATA_DIR/${SHORT_MODEL_NAME}_topk_None_topp_1.0_k1_15_k2_15_temp_1.0_num_adv_tokens_20_lookahead_10_attack_tokens.pkl"

    # Check if both files exist
    if [[ -f "$PROMPT_FILE" && -f "$TOKEN_FILE" ]]; then
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
