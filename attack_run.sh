NUM_ADV_TOKENS=20
NUM_SAMPLES=10

python adv_attack_BEAST.py --model_name "lmsys/vicuna-7b-v1.5" --num_adv_tokens $NUM_ADV_TOKENS --num_data_points $NUM_SAMPLES
python adv_attack_BEAST.py --model_name "EleutherAI/pythia-70m" --num_adv_tokens $NUM_ADV_TOKENS --num_data_points $NUM_SAMPLES
python adv_attack_BEAST.py --model_name "EleutherAI/pythia-410m" --num_adv_tokens $NUM_ADV_TOKENS --num_data_points $NUM_SAMPLES
python adv_attack_BEAST.py --model_name "EleutherAI/pythia-1b" --num_adv_tokens $NUM_ADV_TOKENS --num_data_points $NUM_SAMPLES
python adv_attack_BEAST.py --model_name "EleutherAI/pythia-6.9b" --num_adv_tokens $NUM_ADV_TOKENS --num_data_points $NUM_SAMPLES





