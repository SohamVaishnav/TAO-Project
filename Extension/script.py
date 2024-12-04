import os

layers = [2,3,5,8,13,15]

for layer in layers:
    command = f"python prompting_with_steering.py --vec_pathname ./vectors/None/meta-llama_Llama-3.2-1B-Instruct_layer{layer}_trainable.pt  --model_name meta-llama/Llama-3.2-1B-Instruct --multiplier -1 --steering_layer {layer} --test_data_path ./test_dataset_open_ended.json"
    os.system(command)