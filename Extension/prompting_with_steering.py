import json
import torch
from llama_wrapper import LlamaWrapper
import os
from training_args import parse_args
from tqdm import tqdm
from utils.tokenize import E_INST
import warnings

warnings.filterwarnings("ignore")
def load_steering_vector(behavior, layer, model_name, pathname = None):
    if pathname is not None:
        return torch.load(pathname)

    sanitized_model_name = model_name.replace("/", "_")
    vector_path = os.path.join("vectors", behavior, f"{sanitized_model_name}_layer{layer}.pt")
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found at {vector_path}")
    return torch.load(vector_path)


def process_item(item, model, steering_vector, layer, multiplier):
    question = item["question"]
    model.reset_all()
    model.set_add_activations(layer, -multiplier * steering_vector)
    model_output = model.generate_text(user_input=question, max_new_tokens=100)

    result = {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),  # Extract the relevant part of the output
    }

    if not args.is_open_ended:
        result["answer_matching_behavior"] = item["answer_matching_behavior"]
        result["answer_not_matching_behavior"] = item["answer_not_matching_behavior"]
    return result

def run_test(args):
    if args.vec_pathname != "":
        pathname = args.vec_pathname

    steering_vector = load_steering_vector(args.behavior, args.steering_layer, args.model_name, pathname)
    model = LlamaWrapper(cache_dir=args.cache_dir, model_name=args.model_name)
    steering_vector = steering_vector.to(model.device)

    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)

    results = []
    for item in tqdm(test_data, desc=f"Testing with layer {args.steering_layer}, multiplier {args.multiplier}"):
        result = process_item(
            item=item,
            model=model,
            steering_vector=steering_vector,
            layer=args.steering_layer,
            multiplier=args.multiplier,
        )
        results.append(result)

    results_path = os.path.join(args.output_dir, f"results_layer{args.steering_layer}_multiplier{args.multiplier}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    args = parse_args()
    run_test(args)
