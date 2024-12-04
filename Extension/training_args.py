import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_layer", type=int, help="Layer to apply steering on")
    parser.add_argument("--layers", nargs="+", type=int, default=[10], help="Layers to process")
    parser.add_argument("--multiplier", type=float, help="Multiplier for the steering vector")
    parser.add_argument("--behavior", type=str, help="Behavior name")
    parser.add_argument("--test_data_path", type=str, help="Path to the test dataset JSON file")
    parser.add_argument("--cache_dir", type=str, default="/scratch/debangan.mishra", help="Cache directory")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Directory to save test results")
    parser.add_argument("--is_open_ended", type=bool, default=True, help="Open ended generation")
    parser.add_argument("--save_activations", action="store_true", default=False, help="Save intermediate activations")
    parser.add_argument("--data_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("--vec_pathname", type=str, help="Path to testing vector file")

    args = parser.parse_args()
    return args