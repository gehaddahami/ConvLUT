import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

## in the verilog code change the nn_layers.sparselinear when resolved 
# Importing the model and the data loader
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import MINSTmodelneq, MINSTmodellut, MINSTmodelver
from nn_layers import generate_truth_tables, lut_inference, module_list_to_verilog_module  #type: ignore  
from train import test

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run LUT-based synthesis on MNIST model.")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing (default: 128)')
    parser.add_argument('--cuda', type=bool, default=False, help='Enable CUDA (default: False)')
    parser.add_argument('--log_dir', type=str, default='/home/student/Desktop/CNN_LogicNets/MINST/log_file', help='Directory for saving model checkpoints (default: /home/student/Desktop/Final/model_checkpts/log)')
    parser.add_argument('--checkpoint', type=str, default='/home/student/Desktop/CNN_LogicNets/MINST/log_file/best_model_minst_logicnets.pth', help='Path to checkpoint file (default: best_model_minst_logicnets.pth)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
    parser.add_argument('--add_registers', type=bool, default=True, help='Flag to add registers (default: False)')
    
    return parser.parse_args()

# Simplified configuration dictionary (will take values from argparse)
args = parse_args()

model_cfg = {
    "batch_size": args.batch_size,
    "input_bitwidth": 2,
    "hidden_bitwidth": 2,
    "output_bitwidth": 2,
    "input_fanin": 6,
    "degree": 4,
    "hidden_fanin": 6,
    "output_fanin": 6,
    "hidden_layers": [256, 100, 100, 100, 100],
    "dataset_split": "test",
    "log_dir": args.log_dir,
    "seed": args.seed,
    "device": args.device,  # GPU device index
    "checkpoint": args.checkpoint,
    "add_registers": args.add_registers
    }

# Options that were previously hardcoded

other_options = {
    "seed": args.seed,
    "cuda": args.cuda,
    "device": args.device,
    "log_dir": args.log_dir,
    "checkpoint": args.checkpoint,
    "add_registers": args.add_registers
}


# Function to initialize the model and environment
def initialize_model_and_env(config):
    # Set random seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    os.environ["PYTHONHASHSEED"] = str(config["seed"])

    # Set CUDA configurations
    if torch.cuda.is_available() and config.get("cuda", False):
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(config["device"])
        device = torch.device(f'cuda:{config["device"]}')
    else:
        device = torch.device('cpu')
    
    return device

# Fetch the test set
def get_test_loader(model_cfg2):
    dataset = datasets.MNIST(
        "mnist_data",
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    
    test_loader = DataLoader(
        dataset, batch_size=model_cfg2["batch_size"], shuffle=False
    )
    
    return test_loader

# Main function for the synthesis process
def run_synthesis(config, other_options):
    # Initialize the environment
    device = initialize_model_and_env(config)
    
    # Fetch the test set
    test_loader = get_test_loader(config)
    
    # Instantiate the PyTorch model
    model = MINSTmodelneq(config).to(device)
    
    # Load the model weightslutmodel.
    checkpoint = torch.load(config["checkpoint"])
    model.load_state_dict(checkpoint)

    # Test the baseline model
    print("Running inference on baseline model...")
    baseline_accuracy = test(model, test_loader, cuda=other_options["cuda"])
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    
    # Generate the truth tables in the LUT module
    print("Converting NEQs to LUTs...")
    lut_model = MINSTmodellut(config)
    print(lut_model.module_list)
    lut_model.load_state_dict(checkpoint)
    generate_truth_tables(lut_model, verbose=True)
    
    # Test the LUT-based model
    print("Running inference on LUT-based model...")
    lut_inference(lut_model)
    lut_model.eval()
    lut_accuracy = test(lut_model, test_loader, cuda=other_options["cuda"])
    print(f"LUT-Based Model accuracy: {lut_accuracy:.2f}%")
    
    # Save the model and LUT-based results
    model_save = {"model_dict": model.state_dict(), "test_accuracy": lut_accuracy}
    torch.save(model_save, os.path.join(config["log_dir"], "lut_based_model.pth"))
    
    # Generate the Verilog file
    print(f"Generating Verilog in {config['log_dir']}...")
    module_list_to_verilog_module(
        lut_model.module_list,
        "LogicNets_MINST",
        config["log_dir"],
        add_registers=config["add_registers"], 
        generate_bench = False
    )
    print(f"Top level entity stored at: {config['log_dir']}/logicnet.v ...")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    

    
    # Set additional parameters for the model
    model_cfg["input_length"] = 784
    model_cfg["output_length"] = 10

    # Run the synthesis process
    run_synthesis(model_cfg, other_options)
