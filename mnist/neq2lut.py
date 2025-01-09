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

from model import MINSTmodelneq, MINSTmodellut, MINSTmodelver, QuantizedMNIST_NEQ, QuantizedMNIST_LUT, QuantizedMNIST_Verilog
from nn_layers import generate_truth_tables, lut_inference, module_list_to_verilog_module, SparseConv1dNeq  #type: ignore  
from train import test_mlp, test_cnn
from dataset_dump import dump_io

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run LUT-based synthesis on MNIST model.")
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for testing (default: 128)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer (default: 0)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate (default: 0.004)')
    parser.add_argument('--cuda', type=bool, default=False, help='Enable CUDA (default: False)')
    parser.add_argument('--log_dir', type=str, default='/home/student/Desktop/CNN_LogicNets/MINST/log_file', help='Directory for saving model checkpoints (default: /home/student/Desktop/Final/model_checkpts/log)')
    parser.add_argument('--checkpoint', type=str, default='/home/student/Desktop/CNN_LogicNets/MINST/log_file/best_model_mnist_1dcnn.pth', help='Path to checkpoint file (default: best_model_minst_logicnets.pth)')
    parser.add_argument('--seed', type=int, default=984237, help='Random seed (default: 42)')
    parser.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
    parser.add_argument('--add_registers', type=bool, default=True, help='Flag to add registers (default: False)')
    parser.add_argument('--topology', type=str, default='cnn', help='choose the toplogy for training and testing either linear or cnn')
    parser.add_argument('--dump-io', action='store_true', default=True, help="Dump I/O to the verilog LUT to a text file in the log directory")

    return parser.parse_args()

# Simplified configuration dictionary (will take values from argparse)
args = parse_args()

model_cfg_mlp = {
            "hidden_layers": [256, 100, 100, 100, 100],
            "input_bitwidth": 2,
            "hidden_bitwidth": 2,
            "output_bitwidth": 2,
            "input_fanin": 6,
            "degree": 4,
            "hidden_fanin": 3,
            "output_fanin": 3,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "checkpoint": None,
            "input_length": 256,
            "output_length": 10
        }

model_config_cnn = {
            "batch_size": args.batch_size,
            "input_length": 1,
            "sequence_length": 256, 
            "hidden_layers": [6] * 4 + [128] * 2,
            "output_length": 10,
            "padding": 1, 
            "1st_layer_in_f": 1536, #change whenever the sequence length changed 
            "input_bitwidth": 6,
            "hidden_bitwidth": 2,
            "output_bitwidth": 2,
            "input_fanin": 1,
            "conv_fanin": 2,
            "hidden_fanin": 6,
            "output_fanin": 6
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



# Fetch the test set
def get_test_loader(model_config):
    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST("mnist_data", download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False)
    return test_loader



# Main function for the synthesis process
def run_synthesis(config, other_options, model_arch):
    # Initialize the environment
    # device = initialize_model_and_env(config)
    
    # Fetch the test set
    test_loader = get_test_loader(config)
    
    # Instantiate the PyTorch model
    if model_arch == 'linear': 
        model = MINSTmodelneq(config)
    elif model_arch == 'cnn':
        model = QuantizedMNIST_NEQ(config)
    # Load the model weightslutmodel.
    checkpoint = torch.load(other_options["checkpoint"])
    model.load_state_dict(checkpoint['model_dict'])

    # Test the baseline model
    print("Running inference on baseline model...")
    if model_arch == 'linear': 
        baseline_accuracy = test_mlp(model, test_loader, cuda=other_options["cuda"])
    elif model_arch == 'cnn':
        baseline_accuracy = test_cnn(model, test_loader, options=other_options)
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    
    # Generate the truth tables in the LUT module
    print("Converting NEQs to LUTs...")
    if model_arch == 'linear': 
        lut_model = MINSTmodellut(config)
    elif model_arch == 'cnn':
        lut_model = QuantizedMNIST_LUT(config)
    # print(lut_model.module_list)
    lut_model.load_state_dict(checkpoint['model_dict'])
    for name, layer in lut_model.named_modules():
        if type(layer) == SparseConv1dNeq:
            print(f"Layer: {name}")
            # Clone the weights from conv and assign to flatconv
            trained_weights = layer.conv.weight.data.clone()
            layer.flatconv.weight.data = trained_weights

            # Print to verify the update
            print("Updated flatconv weights:")
            # print(layer.flatconv.weight.data)
            print('................................................................')
    print("Converting NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)
    
    # Test the LUT-based model
    print("Running inference on LUT-based model...")
    lut_inference(lut_model)
    lut_model.eval()
    if model_arch == 'linear': 
        lut_accuracy = test_mlp(lut_model, test_loader, cuda=other_options["cuda"])
    elif model_arch == 'cnn':
        lut_accuracy = test_cnn(lut_model, test_loader, options=other_options)
    print(f"LUT-Based Model accuracy: {lut_accuracy:.2f}%")

    # Save the model and LUT-based results
    model_save = {"model_dict": model.state_dict(), "test_accuracy": lut_accuracy}
    torch.save(model_save, os.path.join(other_options["log_dir"], "lut_based_model.pth"))
    
    # Generate the Verilog file
    print(f"Generating Verilog in {other_options['log_dir']}...")
    module_list_to_verilog_module(
        lut_model.module_list,
        "LogicNets_MINST",
        other_options["log_dir"],
        add_registers=other_options["add_registers"], 
        generate_bench = False
    )
    print(f"Top level entity stored at: {other_options['log_dir']}/logicnet.v ...")

    if args.dump_io:
        io_filename = other_options["log_dir"] + f"io_test_empty.txt"
        with open(io_filename, 'w') as f:
            pass # Create an empty file.
        print(f"Dumping verilog I/O to {io_filename}...")
        test_input_file = other_options['log_dir'] + "/test_input.txt"
        test_output_file = other_options['log_dir'] + "/test_output.txt"
        print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
        dump_io(model, test_loader, test_input_file, test_output_file)

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set additional parameters for the model
    if args.topology == 'linear':
        model_cfg = model_cfg_mlp
        model_arch = 'linear'
    elif args.topology == 'cnn':
        model_cfg = model_config_cnn
        model_arch = 'cnn'

    # Run the synthesis process
    run_synthesis(model_cfg, other_options, model_arch)
