import argparse
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

base_path = os.path.dirname(os.path.abspath(__file__))

# Append the absolute path to the src directory
sys.path.append(os.path.join(base_path, '../src/'))

print(sys.path) 
# Importing the model and the data loader
from dataset import Radioml_18
from model import QuantizedRadiomlNEQ, QuantizedRadiomlLUT
from nn_layers import generate_truth_tables, lut_inference, module_list_to_verilog_module, SparseConv1dNeq #type: ignore
from train_test_loops import train_logicnets, val_test_pytorch

# Default options
options = {
    'cuda': None,
    'log_dir': '/home/student/Desktop/CNN_LogicNets/radioml/log',
    'checkpoint': '/home/student/Desktop/CNN_LogicNets/radioml/log/checkpoint_radioml.pth'
}

def parse_args():
    parser = argparse.ArgumentParser(description="LUT-based model inference, synthesize and convert model into Verilog")
    parser.add_argument('--dataset_path', type=str, default='/home/student/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for data loaders')
    parser.add_argument('--snr_ratio', type=int, default=25, help='The desired signal-to-noise ratio')
    parser.add_argument('--sequence_length', type=int, default=64, help='The desired signal sequence length')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--t0', type=int, default=5, help='T_0 parameter for CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--t_mult', type=int, default=1, help='T_mult parameter for CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save the model checkpoint')
    parser.add_argument('--add_registers', type=bool, default=True, help='Flag to add registers (default: False)')

    return parser.parse_args()

def load_data(dataset_path, batch_size, snr):
    args = parse_args()
    dataset = Radioml_18(dataset_path, sequence_length=args.sequence_length, snr_ratio=snr)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.validation_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)
    return train_loader, validation_loader, test_loader



def main():
    args = parse_args()

    # Create the log directory if it does not exist
    if not os.path.exists(options['log_dir']):
        os.makedirs(options['log_dir'])
        print(f'Log directory {options["log_dir"]} created.')

    # Load data
    train_loader, validation_loader, test_loader = load_data(args.dataset_path, args.batch_size, args.snr_ratio)

    model_config = {
        "input_length": 2,
        "sequence_length": args.sequence_length,
        "hidden_layers": [4] * 4 + [32] * 2,
        "output_length": 24,
        "padding": 1,
        "1st_layer_in_f": 16,
        "input_bitwidth": 3,
        "hidden_bitwidth": 3,
        "output_bitwidth": 3,
        "input_fanin": 2,
        "conv_fanin": 2,
        "hidden_fanin": 6,
        "output_fanin": 6
    }
    # Create and load the model
    model = QuantizedRadiomlLUT(model_config)

    if options['checkpoint']:
        print(f'Loading checkpoint from {options["checkpoint"]}')
        checkpoint = torch.load(options['checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_dict'])
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint provided. Training from scratch.")
    
    # Evaluate the model
    print("Running inference loop on baseline model...")
    model.eval()
    baseline_accuracy = val_test_pytorch(model, test_loader, options)
    print(f"Baseline accuracy: {baseline_accuracy:.6f}")

    # Instantiate and convert to LUT-based model
    lut_model = QuantizedRadiomlLUT(model_config)
    lut_model.load_state_dict(checkpoint['model_dict'])
    for name, layer in lut_model.named_modules():
        if type(layer) == SparseConv1dNeq:
            print(f"Layer: {name}")
            print("Original conv weights:")
            print(layer.conv.weight.data)
            print('................................................................')
            print("Original flatconv weights:")
            print(layer.flatconv.weight.data)
            print('................................................................')

            # Clone the weights from conv and assign to flatconv
            trained_weights = layer.conv.weight.data.clone()
            layer.flatconv.weight.data = trained_weights

            # Print to verify the update
            print("Updated flatconv weights:")
            print(layer.flatconv.weight.data)
            print('................................................................')
    print("Converting NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)

    # Optionally save the model
    print("Running inference on LUT-based model...")
    lut_inference(lut_model)
    lut_model.eval()
    print('Evaluating the model')
    lut_accuracy = val_test_pytorch(lut_model, test_loader, options)
    print("LUT-Based Model accuracy: %f" % (lut_accuracy))
    model_save = {
        'model_dict': lut_model.state_dict(),
        'test_accuracy': lut_accuracy
    }

    torch.save(model_save, os.path.join(options['log_dir'], "lut_based_radioml.pth"))

    # Generate the Verilog file
    print(f"Generating Verilog in {options['log_dir']}...")
    module_list_to_verilog_module(
        lut_model.module_list,
        "LogicNets_MINST",
        options["log_dir"],
        add_registers=True, 
        generate_bench = False
    )
    print(f"Top level entity stored at: {options['log_dir']}/logicnet_radioml.v ...")


if __name__ == "__main__":
    main()
