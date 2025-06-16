# This file is Part of Conv-LUT
# Conv-LUT is based on LogicNets   

# Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from argparse import ArgumentParser
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

from dataset import Radioml_18
from dataset_dump import dump_io
from model import QuantizedRadiomlNEQ, QuantizedRadiomlLUT
from nn_layers import generate_truth_tables, lut_inference, module_list_to_verilog_module, SparseConv1dNeq #type: ignore
from train_test_loops import test_logicnets
from train import configs, model_config, dataset_config, train_cfg

# Default options
other_options = {
    'cuda': None,
    'log_dir': None,
    'checkpoint': None, 
    'add_registers': True,
    'dump_io': True
}

if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets Jet Substructure Classification (RadioML Example)")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="psk-5",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
        help="Initial learning rate (default: %(default)s)")
    parser.add_argument('--cuda', action='store_true', default=False,
        help="Train on a GPU (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=2025,
        help="Seed to use for RNG (default: %(default)s)")
    parser.add_argument('--input_length', type=int, default=None,
            help="Length of input to use (default: %(default)s)")
    parser.add_argument('--output_length', type=int, default=None,
        help="Length of output to use (default: %(default)s)")
    parser.add_argument('--input-bitwidth', type=int, default=None,
        help="Bitwidth to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-bitwidth', type=int, default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)")
    parser.add_argument('--output-bitwidth', type=int, default=None,
        help="Bitwidth to use at the output (default: %(default)s)")
    parser.add_argument('--input-fanin', type=int, default=None,
        help="Fanin to use at the input (default: %(default)s)")
    parser.add_argument('--conv-fanin', type=int, default=None,
        help="Fanin to use for the convolutional layers (default: %(default)s)")
    parser.add_argument('--hidden-fanin', type=int, default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)")
    parser.add_argument('--output-fanin', type=int, default=None,
        help="Fanin to use at the output (default: %(default)s)")
    parser.add_argument('--hidden-layers', nargs='+', type=int, default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)")
    parser.add_argument('--sequence-length', nargs='+', type=int, default=None,
        help="The length of the input sequence (default: %(default)s)")
    parser.add_argument('--snr-ratio', type=int, default=25,
        help="SNR ratio to use (default: %(default)s)")
    parser.add_argument('--selected_modulations', nargs='+', type=str, default=None,
        help="The modulations to include in the dataset (default: %(default)s)")
    parser.add_argument('--1st-layer-in-f', type=int, default=None,
        help="The input feature size of the first layer (default: %(default)s)")
    parser.add_argument('--padding', type=str, default=None,
        help="The padding to use for the input sequence (default: %(default)s)")
    parser.add_argument('--t0', type=int, default=None,
        help="T_0 parameter for CosineAnnealingWarmRestarts scheduler (default: %(default)s)")
    parser.add_argument('--t-mult', type=int, default=None,
        help="T_mult parameter for CosineAnnealingWarmRestarts scheduler (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--dataset_path', type=str, default='/home/student/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
    parser.add_argument('--dump-io', action='store_true', default=False,
        help="Dump I/O to the verilog LUT to a text file in the log directory")
    parser.add_argument('--add-registers', action='store_true', default=False,
                        help="Add registers to the model")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    training_cfg = {}
    for k in train_cfg.keys():
        training_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options = {}
    for k in other_options.keys():
        options[k] = config[k]



if not os.path.exists(options['log_dir']):
        os.makedirs(options['log_dir'])
        print(f'Log directory {options["log_dir"]} created.')

# Fetch datat: 
dataset = Radioml_18(dataset_cfg['dataset_path'], sequence_length=training_cfg['sequence_length'], snr_ratio=training_cfg['snr_ratio'], selected_modulations=training_cfg['selected_modulations'])
test_loader = DataLoader(dataset, batch_size=training_cfg['batch_size'], sampler=dataset.test_sampler)


# Fetch model:
model = QuantizedRadiomlNEQ(model_cfg)
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
baseline_accuracy = test_logicnets(model, test_loader, options, dataset, test=True)
print(f"Baseline accuracy: {baseline_accuracy:.6f}")


# Instantiate and convert to LUT-based model
lut_model = QuantizedRadiomlLUT(model_cfg)
lut_model.load_state_dict(checkpoint['model_dict'])
for name, layer in lut_model.named_modules():
    if type(layer) == SparseConv1dNeq:
        print(f"Layer: {name}")

        trained_weights = layer.conv.weight.data.clone()
        layer.flatconv.weight.data = trained_weights

        # Print to verify the update
        print("Updated flatconv weights:")
        print('................................................................')

print("Converting NEQs to LUTs...")
generate_truth_tables(lut_model, verbose=True)

# Optionally save the model
print("Running inference on LUT-based model...")
lut_inference(lut_model)
lut_model.eval()
print('Evaluating the model')
lut_accuracy = test_logicnets(lut_model, test_loader, options, dataset, test=False)
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
    "LogicNets_Radioml",
    options["log_dir"],
    add_registers=options['add_registers'], 
    generate_bench = False
)
print(f"Top level entity stored at: {options['log_dir']}/logicnet_radioml.v ...")

if options['dump_io']:
    io_filename = options["log-dir"] + f"io_test_empty.txt"
    with open(io_filename, 'w') as f:
        pass # Create an empty file.
    print(f"Dumping verilog I/O to {io_filename}...")
    test_input_file = options['log-dir'] + "/test_input.txt"
    test_output_file = options['log-dir'] + "/test_output.txt"
    print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
    dump_io(model, test_loader, test_input_file, test_output_file)



