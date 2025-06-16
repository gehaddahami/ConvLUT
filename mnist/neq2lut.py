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
from train import test_mlp, test_cnn, configs, model_config, training_config
from dataset_dump import dump_io



other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "topology": None, 
    "mnist_fashion": False,
    'add_registers': True,
    'dump_io': True
}


if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets MNIST Classification")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="mlp28",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
        help="Initial learning rate (default: %(default)s)")
    parser.add_argument('--weight_decay', type=float, default=0, 
        help='Weight decay for optimizer (default: 0)')
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
    parser.add_argument('--kernel-size', type=int, default=None,
        help="The kernel size to use for the convolutional layers (default: %(default)s)")
    parser.add_argument('--image-size', type=int, default=None,
        help="The size of the image (default: %(default)s)")
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
    parser.add_argument('--checkpoint', type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
    parser.add_argument('--topology', type=str, default='cnn', choices=['linear', 'cnn'],
        help='choose the toplogy for training and testing either linear or cnn')
    parser.add_argument('--mnist_fashion', type=bool, default=False, 
        help='choose between normal MNIST and Fashion MNIST')
    parser.add_argument('--dump-io', action='store_true', default=True,
        help="Dump I/O to the verilog LUT to a text file in the log directory")
    parser.add_argument('--add-registers', type=bool, default=True, 
        help='Flag to add registers (default: False)')

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
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]


    if not os.path.exists(options_cfg['log_dir']):
        os.makedirs(options_cfg['log_dir'])
        print("Directory created")

    # Set random seeds
        random.seed(train_cfg['seed'])
        np.random.seed(train_cfg['seed'])
        torch.manual_seed(train_cfg['seed'])
        os.environ['PYTHONHASHSEED'] = str(train_cfg['seed'])
        if options["cuda"]:
            torch.cuda.manual_seed_all(train_cfg['seed'])
            torch.backends.cudnn.deterministic = True


    def get_test_loader(training_config, other_options):
        transform = transforms.Compose([transforms.Resize((training_config['image_size'], training_config['image_size'])), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if other_options['mnist_fashion']: 
            test_dataset = datasets.FashionMNIST("fashion_mnist_data", download=True, train=False, transform=transform)
        else:
            test_dataset = datasets.MNIST("mnist_data", download=True, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)
        return test_loader

    test_loader = get_test_loader(train_cfg, options_cfg)

    if options_cfg['topology'] == 'linear': 
            model = MINSTmodelneq(model_cfg)
            checkpoint = torch.load(options_cfg["checkpoint"])
            model.load_state_dict(checkpoint['model_dict'])


            print("Running inference on baseline model...")
            baseline_accuracy = test_mlp(model, test_loader, options_cfg, train_cfg)
            print(f"Baseline accuracy: {baseline_accuracy:.2f}%")

            print("Converting NEQs to LUTs...")
            lut_model = MINSTmodellut(model_cfg)
            lut_model.load_state_dict(checkpoint['model_dict'])
            generate_truth_tables(lut_model, verbose=True)

            # Test the LUT-based model
            print("Running inference on LUT-based model...")
            lut_inference(lut_model)
            lut_model.eval()
            lut_accuracy = test_mlp(lut_model, test_loader, options, train_cfg)
            print(f"LUT-Based Model accuracy: {lut_accuracy:.2f}%")

            # Save the model and LUT-based results
            model_save = {"model_dict": model.state_dict(), "test_accuracy": lut_accuracy}
            torch.save(model_save, os.path.join(options_cfg["log_dir"], "lut_based_model.pth"))

    
    elif options_cfg['topology'] == 'cnn':
            model = QuantizedMNIST_NEQ(model_cfg)
            checkpoint = torch.load(options_cfg["checkpoint"])
            model.load_state_dict(checkpoint['model_dict'])


            print("Running inference on baseline model...")
            baseline_accuracy = test_cnn(model, test_loader, options=options_cfg)
            print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
        
            print("Converting NEQs to LUTs...")
            lut_model = QuantizedMNIST_LUT(model_cfg)
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
            lut_accuracy = test_cnn(lut_model, test_loader, options)
            print(f"LUT-Based Model accuracy: {lut_accuracy:.2f}%")

            # Save the model and LUT-based results
            model_save = {"model_dict": model.state_dict(), "test_accuracy": lut_accuracy}
            torch.save(model_save, os.path.join(options_cfg["log_dir"], "lut_based_model.pth"))


    
    # Generate the Verilog file
    print(f"Generating Verilog in {options_cfg['log_dir']}...")
    module_list_to_verilog_module(
        lut_model.module_list,
        "LogicNets_MINST",
        options_cfg["log_dir"],
        add_registers=options_cfg["add_registers"], 
        generate_bench = False
    )
    print(f"Top level entity stored at: {options_cfg['log_dir']}/logicnet.v ...")

    if args.dump_io:
        io_filename = options_cfg["log_dir"] + f"io_test_empty.txt"
        with open(io_filename, 'w') as f:
            pass # Create an empty file.
        print(f"Dumping verilog I/O to {io_filename}...")
        test_input_file = options_cfg['log_dir'] + "/test_input.txt"
        test_output_file = options_cfg['log_dir'] + "/test_output.txt"
        print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
        dump_io(model, test_loader, test_input_file, test_output_file, train_cfg)
