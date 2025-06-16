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

import os
import sys 
import torch
from argparse import ArgumentParser
from functools import reduce
from torch.utils.data import DataLoader

                            
from dataset import MITBIHDataset
from model import ECG_NEQ
from train_test_loops import test
from train import configs, model_config, training_config, other_options, test_ecg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def dump_io(model, data_loader, input_file, output_file):
    input_quant = model.module_list[0].input_quant
    _, input_bitwidth = input_quant.get_scale_factor_bits()
    input_bitwidth = int(input_bitwidth)
    total_input_bits = model.module_list[0].in_channels*input_bitwidth* (model.module_list[0].seq_length + 2)
    print(f"Total input bits: {total_input_bits}")
    input_quant.bin_output()

    # padding consideration 
    padding_tensor = torch.full((1, 1), 0, dtype=torch.int64)  # 2-bit padding
    with open(input_file, 'w') as i_f, open(output_file, 'w') as o_f:
        for data, target in data_loader:
            print(f'The shape of data before applying the padding tensor is {data.shape}')
            data_padded = torch.cat([padding_tensor.repeat(data.shape[0], data.shape[1], 1), data, padding_tensor.repeat(data.shape[0], data.shape[1], 1)], dim=-1)
            print(f'The shape of data after applying the padding tensor is {data_padded.shape}')
            print('-----------------------------------------------------------------------------')
            x = input_quant(data_padded)
            indices = target
            for i in range(x.shape[0]):
                # x_i = x[i,:]
                x_i = x[i, :, :].flatten()
                xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
                xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
                i_f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}\n")
                o_f.write(f"{int(indices[i])}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets ECG Classification")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="cnn-a",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--dataset_path', type=str, default="./ecg_data",
        help="Path to the dataset (default: %(default)s)")
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



    # Fetch the dataloaders 
    dataset = MITBIHDataset(records_path=options_cfg['dataset_path'], sequence_length=train_cfg['sequence_length'])
    train_loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], sampler=dataset.train_sampler)
    test_loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], sampler=dataset.test_sampler)

    # Load the model
    model = ECG_NEQ(model_config=model_cfg)


    # load from checkpoint if available: 
    if options_cfg['checkpoint'] is not None:
        print(f'Loading pre-trained checkpoint {options_cfg["checkpoint"]}')
        checkpoint = torch.load(options_cfg['checkpoint'], map_location = 'cpu')
        model.load_state_dict(options_cfg['model_dict'])
        model_loaded_from_checkpoint = True
        print(f'Checkpoint loaded successfully')

    # Evaluate the model
    print("Running inference loop on baseline model...")
    model.eval()
    baseline_accuracy = test_ecg(model, test_loader, options_cfg, train_cfg)
    print(f"Baseline accuracy: {baseline_accuracy:.6f}")

        # Run preprocessing on training set.
    train_input_file = options_cfg['log_dir'] + "/train_input.txt"
    train_output_file = options_cfg['log_dir'] + "/train_output.txt"
    test_input_file = options_cfg['log_dir'] + "/test_input.txt"
    test_output_file = options_cfg['log_dir'] + "/test_output.txt"
    print(f"Dumping train I/O to {train_input_file} and {train_output_file}")
    dump_io(model, train_loader, train_input_file, train_output_file)
    print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
    dump_io(model, test_loader, test_input_file, test_output_file)
    
