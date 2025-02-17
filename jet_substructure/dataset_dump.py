#  Copyright (C) 2021 Xilinx, Inc
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from argparse import ArgumentParser
from functools import reduce

import torch
from torch.utils.data import DataLoader

from train import configs, model_config, dataset_config, other_options, test
from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel, JetSubstructureLutModel
from models import JetSubstructureNeqModel, JetSubstructureLutModel, Quantized_JSC_LUT, Quantized_JSC_Verilog, QuantizedJSC_CNN_NEQ

def dump_io(model, data_loader, input_file, output_file, topology):
    input_quant = model.module_list[0].input_quant
    _, input_bitwidth = input_quant.get_scale_factor_bits()
    input_bitwidth = int(input_bitwidth)
    total_input_bits = model.module_list[0].in_channels*input_bitwidth* (model.module_list[0].seq_length + 2)
    print(f"Total input bits: {total_input_bits}")
    input_quant.bin_output()
    padding_tensor = torch.full((1, 1), 0, dtype=torch.int64)  # 2-bit padding
    with open(input_file, 'w') as i_f, open(output_file, 'w') as o_f:
        for data, target in data_loader:
            if topology == 'cnn':
                reshaped_data = data.view(data.size(0), 1, -1)
                print('shape of the data:', reshaped_data.shape)
                data_padded = torch.cat([padding_tensor.repeat(reshaped_data.shape[0], reshaped_data.shape[1], 1), reshaped_data, padding_tensor.repeat(reshaped_data.shape[0], reshaped_data.shape[1], 1)], dim=-1)
                print('shape of the data after padding:', data_padded.shape)
                x = input_quant(data_padded)
                indices = torch.argmax(target,dim=1)
                for i in range(x.shape[0]):
                # x_i = x[i,:]
                    x_i = x[i, :, :].flatten()
                    xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
                    xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
                    i_f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}\n")
                    o_f.write(f"{int(indices[i])}\n")
            else: 
                x = input_quant(data)
                indices = torch.argmax(target,dim=1)
                for i in range(x.shape[0]):
                    x_i = x[i,:]
                    xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
                    xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
                    i_f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}\n")
                    o_f.write(f"{int(indices[i])}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Dump the train and test datasets (after input quantization) into text files")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="jsc-s",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for evaluation (default: %(default)s)")
    parser.add_argument('--input-bitwidth', type=int, default=None,
        help="Bitwidth to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-bitwidth', type=int, default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)")
    parser.add_argument('--output-bitwidth', type=int, default=None,
        help="Bitwidth to use at the output (default: %(default)s)")
    parser.add_argument('--input-fanin', type=int, default=None,
        help="Fanin to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-fanin', type=int, default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)")
    parser.add_argument('--output-fanin', type=int, default=None,
        help="Fanin to use at the output (default: %(default)s)")
    parser.add_argument('--hidden-layers', nargs='+', type=int, default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)")
    parser.add_argument('--dataset-file', type=str, default='/home/student/CNN_LogicNets/jet_substructure/data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--dataset-config', type=str, default='/home/student/CNN_LogicNets/jet_substructure/config/yaml_IP_OP_config.yml',
        help="The file to use to configure the input dataset (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='/home/student/CNN_LogicNets/jet_substructure/jsc_s/',
        help="A location to store the output I/O text files (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, default='/home/student/CNN_LogicNets/jet_substructure/jsc_s/best_accuracy.pth',
        help="The checkpoint file which contains the model weights")
    parser.add_argument('--topology', type=str, choices=['cnn', 'fc'], default='cnn',
        help="The topology of the model (default: %(default(s))")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        if k == 'cuda':
            continue
        options_cfg[k] = config[k]

    # Fetch the test set
    dataset = {}
    dataset["train"] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="train")
    dataset["test"] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="test")
    train_loader = DataLoader(dataset["train"], batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=config['batch_size'], shuffle=False)

    # Instantiate the PyTorch model
    x, y = dataset["train"][0]
    if args.topology == 'cnn':
        model_cfg['input_length'] = 1 #len(x)
        model_cfg['sequence_length'] = len(x) 
        print('model_cfg:', model_cfg)
        model_cfg['output_length'] = len(y)
        model = QuantizedJSC_CNN_NEQ(model_cfg)
    else:
        model_cfg['input_length'] = len(x)
        print('model_cfg:', model_cfg)
        model_cfg['output_length'] = len(y)
        model = JetSubstructureNeqModel(model_cfg)

    # Load the model weights
    checkpoint = torch.load(options_cfg['checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint['model_dict'])

    # Test the PyTorch model
    print("Running inference on baseline model...")
    model.eval()
    baseline_accuracy, baseline_avg_roc_auc = test(model, test_loader, cuda=False, topology_type=args.topology)
    print("Baseline accuracy: %f" % (baseline_accuracy))
    print("Baseline AVG ROC AUC: %f" % (baseline_avg_roc_auc))

    # Run preprocessing on training set.
    train_input_file = config['log_dir'] + "/train_input.txt"
    train_output_file = config['log_dir'] + "/train_output.txt"
    test_input_file = config['log_dir'] + "/test_input.txt"
    test_output_file = config['log_dir'] + "/test_output.txt"
    print(f"Dumping train I/O to {train_input_file} and {train_output_file}")
    dump_io(model, train_loader, train_input_file, train_output_file)
    print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
    dump_io(model, test_loader, test_input_file, test_output_file)
