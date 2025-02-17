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

'''
Additional functionality and arguments are added for 1D-CNNs

Note that the extended repository is not configured for the following steps: 
1- simulate_pre_synthesis_verilog
2- Running out-of-context synthesis
3- simulate_post_synthesis_verilog

The reason for this is that the oh-my-xilinx and nitropartlibrary are not successfully incorporated into the project.
Hence, ensure that these steps are commented when running 1D-CNN configurations. 
'''

import os
import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

base_path = os.path.dirname(os.path.abspath(__file__))

# Append the absolute path to the src directory
sys.path.append(os.path.join(base_path, '../src/'))

from nn_layers import generate_truth_tables, lut_inference, module_list_to_verilog_module, SparseConv1dNeq  #type:ignore   

from train import configs, model_config, dataset_config, test
from dataset import JetSubstructureDataset
from dataset_dump import dump_io
from models import JetSubstructureNeqModel, JetSubstructureLutModel, Quantized_JSC_LUT, Quantized_JSC_Verilog, QuantizedJSC_CNN_NEQ
from synthesis import synthesize_and_get_resource_counts #type:ignore 
from utils import proc_postsynth_file  #type:ignore 

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "generate_bench": False,
    "add_registers": False,
    "simulate_pre_synthesis_verilog": True,
    "simulate_post_synthesis_verilog": True,
}

if __name__ == "__main__":
    parser = ArgumentParser(description="Synthesize convert a PyTorch trained model into verilog")
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
    parser.add_argument('--1st-layer-in-f', type=int, default=None,
        help="The input feature size of the first layer (default: %(default)s)")
    parser.add_argument('--padding', type=str, default=None,
        help="The padding to use for the input sequence (default: %(default)s)")
    parser.add_argument('--dataset-file', type=str, default='data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--clock-period', type=float, default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)")
    parser.add_argument('--dataset-config', type=str, default='config/yaml_IP_OP_config.yml',
        help="The file to use to configure the input dataset (default: %(default)s)")
    parser.add_argument('--dataset-split', type=str, default='test', choices=['train', 'test'],
        help="Dataset to use for evaluation (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="The checkpoint file which contains the model weights")
    parser.add_argument('--generate-bench', action='store_true', default=False,
        help="Generate the truth table in BENCH format as well as verilog (default: %(default)s)")
    parser.add_argument('--dump-io', action='store_true', default=True,
        help="Dump I/O to the verilog LUT to a text file in the log directory (default: %(default)s)")
    parser.add_argument('--add-registers', action='store_true', default=True,
        help="Add registers between each layer in generated verilog (default: %(default)s)")
    parser.add_argument('--simulate-pre-synthesis-verilog', action='store_true', default=False,
        help="Simulate the verilog generated by LogicNets (default: %(default)s)")
    parser.add_argument('--simulate-post-synthesis-verilog', action='store_true', default=False,
        help="Simulate the post-synthesis verilog produced by vivado (default: %(default)s)")
    parser.add_argument('--fpga-part', type=str, default="xcu280-fsvh2892-2L-e",
        help="Part to use for Vivado project (default: %(default)s)")
    parser.add_argument('--topology', type = str, default='cnn', 
        help="Type of topology to use (default: %(default)s)")
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
    dataset[args.dataset_split] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split=args.dataset_split)
    test_loader = DataLoader(dataset[args.dataset_split], batch_size=config['batch_size'], shuffle=False)

    # Instantiate the PyTorch model
    x, y = dataset[args.dataset_split][0]
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
    baseline_accuracy, baseline_avg_roc_auc = test(model, test_loader, False, topology_type=args.topology)
    print("Baseline accuracy: %f" % (baseline_accuracy))
    print("Baseline AVG ROC AUC: %f" % (baseline_avg_roc_auc))

    # Instantiate LUT-based model
    if args.topology == 'cnn':
        lut_model = Quantized_JSC_LUT(model_cfg)
        lut_model.load_state_dict(checkpoint['model_dict'])
        for name, layer in lut_model.named_modules():
            if type(layer) == SparseConv1dNeq:
                print(f"Layer: {name}")
                # Clone the weights from conv and assign to flatconv
                trained_weights = layer.conv.weight.data.clone()
                layer.flatconv.weight.data = trained_weights
                # Print to verify the update
                print("Updated flatconv weights:")
                print('................................................................')
    else:
        lut_model = JetSubstructureLutModel(model_cfg)
        lut_model.load_state_dict(checkpoint['model_dict'])

    # Generate the truth tables in the LUT module
    print("Converting to NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)

    # Test the LUT-based model
    print("Running inference on LUT-based model...")
    lut_inference(lut_model)
    lut_model.eval()
    lut_accuracy, lut_avg_roc_auc = test(lut_model, test_loader, False, topology_type=args.topology)
    print("LUT-Based Model accuracy: %f" % (lut_accuracy))
    print("LUT-Based AVG ROC AUC: %f" % (lut_avg_roc_auc))
    modelSave = {   'model_dict': lut_model.state_dict(),
                    'test_accuracy': lut_accuracy,
                    'test_avg_roc_auc': lut_avg_roc_auc}

    torch.save(modelSave, options_cfg["log_dir"] + "/lut_based_model.pth")

    print("Generating verilog in %s..." % (options_cfg["log_dir"]))
    module_list_to_verilog_module(lut_model.module_list, "logicnet", options_cfg["log_dir"], generate_bench=options_cfg["generate_bench"], add_registers=options_cfg["add_registers"])
    print("Top level entity stored at: %s/logicnet.v ..." % (options_cfg["log_dir"]))

    if args.dump_io:
        io_filename = options_cfg["log_dir"] + f"io_{args.dataset_split}.txt"
        with open(io_filename, 'w') as f:
            pass # Create an empty file.
        print(f"Dumping verilog I/O to {io_filename}...")

        test_input_file = config['log_dir'] + "/test_input.txt"
        test_output_file = config['log_dir'] + "/test_output.txt"
        print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
        dump_io(model, test_loader, test_input_file, test_output_file, topology=args.topology)

    else:
        io_filename = None

    # if args.simulate_pre_synthesis_verilog:
    #     print("Running inference simulation of Verilog-based model...")
    #     lut_model.verilog_inference(options_cfg["log_dir"], "logicnet.v", logfile=io_filename, add_registers=options_cfg["add_registers"])
    #     verilog_accuracy, verilog_avg_roc_auc = test(lut_model, test_loader, cuda=False)
    #     print("Verilog-Based Model accuracy: %f" % (verilog_accuracy))
    #     print("Verilog-Based AVG ROC AUC: %f" % (verilog_avg_roc_auc))

    # print("Running out-of-context synthesis")
    # ret = synthesize_and_get_resource_counts(options_cfg["log_dir"], "logicnet", fpga_part=args.fpga_part, clk_period_ns=args.clock_period, post_synthesis = 1)

    # if args.simulate_post_synthesis_verilog:
    #     print("Running post-synthesis inference simulation of Verilog-based model...")
    #     proc_postsynth_file(options_cfg["log_dir"])
    #     lut_model.verilog_inference(options_cfg["log_dir"]+"/post_synth", "logicnet_post_synth.v", io_filename, add_registers=options_cfg["add_registers"])
    #     post_synth_accuracy, post_synth_avg_roc_auc = test(lut_model, test_loader, cuda=False)
    #     print("Post-synthesis Verilog-Based Model accuracy: %f" % (post_synth_accuracy))
    #     print("Post-synthesis Verilog-Based AVG ROC AUC: %f" % (post_synth_avg_roc_auc))
    
