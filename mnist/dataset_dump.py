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

import argparse
from functools import reduce

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
                           
from model import QuantizedMNIST_NEQ, MINSTmodelneq
from train import test_cnn, test_mlp



# Fetch the test set
def get_test_loader(model_config):
    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST("mnist_data", download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False)
    return test_loader


def dump_io(model, data_loader, input_file, output_file):
    input_quant = model.module_list[0].input_quant
    _, input_bitwidth = input_quant.get_scale_factor_bits()
    input_bitwidth = int(input_bitwidth)
    total_input_bits = model.module_list[0].in_channels*input_bitwidth* model.module_list[0].seq_length
    input_quant.bin_output()

    # padding consideration 
    padding_tensor = torch.full((1, 1), 0, dtype=torch.int64)  # 2-bit padding
    with open(input_file, 'w') as i_f, open(output_file, 'w') as o_f:
        for data, target in data_loader:
            data = data.view(data.size(0), 1, -1)
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



options = {
    'cuda' : False,
    'log_dir' : '/home/student/Desktop/CNN_LogicNets/MINST/log_file', 
    'checkpoint' : '/home/student/Desktop/CNN_LogicNets/MINST/log_file/best_model_mnist_1dcnn.pth' 
    }
	

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
    return parser.parse_args()

    

def create_model(model_arch, model_config_cnn, model_config_mlp):
    args = parse_args()
    if model_arch == 'cnn':
        model_config = model_config_cnn
        model = QuantizedMNIST_NEQ(model_config=model_config)
    elif model_arch == 'mlp':
        model_config = model_config_mlp
        model = MINSTmodelneq(model_config=model_config)
    return model

def main(): 
    if not os.path.exists(options['log_dir']):
        os.makedirs(options['log_dir'])
        print('directory created..................................')
    
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
    # Load data
    test_loader = get_test_loader(model_config_cnn)

    # Create model
    model = create_model(model_arch= args.topology, model_config_cnn=model_config_cnn, model_config_mlp=model_cfg_mlp)
    
    # load from checkpoint if available: 
    if options['checkpoint'] is not None:
        print(f'Loading pre-trained checkpoint {options["checkpoint"]}')
        checkpoint = torch.load(options['checkpoint'], map_location = 'cpu')
        model.load_state_dict(checkpoint['model_dict'])
        print(f'Checkpoint loaded successfully')

    # Evaluate the model
    print("Running inference loop on baseline model...")
    model.eval()
    if args.topology == 'cnn':
        baseline_accuracy = test_cnn(model, test_loader, options)
    else:
        baseline_accuracy = test_mlp(model, test_loader, cuda=options["cuda"])
    print(f"Baseline accuracy: {baseline_accuracy:.6f}")

        # Run preprocessing on training set.
    # train_input_file = options['log_dir'] + "/train_input.txt"
    # train_output_file = options['log_dir'] + "/train_output.txt"
    test_input_file = options['log_dir'] + "/test_input.txt"
    test_output_file = options['log_dir'] + "/test_output.txt"
    # print(f"Dumping train I/O to {train_input_file} and {train_output_file}")
    # dump_io(model, train_loader, train_input_file, train_output_file)
    print(f"Dumping test I/O to {test_input_file} and {test_output_file}")
    dump_io(model, test_loader, test_input_file, test_output_file)
    

if __name__ == "__main__":
    main()