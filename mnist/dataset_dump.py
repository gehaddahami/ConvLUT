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
from train import test_cnn, test_mlp, training_config, other_options



# Fetch the test set
def get_test_loader(training_config, other_options):
        transform = transforms.Compose([transforms.Resize((training_config['image_size'], training_config['image_size'])), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if other_options['mnist_fashion']: 
            test_dataset = datasets.FashionMNIST("fashion_mnist_data", download=True, train=False, transform=transform)
        else:
            test_dataset = datasets.MNIST("mnist_data", download=True, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)
        return test_loader


def dump_io(model, data_loader, input_file, output_file, training_config):
    input_quant = model.module_list[0].input_quant
    _, input_bitwidth = input_quant.get_scale_factor_bits()
    input_bitwidth = int(input_bitwidth)
    if other_options['topology'] == 'cnn':
        total_input_bits = model.module_list[0].in_channels*input_bitwidth* (model.module_list[0].seq_length + 2) # remove 2 if padding is not considered
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
    else:
        total_input_bits = model.module_list[0].in_features*input_bitwidth
        input_quant.bin_output()

        with open(input_file, 'w') as i_f, open(output_file, 'w') as o_f:
            for data, target in data_loader:
                data = data.reshape(-1, training_config["input_length"])
                x = input_quant(data)
                indices = target
                for i in range(x.shape[0]):
                    x_i = x[i,:]
                    # x_i = x[i, :, :].flatten()
                    xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
                    xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
                    i_f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}\n")
                    o_f.write(f"{int(indices[i])}\n")
