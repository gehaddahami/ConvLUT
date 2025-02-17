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
In this files, CNN topology is added. Pooling layer are not added to this file. 
If pooling is to be applied, the module_layer construct is to be further modified
'''

# Imports
import torch
import torch.nn as nn

from functools import reduce
from os.path import realpath
from torch import nn 

from brevitas.quant import IntBias
from brevitas.nn import QuantReLU, QuantIdentity, QuantSigmoid, QuantHardTanh
from brevitas.core.scaling import ScalingImplType
from brevitas.core.quant import QuantType

from pyverilator import PyVerilator

import sys 
import os

# # Get the absolute path of the directory where model.py is located
base_path = os.path.dirname(os.path.abspath(__file__))

# # Append the absolute path to the src directory
sys.path.append(os.path.join(base_path, '../src/'))

# Importing functions from the directory 
from nn_layers import SparseConv1dNeq, SparseLinearNeq, pooling_layer, RandomFixedSparsityMask2D, RandomFixedSparsityConv1DMask 
from quant import QuantBrevitasActivation, ScalarBiasScale, ScalarScaleBias 

from quant import QuantBrevitasActivation, ScalarBiasScale 
from nn_layers import SparseLinearNeq, RandomFixedSparsityMask2D 
from init import random_restrict_fanin 

# MLP linear models
class JetSubstructureNeqModel(nn.Module):
    def __init__(self, model_config):
        super(JetSubstructureNeqModel, self).__init__()
        self.model_config = model_config
        self.num_neurons = [model_config["input_length"]] + model_config["hidden_layers"] + [model_config["output_length"]]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i-1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
                input_quant = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["input_bitwidth"], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.CONST), pre_transforms=[bn_in, input_bias])
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["input_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=input_quant, output_quant=output_quant, mask=mask, first_linear=False)
                layer_list.append(layer)
            elif i == len(self.num_neurons)-1:
                output_bias_scale = ScalarBiasScale(bias_init=0.33)
                output_quant = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["output_bitwidth"], max_val=1.33, min_val=-2.0, narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=[output_bias_scale])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["output_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, mask=mask, apply_input_quant=False, first_linear=False)
                layer_list.append(layer)
            else:
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, min_val=-2.0, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["hidden_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, mask=mask, apply_input_quant=False, first_linear=False)
                layer_list.append(layer)
        self.module_list = nn.ModuleList(layer_list)
        self.is_verilog_inference = False
        self.latency = 1
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None

    def verilog_inference(self, verilog_dir, top_module_filename, logfile: bool = False, add_registers: bool = False):
        self.verilog_dir = realpath(verilog_dir)
        self.top_module_filename = top_module_filename
        self.dut = PyVerilator.build(f"{self.verilog_dir}/{self.top_module_filename}", verilog_path=[self.verilog_dir], build_dir=f"{self.verilog_dir}/verilator")
        self.is_verilog_inference = True
        self.logfile = logfile
        if add_registers:
            self.latency = len(self.num_neurons)


    def pytorch_inference(self):
        self.is_verilog_inference = False


    def verilog_forward(self, x):
        # Get integer output from the first layer
        input_quant = self.module_list[0].input_quant
        output_quant = self.module_list[-1].output_quant
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.module_list[0].in_features*input_bitwidth
        total_output_bits = self.module_list[-1].out_features*output_bitwidth
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0
        for i in range(x.shape[0]):
            x_i = x[i,:]
            y_i = self.pytorch_forward(x[i:i+1,:])[0]
            xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str(z), y_i))
            xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
            ysc_i = reduce(lambda a,b: a+b, ys_i[::-1])
            self.dut["M0"] = int(xvc_i, 2)
            for j in range(self.latency + 1):
                #print(self.dut.io.M5)
                res = self.dut[f"M{num_layers}"]
                result = f"{res:0{int(total_output_bits)}b}"
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            expected = f"{int(ysc_i,2):0{int(total_output_bits)}b}"
            result = f"{res:0{int(total_output_bits)}b}"
            assert(expected == result)
            res_split = [result[i:i+output_bitwidth] for i in range(0, len(result), output_bitwidth)][::-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split)))
            y[i,:] = yv_i
            # Dump the I/O pairs
            if self.logfile is not None:
                with open(self.logfile, "a") as f:
                    f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}{int(ysc_i,2):0{int(total_output_bits)}b}\n")
        return y


    def pytorch_forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x


    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        else:
            return self.pytorch_forward(x)


class JetSubstructureLutModel(JetSubstructureNeqModel):
    pass


class JetSubstructureVerilogModel(JetSubstructureNeqModel):
    pass

# CNN models 
class QuantizedJSC_CNN_NEQ(nn.Module):
    def __init__(self, model_config): 
        super(QuantizedJSC_CNN_NEQ, self).__init__()
        self.model_config = model_config
        self.num_neurons = [self.model_config['input_length']] + self.model_config['hidden_layers'] + [self.model_config['output_length']]
        seq_length = model_config['sequence_length']
        layer_list = []

        # QNN model structure 
        for i in range(1, len(self.num_neurons)): 
            in_features = self.num_neurons[i-1]
            out_features = self.num_neurons[i]
            
            # applying batch norm for the out_features in each layer
            bn = nn.BatchNorm1d(out_features) 

            if  i == 1:   # first layer architecture 
                bn_in = nn.BatchNorm1d(in_features)
                # input_bias = ScalarBiasScale(scale=False, bias_init=-0.33)
                input_quantized = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config['input_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=[bn_in], post_transforms=None)
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in =model_config['input_fanin']) 
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, seq_length=seq_length, input_quant=input_quantized, output_quant=output_quantized, mask=mask, padding=model_config['padding'], apply_input_quant=True, cnn_output=True)
                layer_list.append(layer)
                
            elif  i == len(self.num_neurons)-1:   # last layer architecture 
                output_bias_scale = ScalarBiasScale(bias_init=0.33) 
                output_quantized = QuantBrevitasActivation(QuantIdentity(bit_width=model_config['output_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=None, post_transforms=[output_bias_scale]) #[output_bias_scale]
                mask = RandomFixedSparsityMask2D(in_features=in_features, out_features=out_features, fan_in = model_config['output_fanin'])
                layer = SparseLinearNeq(in_features=in_features, out_features=out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, apply_output_quant=True, apply_input_quant=False, first_linear=False, bias=True)
                layer_list.append(layer)
            
            elif i == len(self.num_neurons)-2:   #hidden linear layers architecture (normal)
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityMask2D(in_features=in_features, out_features=out_features, fan_in=model_config['hidden_fanin'])
                layer = SparseLinearNeq(in_features=in_features, out_features=out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, apply_input_quant=False, first_linear=False, bias=True)
                layer_list.append(layer)
            
            elif i == len(self.num_neurons)-3:   # first hidden linear layers architecture
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityMask2D(in_features=model_config['1st_layer_in_f'], out_features=out_features, fan_in=model_config['hidden_fanin'])
                layer = SparseLinearNeq(in_features=model_config['1st_layer_in_f'], out_features=out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, reshaped_in_features = model_config['1st_layer_in_f'], apply_input_quant=False, first_linear=True, bias=True)
                layer_list.append(layer)

            elif i == len(self.num_neurons)-4:   # last hidden conv layers architecture 
                # seq_length = (seq_length + (2 * model_config['padding'])- 3) // 2 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in=model_config['conv_fanin'] )
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, seq_length=seq_length, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, padding=model_config['padding'], apply_input_quant=False, cnn_output=False)
                layer_list.append(layer)

            else:   # hidden conv layers architecture 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in=model_config['conv_fanin'] )
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, seq_length=seq_length, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, padding=model_config['padding'], apply_input_quant=False, cnn_output=True)
                layer_list.append(layer)


        self.module_list = nn.ModuleList(layer_list)
        self.is_verilog_inference = False 
        self.latency = 1 
        self.verilog_dir = None 
        self.top_module_filename = None
        self.dut = None
        self.log_file = None 
    

    def verilog_inference(self, verilog_dir, top_module_filename, log_file: bool = False, add_registers: bool = False): 
        self.verilog_dir = realpath(verilog_dir) 
        self.top_module_filename = top_module_filename
        self.dut = PyVerilator.build(f"{self.verilog_dir}/{self.top_module_filename}", verilog_path=[self.verilog_dir], build_dir=f"{self.verilog_dir}/verilator")
        self.is_verilog_inference = True
        self.log_file = log_file
        if add_registers: 
            self.latency = len(self.num_neurons)

    
    def pytorch_inference(self): 
        self.is_verilog_inference = False
    

    def verilog_forward(self, x): 
        # get integer output from the first layer 
        input_quant = self.module_list[0].input_quant
        output_quant = self.module_list[0].output_quant
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.module_list[0].in_features*input_bitwidth
        total_output_bits = self.module_list[-1].out_features*output_bitwidth
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0

        for i in range(x.shape[0]): 
            x_i = x[i, :]
            y_i = self.pytorch_forward(x[i:i+1, :])[0] 
            xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str(z), y_i))
            xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
            ysc_i = reduce(lambda a,b: a+b, ys_i[::-1])
            self.dut['M0'] = int(xvc_i, 2) 

            for j in range(self.latency + 1): 
                res = self.dut[f'M{num_layers}']
                result = f'{res:0{int(total_output_bits)}b}'
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            
            expected = f'{int(ysc_i, 2):0{int(total_output_bits)}b}'
            result = f'{res:0{int(total_output_bits)}b}'
            assert(expected == result)
            res_split = [result[i:i+output_bitwidth] for i in range(0, len(result), output_bitwidth)][:,:,-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split)))            
            y[i, :] = yv_i

            # dump the I/O pairs
            if self.log_file is not None: 
                with open(self.log_file, 'a') as f:
                    f.write(f'{int(xvc_i, 2):0{int(total_input_bits)}b}{int(ysc_i, 2):0{int(total_output_bits)}b}\n')

            return y 
    

    def pytorch_forward(self, x): 
        for i, layer in enumerate(self.module_list): 
            x = layer(x)    
        return x 
    

    def forward(self, x): 
        if self.is_verilog_inference: 
            x = self.verilog_forward(x)
        else: 
            x = self.pytorch_forward(x)

            
        return x 
    

# LUT-model
class Quantized_JSC_LUT(QuantizedJSC_CNN_NEQ): 
    pass 


# Verilog-model
class Quantized_JSC_Verilog(QuantizedJSC_CNN_NEQ): 
    pass 