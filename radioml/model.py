'''
This is the model architecture for RadioML. 
The first linear layer still need to be modified by deceiding the number of the input featrues dynamically without the need 
to manually reconfigure it everytime based on the sequence length and the number of channels of the used dataset
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

# print(sys.path) 

# sys.path.append('../src/')
# print(sys.path)

# Importing functions from the directory 

from nn_layers import SparseConv1dNeq, SparseLinearNeq, RandomFixedSparsityMask2D, RandomFixedSparsityConv1DMask # type: ignore
from quant import QuantBrevitasActivation, ScalarBiasScale, ScalarScaleBias # type: ignore




class QuantizedRadiomlNEQ(nn.Module):
    def __init__(self, model_config): 
        super(QuantizedRadiomlNEQ, self).__init__()
        self.model_config = model_config
        self.maxpool = nn.MaxPool1d(2)
        self.num_neurons = [self.model_config['input_length']] + self.model_config['hidden_layers'] + [self.model_config['output_length']]
        seq_length = model_config['sequence_length']
        layer_list = []


        # QNN model structure 
        for i in range(1, len(self.num_neurons)): 
            in_features = self.num_neurons[i-1]
            out_features = self.num_neurons[i]
            
            # applying batch norm for the out_features in each layer
            bn = nn.BatchNorm1d(out_features) 
            maxpool = nn.MaxPool1d(2)

            if  i == 1:   # first layer architecture 
                input_quantized = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config['input_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=None, post_transforms=None)
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in =model_config['input_fanin']) 
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, seq_length=seq_length, input_quant=input_quantized, output_quant=output_quantized, mask=mask, padding=model_config['padding'])
                layer_list.append(layer)
                
            elif  i == len(self.num_neurons)-1:   # last layer architecture 
                output_bias_scale = ScalarBiasScale(bias_init=0.33) 
                output_quantized = QuantBrevitasActivation(QuantIdentity(bit_width=model_config['output_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=None, post_transforms=None) #[output_bias_scale]
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

            elif i == len(self.num_neurons)-4:   # hidden conv layers architecture 
                # seq_length = (seq_length + (2 * model_config['padding'])- 3) // 2 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in=model_config['conv_fanin'] )
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, seq_length=seq_length, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, padding=model_config['padding'], apply_input_quant=False, cnn_output=False)
                layer_list.append(layer)

            else:   # hidden conv layers architecture 
                # seq_length = (seq_length + (2 * model_config['padding'])- 3) // 2 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in=model_config['conv_fanin'] )
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, seq_length=seq_length, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, padding=model_config['padding'], apply_input_quant=False)
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
            
            # if i < 4:
            #     # print('x shape before the maxpool is: ', x.shape)
            #     x = self.maxpool(x)
                # print('x shape after the maxpool is: ', x.shape)
    
        return x 
    

    def forward(self, x): 
        if self.is_verilog_inference: 
            x = self.verilog_forward(x)

            output_scale, output_bits = self.module_list[-1].output_quant.get_scale_factor_bits()
            x = self.module_list[-1].output_quant.apply_post_transforms((x - 2**(output_bits-1)) * output_scale)

        else: 
            x = self.pytorch_forward(x)
            # scale output if necessary 
            # if self.module_list[-1].is_lut_inference: 
            #     output_scale, output_bits = self.module_list[-1].output_quant.get_scale_factor_bits()
            #     print(f' the output scale is: {output_scale}')
            #     print(f' the output before applying the scaling is: {x}')
            #     x = self.module_list[-1].output_quant.apply_post_transforms(x * output_scale) 
            #     print(f'the output scaling has been applied to x {x}')
            
        return x 
    


# LUT-model
class QuantizedRadiomlLUT(QuantizedRadiomlNEQ): 
    pass 

# Verilog-model
class QuantizedRadiomlVerilog(QuantizedRadiomlNEQ): 
    pass 
