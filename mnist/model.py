# NOTE: (Important!!!) when using different model architectures, the "if" statements that define the layers must be changed accordingly.
# For example, if more pooling layers are included then the "elif" statement that defines the pooling layer should be uncommented and adjusted. 

'''
This is the model.py file for the MNIST dataset processing.
The file include both architecures for MLP and CNN topologies.
The MLP model is adapted from PolyLUT repository ("https://github.com/MartaAndronic/PolyLUT.git") and the CNN model is developed as part of this project
'''

# Imports
import os 
import sys 
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


# Get the absolute path of the directory where model.py is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Append the absolute path to the src directory
sys.path.append(os.path.join(base_path, '../src/'))


# Importing functions from the directory 
from nn_layers import SparseConv1dNeq, SparseLinearNeq, pooling_layer, RandomFixedSparsityMask2D, RandomFixedSparsityConv1DMask # type: ignore
from quant import QuantBrevitasActivation, ScalarBiasScale    #type: ignore

# The models below is for the Linear MLP MNIST processing 
class MINSTmodelneq(nn.Module):
    def __init__(self, model_config):
        super(MINSTmodelneq, self).__init__()
        self.model_config = model_config
        self.num_neurons = [model_config["input_length"]] + model_config["hidden_layers"] + [model_config["output_length"]]
        print(self.num_neurons)
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i-1]
            print('in_features: ', in_features)
            out_features = self.num_neurons[i]
            print('out_features: ', out_features)
            bn = nn.BatchNorm1d(out_features)
            if i == 1:
                input_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["input_bitwidth"], max_val=2., min_val=-2, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.CONST))
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=2, min_val=-2, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["input_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=input_quant, output_quant=output_quant, mask=mask)
                layer_list.append(layer)
            elif i == len(self.num_neurons)-1:
                output_bias_scale = ScalarBiasScale(bias_init=0.33)
                output_quant = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["output_bitwidth"], max_val=2, min_val=-2, narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=[output_bias_scale])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["output_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, mask=mask, apply_input_quant=False)
                layer_list.append(layer)
            else:
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=2, min_val=-2, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["hidden_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, mask=mask, apply_input_quant=False)
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
            x = self.verilog_forward(x)
            output_scale, output_bits = self.module_list[-1].output_quant.get_scale_factor_bits()
            x = self.module_list[-1].output_quant.apply_post_transforms((x - 2**(output_bits-1)) * output_scale)
        else:
            x = self.pytorch_forward(x)
        # Scale output, if necessary
            if self.module_list[-1].is_lut_inference:
                output_scale, output_bits = self.module_list[-1].output_quant.get_scale_factor_bits()
                x = self.module_list[-1].output_quant.apply_post_transforms(x * output_scale)
        return x


class MINSTmodellut(MINSTmodelneq):
    pass


class MINSTmodelver(MINSTmodelneq):
    pass


# The models below is for the 1D-CNN MNIST processing 
class QuantizedMNIST_NEQ(nn.Module):
    def __init__(self, model_config): 
        super(QuantizedMNIST_NEQ, self).__init__()
        self.model_config = model_config
        self.maxpool = nn.MaxPool1d(2)
        self.num_neurons = [self.model_config['input_length']] + self.model_config['hidden_layers'] + [self.model_config['output_length']]
        print(self.num_neurons)
        seq_length = model_config['sequence_length']
        layer_list = []
        conv_layer_list = []


        # QNN model structure 
        for i in range(1, len(self.num_neurons)): 
            in_features = self.num_neurons[i-1]
            out_features = self.num_neurons[i]
            print(f'layer{i}, in_f {in_features}, out_f {out_features}')
            
            # applying batch norm for the out_features in each layer
            bn = nn.BatchNorm1d(out_features) 
            maxpool = nn.MaxPool1d(2)

            if  i == 1:   # first layer architecture 
                input_quantized = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config['input_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=None, post_transforms=None)
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=model_config['kernel_size'], fan_in =model_config['input_fanin']) 
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=model_config['kernel_size'], seq_length=seq_length, input_quant=input_quantized, output_quant=output_quantized, mask=mask, padding=model_config['padding'], cnn_output=False)
                layer_list.append(layer)
                conv_layer_list.append(layer)
                
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

            # elif  i in {2, 4}:   # pooling layers architecture 
            #     output_quantized = QuantBrevitasActivation(QuantIdentity(bit_width=model_config['hidden_bitwidth'], max_val=5.0, min_val=-5.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST))
            #     mask1 = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in =model_config['conv_fanin']) # this mask has been used in the first layer as it returns a mask with all elements set to 1
            #     layer = pooling_layer(in_channels=in_features, out_channels=out_features, seq_length=seq_length, input_quant=layer_list[-1].output_quant, output_quant=layer_list[-1].output_quant, mask=mask1, pooling_kernel_size=2, apply_input_quant=False, apply_output_quant=False, cnn_input=True)
            #     layer_list.append(layer)
            #     seq_length = seq_length // 2
            #     self.seq_length = seq_length
            
            elif  i == len(self.num_neurons)-4 :   # last pooling layers architecture 
                output_quantized = QuantBrevitasActivation(QuantIdentity(bit_width=model_config['hidden_bitwidth'], max_val=5.0, min_val=-5.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST))
                layer = pooling_layer(in_channels=in_features, out_channels=out_features, seq_length=seq_length, input_quant=layer_list[-1].output_quant, output_quant=layer_list[-1].output_quant, pooling_kernel_size=2, apply_input_quant=False, apply_output_quant=False, cnn_input=False)
                layer_list.append(layer)
                seq_length = seq_length // 2
                self.seq_length = seq_length

            else:   # hidden conv layers architecture 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=None)
                mask = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=model_config['kernel_size'], fan_in=model_config['conv_fanin'] )
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=model_config['kernel_size'], seq_length=seq_length, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask, padding=model_config['padding'], apply_input_quant=False, cnn_output=False)
                layer_list.append(layer)
                conv_layer_list.append(layer)



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

            output_scale, output_bits = self.module_list[-1].output_quant.get_scale_factor_bits()
            x = self.module_list[-1].output_quant.apply_post_transforms((x - 2**(output_bits-1)) * output_scale)

        else: 
            x = self.pytorch_forward(x)
            # scale output if necessary 
            # if self.module_list[-1].is_lut_inference: 
            #     output_scale, output_bits = self.module_list[-1].output_quant.get_scale_factor_bits()
            #     x = self.module_list[-1].output_quant.apply_post_transforms(x * output_scale) 
        return x 
    

# LUT-model
class QuantizedMNIST_LUT(QuantizedMNIST_NEQ): 
    pass 


# Verilog-model
class QuantizedMNIST_Verilog(QuantizedMNIST_NEQ): 
    pass 

