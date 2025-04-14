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


# This file contains: 
# 1) Functions to generate the state space for floating or integer bit representation 
# 2) The main quantization class (developed by LogicNets (https://arxiv.org/abs/2004.03021)) 
#    with additional functionality inreoduced for 1D-CNNs, namely: 
#        a) activate_transforms():  to activate additional functions included in the quantization class 
#        b) deactivate_transforms(): to deactivate additional functions included in the quantization class 
#    These functions are used during pooling kernels truth tables generation 
# 3) Biasing and scaling functions that can be used as post or transformation functions within the quantization class  


# Imports 
import torch
from torch import nn 
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from brevitas.core.quant import QuantType
from brevitas.core.quant import RescalingIntQuant, ClampedBinaryQuant 


# brevitas activation class and functions 
def get_int_state_space(bits: int, signed: bool, narrow_range: bool):
    start = int(0 if not signed else (-2**(bits-1) + int(narrow_range))) # calculate the minimum value in the range
    end = int(start + 2**(bits) - int(narrow_range)) # calculate the maximum of the range
    state_space = torch.as_tensor(range(start, end))
    return state_space


# TODO: Put this inside an abstract base class
def get_float_state_space(bits: int, scale_factor: float, signed: bool, narrow_range: bool, quant_type: QuantType):
    if quant_type == QuantType.INT:
        bin_state_space = get_int_state_space(bits, signed, narrow_range)
    elif quant_type == QuantType.BINARY:
        bin_state_space = torch.as_tensor([-1., 1.])
    state_space = scale_factor * bin_state_space
    return state_space


# Some of the functions and calls included in the class below have been edited to match with the new release of the Brevitas Library.
class QuantBrevitasActivation(nn.Module):
    def __init__(self, brevitas_module, pre_transforms: list = [], post_transforms: list = []):
        super(QuantBrevitasActivation, self).__init__()
        self.brevitas_module = brevitas_module
        self.pre_transforms = nn.ModuleList(pre_transforms)
        self.post_transforms = nn.ModuleList(post_transforms)
        self.is_bin_output = False

    # TODO: Move to a base class
    # TODO: Move the string templates to verilog.py
    def get_bin_str(self, x: int):
        quant_type = self.get_quant_type()
        scale_factor, bits = self.get_scale_factor_bits()
        if quant_type == QuantType.INT:
            tensor_quant = self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            offset = 2**(bits-1) -int(narrow_range) if signed else 0
            return f"{int(x+offset):0{int(bits)}b}"
        elif quant_type == QuantType.BINARY:
            return f"{int(x):0{int(bits)}b}"
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))

    # TODO: Move to a base class
    def bin_output(self):
        self.is_bin_output = True

    # TODO: Move to a base class
    def float_output(self):
        self.is_bin_output = False

    def deactivate_transforms(self): 
        self.pre_transforms = nn.ModuleList()
        self.post_transforms = nn.ModuleList() 
    
    def activate_transforms(self):
        self.pre_transforms = self.pre_transforms
        self.post_transforms = self.post_transforms
    
    def get_quant_type(self): 
        brevitas_module_type = type(self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant)
        if brevitas_module_type == RescalingIntQuant:
            return QuantType.INT
        elif brevitas_module_type == ClampedBinaryQuant:
            return QuantType.BINARY
        else:
            raise Exception("Unknown quantization type for tensor_quant: {}".format(brevitas_module_type))
    
    def get_scale_factor_bits(self):
        if hasattr(self.brevitas_module, 'quant_act_scale') and hasattr(self.brevitas_module, 'quant_act_bit_width'):
            scale_factor = self.brevitas_module.quant_act_scale()
            bits = self.brevitas_module.quant_act_bit_width()
            return scale_factor, bits
        else:
            raise AttributeError(f"'{type(self.brevitas_module).__name__}' object does not have the required quantization methods")

    # TODO: Merge this function with 'get_bin_state_space' and remove duplicated code.
    def get_state_space(self):
        quant_type = self.get_quant_type()
        scale_factor, bits = self.get_scale_factor_bits()
        if quant_type == QuantType.INT:
            tensor_quant = self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            state_space = get_float_state_space(bits, scale_factor, signed, narrow_range, quant_type)
        elif quant_type == QuantType.BINARY:
            state_space = scale_factor*torch.tensor([-1, 1])
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))
        return self.apply_post_transforms(state_space)

    def get_bin_state_space(self):
        quant_type = self.get_quant_type()
        _, bits = self.get_scale_factor_bits()
        if quant_type == QuantType.INT:
            tensor_quant = self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            state_space = get_int_state_space(bits, signed, narrow_range)
        elif quant_type == QuantType.BINARY:
            state_space = torch.tensor([0, 1])
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))
        return state_space

    def apply_pre_transforms(self, x):
        if self.pre_transforms is not None:
            for i in range(len(self.pre_transforms)):
                x= self.pre_transforms[i](x)
            return x

    def apply_post_transforms(self, x):
        if self.post_transforms is not None:
            for i in range(len(self.post_transforms)):
                x = self.post_transforms[i](x)
            return x
        
    def forward(self, x):
        if self.is_bin_output:
            s, _ = self.get_scale_factor_bits()
            x = self.apply_pre_transforms(x)
            x = self.brevitas_module(x)
            x = torch.round(x/s).type(torch.int64)
        else:
            x = self.apply_pre_transforms(x)
            x = self.brevitas_module(x)
            x= self.apply_post_transforms(x)
        return x   


# The scaling and bias classes. one applies scaling before the bias and the other does the opposite
class ScalarScaleBias(nn.Module):   # Scalar then bias scaling 
    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0) -> None:
        super(ScalarScaleBias, self).__init__()
        if scale:
            self.weight = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.weight_init = scale_init
        self.bias_init = bias_init
        self.reset_parameters()

    # Change the default initialisation for BatchNorm
    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        if self.weight is not None:
            x = x*self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class ScalarBiasScale(ScalarScaleBias): # Bias then scalar scaling
    def forward(self, x):
        if self.bias is not None:
            x = x + self.bias
        if self.weight is not None:
            x = x*self.weight
        return x