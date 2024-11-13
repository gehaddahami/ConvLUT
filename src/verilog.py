'''
This file contains the Verilog functions that are to be used within layers, within neurons or for the entire model 
'''
 


def generate_register_verilog(module_name="myreg", param_name="DataWidth", input_name="data_in", output_name="data_out"):
    register_template = """\
module {module_name} #(parameter {param_name}=512) (
    input [{param_name}-1:0] {input_name},
    input wire clk,
    input wire rst,
    output reg [{param_name}-1:0] {output_name}
    );
    always@(posedge clk) begin
    if(!rst)
        {output_name}<={input_name};
    else
        {output_name}<=0;
    end
endmodule\n
"""
    return register_template.format(    module_name=module_name,
                                        param_name=param_name,
                                        input_name=input_name,
                                        output_name=output_name)





def generate_logicnets_verilog(module_name: str, input_name: str, input_bits: int, output_name: str, output_bits: int, module_contents: str):
    logicnets_template = """\
module {module_name} (input [{input_bits_1:d}:0] {input_name}, input clk, input rst, output[{output_bits_1:d}:0] {output_name});
{module_contents}
endmodule\n"""
    return logicnets_template.format( module_name=module_name,
                                input_name=input_name,
                                input_bits_1=input_bits-1,
                                output_name=output_name,
                                output_bits_1=output_bits-1,
                                module_contents=module_contents)






def layer_connection_verilog(layer_string: str, input_string: str, input_bits: int, output_string: str, output_bits: int, output_wire=True, register=False):
    if register:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
myreg #(.DataWidth({input_bits})) {layer_string}_reg (.data_in({input_string}), .clk(clk), .rst(rst), .data_out({input_string}w));\n"""
    else:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
assign {input_string}w = {input_string};\n"""
    layer_connection_template += "wire [{output_bits_1:d}:0] {output_string};\n" if output_wire else ""
    layer_connection_template += "{layer_string} {layer_string}_inst (.M0({input_string}w), .M1({output_string}));\n"
    return layer_connection_template.format(    layer_string=layer_string,
                                                input_string=input_string,
                                                input_bits=input_bits,
                                                input_bits_1=input_bits-1,
                                                output_string=output_string,
                                                output_bits_1=output_bits-1)






def generate_lut_verilog(module_name, input_fanin_bits, output_bits, lut_string):
    lut_neuron_template = """\
module {module_name} ( input [{input_fanin_bits_1:d}:0] M0, output [{output_bits_1:d}:0] M1 );

	(*rom_style = "distributed" *) reg [{output_bits_1:d}:0] M1r;
	assign M1 = M1r;
	always @ (M0) begin
		case (M0)
{lut_string}
		endcase
	end
endmodule\n"""
    return lut_neuron_template.format(  module_name=module_name,
                                        input_fanin_bits_1=input_fanin_bits-1,
                                        output_bits_1=output_bits-1,
                                        lut_string=lut_string)





def generate_neuron_connection_verilog(input_indices, input_bitwidth):
    connection_string = ""
    for i in range(len(input_indices)):
        index = input_indices[i]
        offset = index*input_bitwidth
        for b in reversed(range(input_bitwidth)):
            connection_string += f"M0[{offset+b}]"
            if not (i == len(input_indices)-1 and b == 0):
                connection_string += ", "
    return connection_string


def generate_neuron_connection_verilog_conv(active_channels, state_space_indices, input_bitwidth, seq_position, kernel_size, total_channels, seq_length):
    connection_string = ""
    for channel in active_channels:
        # Calculate the starting bit offset for the current active channel and sequence position
        start_offset = channel * seq_length * input_bitwidth + seq_position * input_bitwidth
        for idx in range(kernel_size):
            # print('idx ',idx)
            if idx >= len(state_space_indices):  # Stop if kernel_size exceeds state_space_indices length
                break
            index = state_space_indices[idx]
            offset = start_offset + index * input_bitwidth
            # print('offset', offset)
            # Collect the bits for this index position
            for b in reversed(range(input_bitwidth)):
                connection_string += f"M0[{offset + b}]"
                if not (channel == active_channels[-1] and idx == kernel_size - 1 and b == 0):
                    connection_string += ", "
    return connection_string