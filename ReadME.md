# CNN LUT-based Acceleration (ConvLUT)

This repository presents a novel method in accelerating 1D-Convolutional networks following [LogicNets](https://github.com/Xilinx/logicnets) approach.

# Description 

This repository provides a complete guide for training, testing, and converting trained 1d-CNN channels and neurons to truth tables and Verilog representation.

# Examples 

So far, this repository provides four examples: 

- [RadioML modulations classification](./radioml) 
- [MNIST digit Recognition](./mnist) 
- [Jet Substructure Classification](./jet_substructure)
- [ElectroCardioGraphy classification](./ecg) 

Each example contains multiple configurations that allows for exploring both shallow and deep 1D-CNNs. 



# Summary of major modification from LogicNets: 

- 1D-CNN customized layers (Convolutional and Pooling layers) are developed for the purpose of: 
    - Training and validating 1D-CNN layers and generating the channels NEQs 
    - Generating truth tables from the trained CNN channels 
    - Generating Verilog script for CNN topologies 

