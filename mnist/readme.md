# LogicNets for MNIST Classification

In this example, MNIST dataset is trained using the Extended LogicNets methodolody.

## Prerequisites

* LogicNets
* h5py
* numpy
* scikit-learn
* PyTorch


## Installation

The extenstion of the original framework is not included as a docker image. Hence, only manual installation is available.

## Download the Dataset

MNIST is available in torchvision. Thus, the data downloader is included in the training functions in train.py 
MNIST fashion is also available and included in this example and can be trained through the same files by setting the "mnist_fashion" argument to True
## Usage

# To train the MLP configurations 
Train: 

```bash
python train.py --arch <mlp8|mlp16|mlp28> --log-dir ./<mlp8|mlp16|mlp28>/
```

Then, generate verilog: 

```bash
python neq2lut.py --arch <mlp8|mlp16|mlp28> --checkpoint ./<best_acc>/best_acc.pth --log-dir ./<mlp8|mlp16|mlp28>/verilog/ 
```

# To train the MLP configurations 
Train:

```bash
python train.py --arch <cnn8|cnn16> --log-dir ./<cnn8|cnn16>/
```

To then generate verilog:

```bash
python neq2lut.py --arch <cnn8|cnn16> --checkpoint ./<cnn8|cnn16>/best_acc.pth --log-dir ./<cnn8|cnn16>/verilog/ 
```

## Results

Results may vary slightly. 

| Network Architecture  | Accuracy@30db (%) | LUTs  | Flip Flops    | Fmax (Mhz)    |    Latency (ns)   |
| --------------------- | ----------------- | ----- | ------------- | ------------- | ----------------- |
| MLP8                  |              90.3 | 35109 |          1812 |         420.0 |              14.2 |
| MLP16                 |              93.7 | 34381 |          1852 |         414.0 |              15.0 |
| MLP28                 |              93.0 | 35569 |          2668 |         354.0 |              16.9 |
| ----------------------------------------------------------------------------------------------------- |
| CNN8                  |              87.0 | 51492 |          2477 |         345.0 |              17.4 |
| CNN16                 |              89.0 |123920 |          6606 |         355.3 |              17.1 |



