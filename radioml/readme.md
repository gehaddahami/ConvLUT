# LogicNets for RadioML Classification

In this example, RadioML dataset is trained using the Extended LogicNets methodolody.

## Prerequisites

* LogicNets
* h5py
* numpy
* scikit-learn
* PyTorch


## Installation

The extenstion of the original framework is not included as a docker image. Hence, only manual installation is available.

## Download the Dataset

Dataset can be downloaded from the "DeepSig" website. user-email is required to get the download link. 
After download, make sure to replace the dataset_path in both train.py and neq2lut.py files. 

## Usage

To train the \"2-mods-s\", \"2-mods-l\" and \"psk-5\" examples shown in the training files, run the following: 

```bash
python train.py --arch <2-mods-s|2-mods-l|psk-5> --log-dir ./<2_mods_s|2_mods_l|psk_5>/
```

To then generate verilog from this trained model, run the following:

```bash
python neq2lut.py --arch <2-mods-s|2-mods-l|psk-5> --checkpoint ./<2_mods_s|2_mods_l|psk_5>/config_1.pth --log-dir ./<2_mods_s|2_mods_l|psk_5>/verilog/ --add-registers
```

## Results

Results may vary slightly. 

| Network Architecture  | Accuracy@30db (%) | LUTs  | Flip Flops    | Fmax (Mhz)    |    Latency (ns)   |
| --------------------- | ----------------- | ----- | ------------- | ------------- | ----------------- |
| 2-mods-s              |              92.3 | 22820 |          1337 |         504.0 |              9.92 |
| 2-mods-l              |              91.0 | 41402 |          2514 |         467.1 |             10.71 |
| psk-5                 |              81.0 |463175 |         21414 |         200.6 |             49.90 |



