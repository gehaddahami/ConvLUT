# LogicNets for Jet-Substructure Classification

This example shows the accuracy that is attainable using the LogicNets methodology on the jet substructure classification task using a 1D-CNN model

In additions to the results shown in LogicNets paper [FPL'20 paper](https://arxiv.org/abs/2004.03021), this repository conducted 2 extra tests using 1D-CNN topology for the JSC dataset.  

## Usage

To train the added \"CNN-S\" and \"CNN-L\" networks described in this project, run the following:

```bash
python train.py --arch <cnn-s|cnn-l> --log-dir ./<cnn_s|cnn_l>/
```

To then generate verilog from this trained model, run the following:

```bash
python neq2lut.py --arch <cnn-s|cnn-l> --checkpoint ./<cnn_s|cnn_l>/best_accuracy.pth --log-dir ./<cnn_s|cnn_l>/verilog/ --add-registers
```

## Results

Your results may vary slightly, depending on your system configuration.
The following results are attained when training on a CPU and synthesising with Vivado 2022.2:

| Network Architecture  | Test Accuracy (%) | LUTs  | Flip Flops    | Fmax (Mhz)    |     Latency (ns)  |
| --------------------- | ----------------- | ----- | ------------- | ------------- | ----------------- |
| cnn-S                 |              70.0 |  6819 |           478 |         555.9 |               8.9 |
| cnn-L                 |              70.0 | 14251 |           934 |         535.6 |              13.1 |

