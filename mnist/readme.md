# LogicNets for MNIST Classification (CNN & MLP )


## Download the Dataset

MNIST is available in torchvision. Thus, the data downloader is included in the training functions in train.py 
MNIST fashion is also available and included in this example and can be trained using the same files 


## Usage

### To train the MLP configurations 
- Train: 

```bash
python train.py --arch <mlp8|mlp16|mlp28> --log-dir ./<mlp8|mlp16|mlp28>/ --topology "linear"
```

- Then, generate verilog: 

```bash
python neq2lut.py --arch <mlp8|mlp16|mlp28> --checkpoint ./<best_acc>/best_acc.pth --log-dir ./<mlp8|mlp16|mlp28>/verilog/ --topology "linear"
```

### To train the CNN configurations 
- Train:

```bash
python train.py --arch <cnn8|cnn16> --log-dir ./<cnn8|cnn16>/ --topology "cnn"
```

- To then generate verilog:

```bash
python neq2lut.py --arch <cnn8|cnn16> --checkpoint ./<cnn8|cnn16>/best_acc.pth --log-dir ./<cnn8|cnn16>/verilog/ --topology "cnn"
```

➕ To enable Fashion MNIST, add:
--mnist_fashion True 
to both training and verilog generation steps


## Results

Results may vary slightly. 

| Model       |Test Accuracy  |LUTs      |FFs       | f-max (MHz)   |Latency (ns)  |
| :-----------|:------------- |:---------|:---------|:------------- |:------------ |
| MLP8        | 90.3          | 35109    |1812      | 420.0         |14.2          |
| MLP16       | 93.7          | 34381    |1852      | 414.0         |15.0          |
| MLP28       | 93.0          | 35569    |2668      | 354.0         |16.9          |  ------------------------------------------------------------------------------------
| CNN8        | 87.0          | 51492    |2477      | 345.0         |17.4          |
| CNN16       | 89.0          | 123920   |6606      | 355.3         |17.1          |



