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
(--mnist_fashion True) 
to both training and verilog generation steps


## Results

Results may vary slightly. 

| Model       |Test Accuracy  |LUTs      |FFs       | f-max (MHz)   |Latency (ns)  |
| :-----------|:------------- |:---------|:---------|:------------- |:------------ |
| MLP8        | 90.3          | 35109    |1812      | 420.0         |14.2          |
| MLP16       | 93.7          | 34381    |1852      | 414.0         |15.0          |
| MLP28       | 93.0          | 35569    |2668      | 354.0         |16.9          |  
|-------------|---------------|----------|----------|---------------|--------------|
| CNN8        | 87.0          | 51492    |2477      | 345.0         |17.4          |
| CNN16       | 89.0          | 123920   |6606      | 355.3         |17.1          |


<table style="border: 1px solid black; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border: 1px solid black; padding: 5px;">Model</th>
      <th style="border: 1px solid black; padding: 5px;">Test Accuracy</th>
      <th style="border: 1px solid black; padding: 5px;">LUTs</th>
      <th style="border: 1px solid black; padding: 5px;">FFs</th>
      <th style="border: 1px solid black; padding: 5px;">f-max (MHz)</th>
      <th style="border: 1px solid black; padding: 5px;">Latency (ns)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black; padding: 5px;">MLP8</td>
      <td style="border: 1px solid black; padding: 5px;">90.3</td>
      <td style="border: 1px solid black; padding: 5px;">35109</td>
      <td style="border: 1px solid black; padding: 5px;">1812</td>
      <td style="border: 1px solid black; padding: 5px;">420.0</td>
      <td style="border: 1px solid black; padding: 5px;">14.2</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;">MLP16</td>
      <td style="border: 1px solid black; padding: 5px;">93.7</td>
      <td style="border: 1px solid black; padding: 5px;">34381</td>
      <td style="border: 1px solid black; padding: 5px;">1852</td>
      <td style="border: 1px solid black; padding: 5px;">414.0</td>
      <td style="border: 1px solid black; padding: 5px;">15.0</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;">MLP28</td>
      <td style="border: 1px solid black; padding: 5px;">93.0</td>
      <td style="border: 1px solid black; padding: 5px;">35569</td>
      <td style="border: 1px solid black; padding: 5px;">2668</td>
      <td style="border: 1px solid black; padding: 5px;">354.0</td>
      <td style="border: 1px solid black; padding: 5px;">16.9</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;">CNN8</td>
      <td style="border: 1px solid black; padding: 5px;">87.0</td>
      <td style="border: 1px solid black; padding: 5px;">51492</td>
      <td style="border: 1px solid black; padding: 5px;">2477</td>
      <td style="border: 1px solid black; padding: 5px;">345.0</td>
      <td style="border: 1px solid black; padding: 5px;">17.4</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;">CNN16</td>
      <td style="border: 1px solid black; padding: 5px;">89.0</td>
      <td style="border: 1px solid black; padding: 5px;">123920</td>
      <td style="border: 1px solid black; padding: 5px;">6606</td>
      <td style="border: 1px solid black; padding: 5px;">355.3</td>
      <td style="border: 1px solid black; padding: 5px;">17.1</td>
    </tr>
  </tbody>
</table>

