# CNN-based LogicNets for Electrocardiography classification (ECG)

## Download the dataset 

```bash
# Make a data directory
mkdir -p data

# Navigate into it
cd data

# Download ECG dataset directly into ./data
wget -r -nH --cut-dirs=3 -N -c -np https://physionet.org/files/mitdb/1.0.0/
```
## Training and Verilog generation 

Two model variations are provided (cnn-a) and (cnn-l), Their model topology (number of layers and channels/neurons) are different. To train the networks, run: 

```python3 train.py --arch <cnn-a|cnn-l> --log-dir ./<cnn-a | cnn-l>/ ```


After training is complete, Generate truth tables and Verilog script for the trained model using: 

``` python3 neq2lut.py --log-dir ./<cnn-a|cnn-l>/ --checkpoint ./<cnn-a|cnn-l>/best_acc.pth --arch <cnn-a|cnn-l> --dump-io --add-registers```


## Results
The achieved results may vary slightly, depending on the system configuration. The results shown below are acheived when training on CPU and using Vivado 2022.2: 

| Model      | Test Accuracy | LUTs     | FFs      | f-max (MHz)  |Latency (ns) |
|:-----------|:------------- |:---------|:---------|:-------------|:------------|
| cnn-a      | 89%           | 4138     |1034      |593.8         |5.0          |
| cnn-l      | 91%           | 83874    |7575      |310.0         |16.1         |


