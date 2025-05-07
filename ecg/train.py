import argparse
import os 
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser


# Importing the model and the data loader
from dataset import MITBIHDataset
from model import ECG_NEQ
from train_test_loops import train, test



configs = {
    "cnn-a": {
        "input_length": 1,
        "sequence_length": 300, 
        "kernel_size": 3,
        "hidden_layers": [2] * 2 + [64],
        "output_length": 5,
        "padding": 1, 
        "1st_layer_in_f": 300, #change whenever the sequence length changed 
        "input_bitwidth": 3,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 1,
        "conv_fanin": 2,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "weight_decay": 0,
        "t0": 5,
        "t_mult": 1,
        "checkpoint": None,
        "batch_size": 1024,
        "epochs": 100,
        }, 

    "cnn-l": {
        "input_length": 1,
        "sequence_length": 300, 
        "kernel_size": 3,
        "hidden_layers": [4] * 4 + [64],
        "output_length": 5,
        "padding": 1, 
        "1st_layer_in_f": 300,
        "input_bitwidth": 4,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 1,
        "conv_fanin": 2,
        "hidden_fanin": 6,
        "output_fanin": 6, 
        "weight_decay": 0,
        "t0": 5,
        "t_mult": 1,
        "checkpoint": None,
        "batch_size": 1024,
        "epochs": 100,
    }}

model_config = {
    "hidden_layers": None,
    "input_bitwidth": None,
    "hidden_bitwidth": None,
    "output_bitwidth": None,
    "input_fanin": None,
    "conv_fanin": None,
    "hidden_fanin": None,
    "output_fanin": None,
    "sequence_length": None,
    "kernel_size": None,
    "padding": None,
    "1st_layer_in_f": None, 
    "input_length": None,
    "output_length": None
}

training_config = {
    "sequence_length": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
    "t0": None,
    "t_mult": None, 
    "weight_decay": None, 
    "input_length": None
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "dataset_path": None,
}


def train_ecg(model, train_cfg, other_options, dataset):
    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], sampler=dataset.train_sampler)
    val_loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], sampler=dataset.val_sampler)
    test_loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], sampler=dataset.test_sampler)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    if other_options.get("cuda", False):
        model.cuda()

    maxAcc = 0.0
    num_epochs = train_cfg["epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            if other_options.get("cuda", False):
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_accuracy = test_ecg(model, val_loader, other_options, train_cfg)
        test_accuracy = test_ecg(model, test_loader, other_options, train_cfg)

        print(f"Epoch {epoch}: Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # Save model if test accuracy improves
        if maxAcc < test_accuracy:
            model_save = {
                "model_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "epoch": epoch
            }
            torch.save(model_save, f"{other_options["log_dir"]}/best_acc.pth")
            print(f"Model saved at {other_options["log_dir"]}/best_acc.pth")
            maxAcc = test_accuracy

    # return total_loss / len(train_loader), correct / total

# Function to evaluate the model
def test_ecg(model, dataset_loader, other_options, train_cfg):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataset_loader:
            if other_options["cuda"] == True: 
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct / total


if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets ECG Classification")
    parser.add_argument("--arch", type=str, choices=configs.keys(), default="cnn-a",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument("--dataset_path", type=str, default="./data",
        help="Path to the dataset (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
        help="Batch size for training (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=None, metavar="N",
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.001, metavar="LR",
        help="Initial learning rate (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0, 
        help="Weight decay for optimizer (default: 0)")
    parser.add_argument("--cuda", action="store_true", default=False,
        help="Train on a GPU (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=2025,
        help="Seed to use for RNG (default: %(default)s)")
    parser.add_argument("--input_length", type=int, default=None,
            help="Length of input to use (default: %(default)s)")
    parser.add_argument("--output_length", type=int, default=None,
        help="Length of output to use (default: %(default)s)")
    parser.add_argument("--input-bitwidth", type=int, default=None,
        help="Bitwidth to use at the input (default: %(default)s)")
    parser.add_argument("--hidden-bitwidth", type=int, default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)")
    parser.add_argument("--output-bitwidth", type=int, default=None,
        help="Bitwidth to use at the output (default: %(default)s)")
    parser.add_argument("--input-fanin", type=int, default=None,
        help="Fanin to use at the input (default: %(default)s)")
    parser.add_argument("--conv-fanin", type=int, default=None,
        help="Fanin to use for the convolutional layers (default: %(default)s)")
    parser.add_argument("--hidden-fanin", type=int, default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)")
    parser.add_argument("--output-fanin", type=int, default=None,
        help="Fanin to use at the output (default: %(default)s)")
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)")
    parser.add_argument("--sequence-length", nargs="+", type=int, default=None,
        help="The length of the input sequence (default: %(default)s)")
    parser.add_argument("--kernel-size", type=int, default=None,
        help="The kernel size to use for the convolutional layers (default: %(default)s)")
    parser.add_argument("--1st-layer-in-f", type=int, default=None,
        help="The input feature size of the first layer (default: %(default)s)")
    parser.add_argument("--padding", type=str, default=None,
        help="The padding to use for the input sequence (default: %(default)s)")
    parser.add_argument("--t0", type=int, default=None,
        help="T_0 parameter for CosineAnnealingWarmRestarts scheduler (default: %(default)s)")
    parser.add_argument("--t-mult", type=int, default=None,
        help="T_mult parameter for CosineAnnealingWarmRestarts scheduler (default: %(default)s)")
    parser.add_argument("--log-dir", type=str, default="./log",
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options["arch"]
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]


    if not os.path.exists(options_cfg["log_dir"]):
        os.makedirs(options_cfg["log_dir"])
        print("Directory created")

    # Set random seeds
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    os.environ["PYTHONHASHSEED"] = str(train_cfg["seed"])
    if options["cuda"]:
        torch.cuda.manual_seed_all(train_cfg["seed"])
        torch.backends.cudnn.deterministic = True


    # Load the dataset
    dataset = MITBIHDataset(data_path=options_cfg["dataset_path"], sequence_length=train_cfg["sequence_length"]) 
    print(f"Dataset loaded...")
    # Load the model 
    model = ECG_NEQ(model_cfg)
    if options_cfg["checkpoint"] is not None:
        print(f"Loading pre-trained checkpoint {options_cfg["checkpoint"]}")
        checkpoint = torch.load(options_cfg["checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint["model_dict"])

    # train the model: 
    train_ecg(model, train_cfg, options_cfg, dataset)


