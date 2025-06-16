# This file is Part of Conv-LUT
# Conv-LUT is based on LogicNets   

# Copyright (C) 2021 Xilinx, Inc
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

# This file contains the functions for MNIST training task in both MLP and CNN configurations 
# The file also contain the possibilty to run MNIST fashion dataset, however, it was included in this project as it did not add any result variations 
# Part of the MLP functions are adapted from PolyLUT ("https://github.com/MartaAndronic/PolyLUT.git")

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from argparse import ArgumentParser

from model import MINSTmodelneq, QuantizedMNIST_NEQ

configs = {
    "mlp8": {
        "hidden_layers": [256, 100, 100, 100, 100],
        "input_bitwidth": 2,
        "output_bitwidth": 2,
        "hidden_bitwidth": 2,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "input_length": 64,
        "output_length": 10,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 0.004,
        "seed": 984237, 
        "image_size": 8, 
        "weight_decay": 0, 
        "conv_fanin": None,
        "sequence_length": None, 
        "kernel_size": None,
        "padding": None,
        "1st_layer_in_f": None,
        "t0": 5,
        "t_mult": 1,
        "checkpoint": None,

    }, 
    "mlp16": {
        "hidden_layers": [256, 100, 100, 100, 100],
        "input_bitwidth": 2,
        "output_bitwidth": 2,
        "hidden_bitwidth": 2,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "input_length": 256,
        "output_length": 10,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 0.004,
        "seed": 984237,
        "image_size": 16,
        "weight_decay": 0,
        "conv_fanin": None,
        "sequence_length": None, 
        "kernel_size": None,
        "padding": None,
        "1st_layer_in_f": None,
        "t0": 5,
        "t_mult": 1,
        "checkpoint": None,
    },
    "mlp28": {
        "hidden_layers": [256, 100, 100, 100, 100],
        "input_bitwidth": 2,
        "output_bitwidth": 2,
        "hidden_bitwidth": 2,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "input_length": 784,
        "output_length": 10,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 0.004,
        "seed": 984237,
        "image_size": 28,
        "weight_decay": 0,
        "conv_fanin": None,
        "sequence_length": None, 
        "kernel_size": None,
        "padding": None,
        "1st_layer_in_f": None,
        "t0": 5,
        "t_mult": 1,
        "checkpoint": None
    },
    "cnn8": {
        "input_length": 1,
        "sequence_length": 64, 
        "kernel_size": 3,
        "hidden_layers": [4] * 4 + [100, 100],
        "output_length": 10,
        "padding": 1, 
        "1st_layer_in_f": 128, 
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 1,
        "conv_fanin": 2,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "image_size": 8,
        "weight_decay": 0,
        "t0": 5,
        "t_mult": 1,
        "checkpoint": None,
        "batch_size": 1024,
        "epochs": 2,
    }, 
    "cnn16": {
        "input_length": 1,
        "sequence_length": 256, 
        "kernel_size": 3,
        "hidden_layers": [4] * 4 + [100, 100],
        "output_length": 10,
        "padding": 1, 
        "1st_layer_in_f": 512,
        "input_bitwidth": 4,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 1,
        "conv_fanin": 2,
        "hidden_fanin": 6,
        "output_fanin": 6, 
        "image_size": 16,
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
    'input_length': None,
    'output_length': None
}

training_config = {
    "sequence_length": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
    "t0": None,
    "t_mult": None, 
    "image_size": None,
    "weight_decay": None, 
    "input_length": None
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "topology": None, 
    "mnist_fashion": None
}

# Function to train the model
def train_mlp(model, train_cfg, options):
    # Create data loaders for training and validation
    if options['mnist_fashion']: 
        train_loader = DataLoader(
        datasets.FashionMNIST(
            "fashion_mnist_data",
            download=True,
            train=True,
            transform=transforms.Compose([transforms.Resize((train_cfg['image_size'], train_cfg['image_size'])), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        )
        val_loader = DataLoader(
            datasets.FashionMNIST(
                "fashion_mnist_data",
                download=True,
                train=False,
                transform=transforms.Compose([transforms.Resize((train_cfg['image_size'], train_cfg['image_size'] )), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
            ),
            batch_size=train_cfg["batch_size"],
            shuffle=False,
        )
        test_loader = DataLoader(
            datasets.FashionMNIST(
                "fashion_mnist_data",
                download=True,
                train=False,
                transform=transforms.Compose([transforms.Resize((train_cfg['image_size'], train_cfg['image_size'] )), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
            ),
            batch_size=train_cfg["batch_size"],
            shuffle=False,
        )
    else:  #MNSIT dataset
        train_loader = DataLoader(
            datasets.MNIST(
                "mnist_data",
                download=True,
                train=True,
                transform=transforms.Compose([transforms.Resize((train_cfg['image_size'], train_cfg['image_size'] )), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
            ),
            batch_size=train_cfg["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            datasets.MNIST(
                "mnist_data",
                download=True,
                train=False,
                transform=transforms.Compose([transforms.Resize((train_cfg['image_size'], train_cfg['image_size'] )), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
            ),
            batch_size=train_cfg["batch_size"],
            shuffle=False,
        )
        test_loader = DataLoader(
            datasets.MNIST(
                "mnist_data",
                download=True,
                train=False,
                transform=transforms.Compose([transforms.Resize((train_cfg['image_size'], train_cfg['image_size'] )), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
            ),
            batch_size=train_cfg["batch_size"],
            shuffle=False,
        )

    # Configure optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    if options["cuda"]:
        model.cuda()

    maxAcc = 0.0
    num_epochs = train_cfg["epochs"]
    for epoch in range(0, num_epochs):
        model.train()
        accLoss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            data = data.reshape(-1, train_cfg["input_length"])
            target = torch.nn.functional.one_hot(target, num_classes=10).float()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()

        val_accuracy = test_mlp(model, val_loader, options, train_cfg)
        test_accuracy = test_mlp(model, test_loader, options, train_cfg)

        print(f'Epoch {epoch}: Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    if maxAcc < test_accuracy:
        model_save = {
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'epoch': epoch
        }

        torch.save(model_save, f"{options['log_dir']}/best_acc.pth")
        print(f"Model saved at {options['log_dir']}/best_acc.pth")
        maxAcc = test_accuracy

# Function to evaluate the model
def test_mlp(model, dataset_loader, options, train_cfg):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if options["cuda"]:
                data, target = data.cuda(), target.cuda()
        
        data = data.reshape(-1, train_cfg["input_length"])
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
    
    accuracy = 100.0 * correct / len(dataset_loader.dataset)
    return accuracy



def train_cnn(model, train_cfg, options):
    # Create data loaders for training and validation
    transform = transforms.Compose([transforms.Resize((train_cfg['image_size'], train_cfg['image_size'])), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Full training dataset
    if options["mnist_fashion"]:
        full_train_dataset = datasets.FashionMNIST("fashion_mnist_data", download=True, train=True, transform=transform)
    else:
        full_train_dataset = datasets.MNIST("mnist_data", download=True, train=True, transform=transform)

    # Split training dataset into training and validation sets
    train_size = int(0.8 * len(full_train_dataset))  # 80% training
    val_size = len(full_train_dataset) - train_size  # 20% validation
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Test dataset
    if options["mnist_fashion"]: 
        test_dataset = datasets.FashionMNIST("fashion_mnist_data", download=True, train=False, transform=transform)
    else:
        test_dataset = datasets.MNIST("mnist_data", download=True, train=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    # Configure optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    if options["cuda"]:
        model.cuda()

    maxAcc = 0.0
    num_epochs = train_cfg['epochs']
    for epoch in range(num_epochs):
        model.train()
        acc_loss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), 1, -1)  # Flatten spatial dimensions into 1D sequence
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
    
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate the model on validation and test datasets
        val_accuracy = test_cnn(model, val_loader, options)
        test_accuracy = test_cnn(model, test_loader, options)

        print(f'Epoch {epoch}: Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

        # Save the best model
    if maxAcc < test_accuracy:
        model_save = {
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'epoch': epoch
        }

        torch.save(model_save, f"{options['log_dir']}/best_acc.pth")
        print(f"Model saved at {options['log_dir']}/best_acc.pth")
        maxAcc = test_accuracy

            
def test_cnn(model, test_loader, options): 
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Reshape data for 1D CNN
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            data = data.view(data.size(0), 1, -1)  

            # Forward pass
            output = model(data)
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy



if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets MNIST Classification")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="mlp28",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
        help="Initial learning rate (default: %(default)s)")
    parser.add_argument('--weight_decay', type=float, default=0, 
        help='Weight decay for optimizer (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=False,
        help="Train on a GPU (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=2025,
        help="Seed to use for RNG (default: %(default)s)")
    parser.add_argument('--input_length', type=int, default=None,
            help="Length of input to use (default: %(default)s)")
    parser.add_argument('--output_length', type=int, default=None,
        help="Length of output to use (default: %(default)s)")
    parser.add_argument('--input-bitwidth', type=int, default=None,
        help="Bitwidth to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-bitwidth', type=int, default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)")
    parser.add_argument('--output-bitwidth', type=int, default=None,
        help="Bitwidth to use at the output (default: %(default)s)")
    parser.add_argument('--input-fanin', type=int, default=None,
        help="Fanin to use at the input (default: %(default)s)")
    parser.add_argument('--conv-fanin', type=int, default=None,
        help="Fanin to use for the convolutional layers (default: %(default)s)")
    parser.add_argument('--hidden-fanin', type=int, default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)")
    parser.add_argument('--output-fanin', type=int, default=None,
        help="Fanin to use at the output (default: %(default)s)")
    parser.add_argument('--hidden-layers', nargs='+', type=int, default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)")
    parser.add_argument('--sequence-length', nargs='+', type=int, default=None,
        help="The length of the input sequence (default: %(default)s)")
    parser.add_argument('--kernel-size', type=int, default=None,
        help="The kernel size to use for the convolutional layers (default: %(default)s)")
    parser.add_argument('--image-size', type=int, default=None,
        help="The size of the image (default: %(default)s)")
    parser.add_argument('--1st-layer-in-f', type=int, default=None,
        help="The input feature size of the first layer (default: %(default)s)")
    parser.add_argument('--padding', type=str, default=None,
        help="The padding to use for the input sequence (default: %(default)s)")
    parser.add_argument('--t0', type=int, default=None,
        help="T_0 parameter for CosineAnnealingWarmRestarts scheduler (default: %(default)s)")
    parser.add_argument('--t-mult', type=int, default=None,
        help="T_mult parameter for CosineAnnealingWarmRestarts scheduler (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
    parser.add_argument('--topology', type=str, default='mlp', choices=['linear', 'cnn'],
        help='choose the toplogy for training and testing either linear or cnn')
    parser.add_argument('--mnist_fashion', type=bool, default=False, 
        help='choose between normal MNIST and Fashion MNIST')
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
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


    if not os.path.exists(options_cfg['log_dir']):
        os.makedirs(options_cfg['log_dir'])
        print("Directory created")

    # Set random seeds
        random.seed(train_cfg['seed'])
        np.random.seed(train_cfg['seed'])
        torch.manual_seed(train_cfg['seed'])
        os.environ['PYTHONHASHSEED'] = str(train_cfg['seed'])
        if options["cuda"]:
            torch.cuda.manual_seed_all(train_cfg['seed'])
            torch.backends.cudnn.deterministic = True


    if options_cfg['topology'] == 'linear': 
        # Load the model 
        model = MINSTmodelneq(model_cfg)
        if options_cfg["checkpoint"] is not None:
            print(f"Loading pre-trained checkpoint {options_cfg['checkpoint']}")
            checkpoint = torch.load(options_cfg["checkpoint"], map_location="cpu")
            model.load_state_dict(checkpoint["model_dict"])

        # train the model: 
        train_mlp(model, train_cfg, options_cfg)

    elif options_cfg['topology'] == 'cnn':
        # Load the model
        model = QuantizedMNIST_NEQ(model_cfg)
        if options_cfg["checkpoint"] is not None:
            print(f"Loading pre-trained checkpoint {options_cfg['checkpoint']}")
            checkpoint = torch.load(options_cfg["checkpoint"], map_location="cpu")
            model.load_state_dict(checkpoint["model_dict"])
        
        # train the model
        train_cnn(model, train_cfg, options_cfg)
