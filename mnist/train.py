import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

from model import MINSTmodelneq
# Define the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST model with customizable parameters.")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate (default: 0.004)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer (default: 0)')
    parser.add_argument('--cuda', default=False, help='Enable CUDA training')
    parser.add_argument('--seed', type=int, default=984237, help='Random seed (default: 984237)')
    parser.add_argument('--log_dir', type=str, default= '/home/student/Desktop/CNN_LogicNets/MINST/log_file', help='Directory for saving model checkpoints')
    
    return parser.parse_args()

# Function to train the model
def train(model, train_cfg, options):
    # Create data loaders for training and validation
    train_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=True,
            train=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=True,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=True,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
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
            data = data.reshape(-1, 784)
            target = torch.nn.functional.one_hot(target, num_classes=10).float()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()

        val_accuracy = test(model, val_loader, options["cuda"])
        test_accuracy = test(model, test_loader, options["cuda"])

        print(f'Epoch {epoch}: Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

        if maxAcc < test_accuracy:
            torch.save(model.state_dict(), f"{options['log_dir']}/best_model_minst_logicnets.pth")
            maxAcc = test_accuracy

# Function to evaluate the model
def test(model, dataset_loader, cuda):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data = data.reshape(-1, 784)
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
    
    accuracy = 100.0 * correct / len(dataset_loader.dataset)
    return accuracy


# Set seeds and prepare environment
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(0)

    # Model configuration
    model_cfg = {
        "hidden_layers": [256, 100, 100, 100, 100],
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 6,
        "degree": 4,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "checkpoint": None,
        "input_length": 784,
        "output_length": 10
    }

    other_options = {
        "cuda": args.cuda,
        "log_dir": args.log_dir,
        "checkpoint": None,
        "device": 0,
    }

    model = MINSTmodelneq(model_cfg)

    train(model, model_cfg, other_options)

if __name__ == '__main__':
    main()
