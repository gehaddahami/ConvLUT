import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse

from model import MINSTmodelneq, QuantizedMNIST_NEQ
# Define the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST model with customizable parameters.")
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate (default: 0.004)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer (default: 0)')
    parser.add_argument('--cuda', default=False, help='Enable CUDA training')
    parser.add_argument('--seed', type=int, default=984237, help='Random seed (default: 984237)')
    parser.add_argument('--log_dir', type=str, default= '/home/student/Desktop/CNN_LogicNets/MINST/log_file', help='Directory for saving model checkpoints')
    parser.add_argument('--topology', type=str, default='cnn', help='choose the toplogy for training and testing either linear or cnn')

    
    return parser.parse_args()

# Function to train the model
def train_mlp(model, train_cfg, options):
    # Create data loaders for training and validation
    train_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=True,
            train=True,
            transform=transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=True,
            train=False,
            transform=transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=True,
            train=False,
            transform=transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1309,), (0.3081,))]),
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
            data = data.reshape(-1, 256)
            target = torch.nn.functional.one_hot(target, num_classes=10).float()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            loss.backward()
            optimizer.step()

        val_accuracy = test_mlp(model, val_loader, options["cuda"])
        test_accuracy = test_mlp(model, test_loader, options["cuda"])

        print(f'Epoch {epoch}: Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

        if maxAcc < test_accuracy:
            torch.save(model.state_dict(), f"{options['log_dir']}/best_model_minst_logicnets_MLP.pth")
            maxAcc = test_accuracy

# Function to evaluate the model
def test_mlp(model, dataset_loader, cuda):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data = data.reshape(-1, 256)
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
    
    accuracy = 100.0 * correct / len(dataset_loader.dataset)
    return accuracy



def train_cnn(model, args, options):
    # Create data loaders for training and validation
    # Dataset and transformations
    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Full training dataset
    full_train_dataset = datasets.MNIST("mnist_data", download=True, train=True, transform=transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    # Split training dataset into training and validation sets
    train_size = int(0.8 * len(full_train_dataset))  # 80% training
    val_size = len(full_train_dataset) - train_size  # 20% validation
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Test dataset
    test_dataset = datasets.MNIST("mnist_data", download=True, train=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Configure optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if options["cuda"]:
        model.cuda()

    maxAcc = 0.0
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        acc_loss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), 1, -1)  # Flatten spatial dimensions into 1D sequence
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
    
            optimizer.zero_grad()
            
            # Reshape the data for 1D CNN: (batch_size, 1, 28*28)


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

        torch.save(model_save, f"{options['log_dir']}/best_model_mnist_1dcnn.pth")
        print(f"Model saved at {options['log_dir']}/best_model_mnist_1dcnn.pth")
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
            data = data.view(data.size(0), 1, -1)  # Shape: (batch_size, channels, sequence_length)

            # Forward pass
            output = model(data)
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Set seeds and prepare environment
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(0)
    if args.topology == 'linear' : 

        # Model configuration
        model_cfg = {
            "hidden_layers": [256, 150, 150, 150, 100],
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
            "input_length": 256,
            "output_length": 10
        }

        other_options = {
            "cuda": args.cuda,
            "log_dir": args.log_dir,
            "checkpoint": None,
        }

        model = MINSTmodelneq(model_cfg)

        train_mlp(model, model_cfg, other_options)

    elif  args.topology == 'cnn':   
        model_config = {
                "input_length": 1,
                "sequence_length": 256, 
                "hidden_layers": [6] * 4 + [128] * 2,
                "output_length": 10,
                "padding": 1, 
                "1st_layer_in_f": 1536, #change whenever the sequence length changed 
                "input_bitwidth": 6,
                "hidden_bitwidth": 2,
                "output_bitwidth": 2,
                "input_fanin": 1,
                "conv_fanin": 2,
                "hidden_fanin": 6,
                "output_fanin": 6
            }

        other_options = {
            "cuda": args.cuda,
            "log_dir": args.log_dir,
            "checkpoint": None,
            "device": 0,
        }


        model = QuantizedMNIST_NEQ(model_config)
        train_cnn(model, args, other_options) 

if __name__ == '__main__':
    main()
