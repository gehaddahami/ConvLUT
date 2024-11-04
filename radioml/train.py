import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importing the model and the data loader
from dataset import Radioml_18
from model import QuantizedRadiomlNEQ
from train_test_loops import train_loop_pytorch, train_logicnets, val_test_pytorch, test_logicnets, display_loss, plot_training_results


options = {
    'cuda' : None,
    'log_dir' : '/home/student/Desktop/CNN_LogicNets/radioml/log', 
    'checkpoint' : None 
    }
	

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate QuantizedRadioml CNN model")
    parser.add_argument('--dataset_path', type=str, default='/home/student/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', help='Path to the dataset') # if the default is removed add the argument required = True
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders')
    parser.add_argument('--snr_ratio', type=int, default=25, help='The desired signal-to-noice Ratio')
    parser.add_argument('--sequence_length', type=int, default=64, help='the signal desired sequence length -if faster processing is desired-')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--t0', type=int, default=5, help='T_0 parameter for CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--t_mult', type=int, default=1, help='T_mult parameter for CosineAnnealingWarmRestarts scheduler')
    return parser.parse_args()



def load_data(dataset_path, batch_size, snr):
    args = parse_args()
    dataset = Radioml_18(dataset_path, sequence_length=args.sequence_length, snr_ratio=snr)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.validation_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)
    return train_loader, validation_loader, test_loader


def create_model():
    args = parse_args()
    model_config = {
        "input_length": 2,
        "sequence_length": args.sequence_length, 
        "hidden_layers": [4] * 4 + [32] * 2,
        "output_length": 24,
        "padding": 1, 
        "1st_layer_in_f": 16, 
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 3,
        "input_fanin": 2,
        "conv_fanin": 2,
        "hidden_fanin": 4,
        "output_fanin": 4
    }
    model = QuantizedRadiomlNEQ(model_config=model_config)
    return model


def main():
    args = parse_args()

    # Load data
    train_loader, validation_loader, test_loader = load_data(args.dataset_path, args.batch_size, args.snr_ratio)

    # Create model
    model = create_model()
    
    # load from checkpoint if available: 
    if options['checkpoint'] is not None:
        print(f'Loading pre-trained checkpoint {options["checkpoint"]}')
        checkpoint = torch.load(options['checkpoint'], map_location = 'cpu')
        model.load_state_dict(checkpoint['model_dict'])
        model_loaded_from_checkpoint = True
        print(f'Checkpoint loaded successfully')

    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)

    # Training loop
    running_loss = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in tqdm(range(args.num_epochs)):
        loss_epoch, acc_train = train_logicnets(model, train_loader, optimizer, criterion, options)
        val_accuracy = val_test_pytorch(model, validation_loader, options)
        print(f"Epoch {epoch}: Training loss = {loss_epoch:.6f}, validation accuracy = {val_accuracy:.6f}")
        running_loss.append(loss_epoch)
        train_accuracies.append(acc_train)
        val_accuracies.append(val_accuracy)

        # Step the scheduler
        scheduler.step()

    test_accuracy = val_test_pytorch(model, test_loader, options)
    print(f"Test accuracy = {test_accuracy:.6f}")

    # Plot the running loss and accuracy
    plot_training_results(running_loss, val_accuracies)

    if options["checkpoint"] is None:
        model_save = {
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'epoch': epoch
        }

        torch.save(model_save, options["log_dir"] + "/checkpoint_radioml.pth")


if __name__ == "__main__":
    main()