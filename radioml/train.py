# Description: This script is used to train the QuantizedRadiomlNEQ model on the RadioML 2018.01 dataset.
# NOTE Important!! : before training the model, make sure that the number of layers are matching with those in the model class, 
# The project is based on LogicNets, which is under the apache 2.0 license. 


from argparse import ArgumentParser
import os 
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importing the model and the data loader
from dataset import Radioml_18
from model import QuantizedRadiomlNEQ
from train_test_loops import train_logicnets, test_logicnets, plot_training_results



configs = {
    "2mod-s": {
        "input_length": 2,
        "sequence_length": 128,
        "hidden_layers": [4] * 4 + [32, 32],
        "output_length": 2,
        "padding": 1,
        "1st_layer_in_f": 128,
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 2,
        "conv_fanin": 2,
        "hidden_fanin": 7,
        "output_fanin": 7, 
        "selected_modulations": ['BPSK', 'QPSK'],
        "checkpoint": None, 
        "batch_size": 1024,
        "snr_ratio": 0,
        "epochs": 20,
        "t0": 5,
        "t_mult": 1
    },
    "2mod-l": {
        "input_length": 2,
        "sequence_length": 128,
        "hidden_layers": [8] * 4 + [64, 64],
        "output_length": 2,
        "padding": 1,
        "1st_layer_in_f": 256,
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 2,
        "conv_fanin": 2,
        "hidden_fanin": 7,
        "output_fanin": 7,
        "selected_modulations": ['BPSK', 'QPSK'],
        "checkpoint": None, 
        "batch_size": 1024,
        "snr_ratio": 0,
        "epochs": 20,
        "t0": 5,
        "t_mult": 1
    },
    "psk-5": {
        "input_length": 2,
        "sequence_length": 256,
        "hidden_layers": [16] * 14 + [128, 128],
        "output_length": 5,
        "padding": 1,
        "1st_layer_in_f": 32,
        "input_bitwidth": 3,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 2,
        "conv_fanin": 2,
        "hidden_fanin": 7,
        "output_fanin": 7,
        "selected_modulations": ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK'],
        "checkpoint": None, 
        "batch_size": 1024,
        "snr_ratio": 0,
        "epochs": 20,
        "t0": 5,
        "t_mult": 1
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
    "padding": None,
    "1st_layer_in_f": None, 
    'input_length': None,
    'output_length': None
}


train_cfg = {
    "sequence_length": None,
    "selected_modulations": None,
    "snr_ratio": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
    "t0": None,
    "t_mult": None,
}

dataset_config = {
    "dataset_path": None,
}

options = {
    'cuda' : None,
    'log_dir' : None, 
    'checkpoint' : None
}
	


def train(model, dataset, train_cfg, options): 

    if not os.path.exists(options['log_dir']):
        os.makedirs(options['log_dir'])
        print('directory created..................................')

    # Create loaders: 
    train_loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], sampler=dataset.train_sampler)
    validation_loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], sampler=dataset.validation_sampler)
    test_loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], sampler=dataset.test_sampler)

    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg['t0'], T_mult=train_cfg['t_mult'])

    running_loss = []
    train_accuracies = []
    val_accuracies = []


    for epoch in tqdm(range(train_cfg['epochs'])):
        loss_epoch, acc_train = train_logicnets(model, train_loader, optimizer, criterion, options)
        val_accuracy = test_logicnets(model, validation_loader, options, dataset, test=False)
        print(f"Epoch {epoch}: Training loss = {loss_epoch:.6f}, validation accuracy = {val_accuracy:.6f}")
        running_loss.append(loss_epoch)
        train_accuracies.append(acc_train)
        val_accuracies.append(val_accuracy)

        # Step the scheduler
        scheduler.step()

    test_accuracy = test_logicnets(model, test_loader, options, dataset, test=True)
    print(f"Test accuracy = {test_accuracy:.6f}")

    # Plot the running loss and accuracy
    plot_training_results(running_loss, val_accuracies, log_dir=options["log_dir"], plot_name=f'loss_curve.pdf')


    if options["checkpoint"] is None:
        model_save = {
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'epoch': epoch
        }

        torch.save(model_save, options["log_dir"] + "/config_1.pth")

if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets Classification (RadioML Example)")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="psk-5",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
        help="Initial learning rate (default: %(default)s)")
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
    parser.add_argument('--snr-ratio', type=int, default=None,
        help="SNR ratio to use (default: %(default)s)")
    parser.add_argument('--selected_modulations', nargs='+', type=str, default=None,
        help="The modulations to include in the dataset (default: %(default)s)")
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
    parser.add_argument('--dataset_path', type=str, default='/home/student/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
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
    training_cfg = {}
    for k in train_cfg.keys():
        training_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in options.keys():
        options_cfg[k] = config[k]

    # Set random seeds
        random.seed(training_cfg['seed'])
        np.random.seed(training_cfg['seed'])
        torch.manual_seed(training_cfg['seed'])
        os.environ['PYTHONHASHSEED'] = str(training_cfg['seed'])
        if options["cuda"]:
            torch.cuda.manual_seed_all(training_cfg['seed'])
            torch.backends.cudnn.deterministic = True



    # Load data
    dataset = Radioml_18(dataset_cfg['dataset_path'], sequence_length=training_cfg['sequence_length'], snr_ratio=training_cfg['snr_ratio'], selected_modulations=training_cfg['selected_modulations'])

    # Create model
    model = QuantizedRadiomlNEQ(model_config=model_cfg)
    if options_cfg['checkpoint'] is not None:
            print(f'Loading pre-trained checkpoint {options_cfg["checkpoint"]}')
            checkpoint = torch.load(options_cfg['checkpoint'], map_location = 'cpu')
            model.load_state_dict(checkpoint['model_dict'])
            model_loaded_from_checkpoint = True
            print(f'Checkpoint loaded successfully')

    # Train the model
    train(model, dataset, training_cfg, options)