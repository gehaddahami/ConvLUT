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

'''
Additional functionality has been incorporated to train and test 1D-CNN functions 
'''

import os
from argparse import ArgumentParser
from functools import reduce
import random

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel, QuantizedJSC_CNN_NEQ

# TODO: Replace default configs with YAML files.
configs = {
    "jsc-s": {
        "hidden_layers": [64, 32, 32, 32],
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 3,
        "hidden_fanin": 3,
        "output_fanin": 3,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 1e-3,
        "seed": 2,
        "checkpoint": None,
    },
    "jsc-m": {
        "hidden_layers": [64, 32, 32, 32],
        "input_bitwidth": 3,
        "hidden_bitwidth": 3,
        "output_bitwidth": 3,
        "input_fanin": 4,
        "hidden_fanin": 4,
        "output_fanin": 4,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 1e-3,
        "seed": 3,
        "checkpoint": None,
    },
    "jsc-l": {
        "hidden_layers": [32, 64, 192, 192, 16],
        "input_bitwidth": 4,
        "hidden_bitwidth": 3,
        "output_bitwidth": 7,
        "input_fanin": 3,
        "hidden_fanin": 4,
        "output_fanin": 5,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 1e-3,
        "seed": 16,
        "checkpoint": None,
    },

    "jsc-cnn-s": {
        # "input_length": 1,
        "sequence_length": 16, 
        "hidden_layers": [4] * 2 + [64] + [32],
        "output_length": 1,
        "padding": 1, 
        "1st_layer_in_f": 64, 
        "input_bitwidth": 3,
        "hidden_bitwidth": 2,
        "output_bitwidth": 3,
        "input_fanin": 1,
        "conv_fanin": 2,
        "hidden_fanin": 4,
        "output_fanin": 4,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 1e-3,
        "checkpoint": None,
        "seed": 16,
    }, 
    "jsc-cnn-l": {
        # "input_length": 1,
        "sequence_length": 16, 
        "hidden_layers": [4] * 4 + [128] * 2,
        "output_length": 1,
        "padding": 1, 
        "1st_layer_in_f": 64, 
        "input_bitwidth": 5,
        "hidden_bitwidth": 2,
        "output_bitwidth": 5,
        "input_fanin": 1,
        "conv_fanin": 2,
        "hidden_fanin": 4,
        "output_fanin": 5,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 100,
        "learning_rate": 1e-3,
        "checkpoint": None,
        "seed": 16,
    }
}

# A dictionary, so we can set some defaults if necessary
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
}

training_config = {
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
}

dataset_config = {
    "dataset_file": None,
    "dataset_config": None,
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
}

def train(model, datasets, train_cfg, options, topology_type='cnn'):
    # Create data loaders for training and inference:
    train_loader = DataLoader(datasets["train"], batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(datasets["valid"], batch_size=train_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=train_cfg['batch_size'], shuffle=False)

    # Configure optimizer
    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = ["bn", "bias", "learned_value"] # Make a list of parameters name fragments which will ignore weight decay TODO: make this list part of the train_cfg
    decay_params = []
    no_decay_params = []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if reduce(lambda a,b: a or b, map(lambda x: x in pname, decay_exclusions)): # check if the current label should be excluded from weight decay
                #print("Disabling weight decay for %s" % (pname))
                no_decay_params.append(params)
            else:
                #print("Enabling weight decay for %s" % (pname))
                decay_params.append(params)
        #else:
            #print("Ignoring %s" % (pname))
    params =    [{'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}]
    optimizer = optim.AdamW(params, lr=train_cfg['learning_rate'], betas=(0.5, 0.999), weight_decay=weight_decay)

    # Configure scheduler
    steps = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps*100, T_mult=1)

    # Configure criterion
    criterion = nn.CrossEntropyLoss()

    # Push the model to the GPU, if necessary
    if options["cuda"]:
        model.cuda()

    # Setup tensorboard
    writer = SummaryWriter(options["log_dir"])

    # Main training loop
    maxAcc = 0.0
    num_epochs = train_cfg["epochs"]
    for epoch in range(0, num_epochs):
        # Train for this epoch
        model.train()
        accLoss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            if topology_type == 'cnn':
                # Reshape from (batch_size, features) to (batch_size, channels, height, width)
                data = data.view(data.size(0), 1, -1)

            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            pred = output.detach().max(1, keepdim=True)[1]
            target_label = torch.max(target.detach(), 1, keepdim=True)[1]
            curCorrect = pred.eq(target_label).long().sum()
            curAcc = 100.0*curCorrect / len(data)
            correct += curCorrect
            accLoss += loss.detach()*len(data)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log stats to tensorboard
            #writer.add_scalar('train_loss', loss.detach().cpu().numpy(), epoch*steps + batch_idx)
            #writer.add_scalar('train_accuracy', curAcc.detach().cpu().numpy(), epoch*steps + batch_idx)
            #g = optimizer.param_groups[0]
            #writer.add_scalar('LR', g['lr'], epoch*steps + batch_idx)

        accLoss /= len(train_loader.dataset)
        accuracy = 100.0*correct / len(train_loader.dataset)
        print(f"Epoch: {epoch}/{num_epochs}\tTrain Acc (%): {accuracy.detach().cpu().numpy():.2f}\tTrain Loss: {accLoss.detach().cpu().numpy():.3e}")
        #for g in optimizer.param_groups:
        #        print("LR: {:.6f} ".format(g['lr']))
        #        print("LR: {:.6f} ".format(g['weight_decay']))
        writer.add_scalar('avg_train_loss', accLoss.detach().cpu().numpy(), (epoch+1)*steps)
        writer.add_scalar('avg_train_accuracy', accuracy.detach().cpu().numpy(), (epoch+1)*steps)
        val_accuracy, val_avg_roc_auc = test(model, val_loader, options["cuda"])
        test_accuracy, test_avg_roc_auc = test(model, test_loader, options["cuda"])
        modelSave = {   'model_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'val_avg_roc_auc': val_avg_roc_auc,
                        'test_avg_roc_auc': test_avg_roc_auc,
                        'epoch': epoch}
        torch.save(modelSave, options["log_dir"] + "/checkpoint.pth")
        if(maxAcc<val_accuracy):
            torch.save(modelSave, options["log_dir"] + "/best_accuracy.pth")
            maxAcc = val_accuracy
        writer.add_scalar('val_accuracy', val_accuracy, (epoch+1)*steps)
        writer.add_scalar('test_accuracy', test_accuracy, (epoch+1)*steps)
        writer.add_scalar('val_avg_roc_auc', val_avg_roc_auc, (epoch+1)*steps)
        writer.add_scalar('test_avg_roc_auc', test_avg_roc_auc, (epoch+1)*steps)
        print(f"Epoch: {epoch}/{num_epochs}\tValid Acc (%): {val_accuracy:.2f}\tTest Acc: {test_accuracy:.2f}")


def test(model, dataset_loader, cuda, topology_type='cnn'):
    with torch.no_grad():
        model.eval()
        entire_prob = None
        golden_ref = None
        correct = 0
        accLoss = 0.0
        for batch_idx, (data, target) in enumerate(dataset_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()

            if topology_type == 'cnn':
                # Reshape from (batch_size, features) to (batch_size, channels, height, width)
                data = data.view(data.size(0), 1, -1)
            output = model(data)
            prob = F.softmax(output, dim=1)
            pred = output.detach().max(1, keepdim=True)[1]
            
            target_label = torch.max(target.detach(), 1, keepdim=True)[1]
            curCorrect = pred.eq(target_label).long().sum()
            curAcc = 100.0*curCorrect / len(data)
            correct += curCorrect
            if batch_idx == 0:
                entire_prob = prob
                golden_ref = target_label
            else:
                entire_prob = torch.cat((entire_prob, prob), dim=0)
                golden_ref = torch.cat((golden_ref, target_label))
        accuracy = 100*float(correct) / len(dataset_loader.dataset)
        avg_roc_auc = roc_auc_score(golden_ref.detach().cpu().numpy(), entire_prob.detach().cpu().numpy(), average='macro', multi_class='ovr')
        return accuracy, avg_roc_auc


if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets Jet Substructure Classification Example")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="jsc-s",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--weight-decay', type=float, default=None, metavar='D',
        help="Weight decay (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=None, metavar='LR',
        help="Initial learning rate (default: %(default)s)")
    parser.add_argument('--cuda', action='store_true', default=False,
        help="Train on a GPU (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=None,
        help="Seed to use for RNG (default: %(default)s)")
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
    parser.add_argument('--1st-layer-in-f', type=int, default=None,
        help="The input feature size of the first layer (default: %(default)s)")
    parser.add_argument('--padding', type=str, default=None,
        help="The padding to use for the input sequence (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--dataset-file', type=str, default='data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--dataset-config', type=str, default='config/yaml_IP_OP_config.yml',
        help="The file to use to configure the input dataset (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
    parser.add_argument('--topology', type = str, default=None, 
        help="Type of topology to use (default: %(default)s)")
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]

    # Set random seeds
    random.seed(train_cfg['seed'])
    np.random.seed(train_cfg['seed'])
    torch.manual_seed(train_cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(train_cfg['seed'])
    if options["cuda"]:
        torch.cuda.manual_seed_all(train_cfg['seed'])
        torch.backends.cudnn.deterministic = True

    # Fetch the datasets
    dataset = {}
    dataset['train'] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="train")
    dataset['valid'] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="train") 
    dataset['test'] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="test")

    # Instantiate model
    x, y = dataset['train'][0]
    if args.topology == 'cnn':
        model_cfg['input_length'] = 1 #len(x)
        model_cfg['sequence_length'] = len(x) 
        print('model_cfg:', model_cfg)
        model_cfg['output_length'] = len(y)
        torch.manual_seed(train_cfg['seed'])
        np.random.seed(train_cfg['seed'])
        model = QuantizedJSC_CNN_NEQ(model_cfg)
    else:
        model_cfg['input_length'] = len(x)
        print('model_cfg:', model_cfg)
        model_cfg['output_length'] = len(y)
        torch.manual_seed(train_cfg['seed'])
        np.random.seed(train_cfg['seed'])
        model = JetSubstructureNeqModel(model_cfg)

    if options_cfg['checkpoint'] is not None:
        print(f"Loading pre-trained checkpoint {options_cfg['checkpoint']}")
        checkpoint = torch.load(options_cfg['checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_dict'])

    train(model, dataset, train_cfg, options_cfg, topology_type='cnn')
