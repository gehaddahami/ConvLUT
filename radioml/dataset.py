'''
This file contains the data loader for the RadioML dataset. 
The class is created in a way that specific modulations can be imported and their relative sequence length can also be chosen. 
This is done to reduce the complexity to ensure that the concept of processing 1D-CNN via LogicNets Framework is properly working. 
- The dataset full content is to be added and processed in the future '''

# Imports 
import numpy as np 
import h5py 
import torch
from torch.utils.data import Dataset, SubsetRandomSampler

import sys 
sys.path.append('../../src')

# Dataloader Class 
class Radioml_18(Dataset):
    def __init__(self, dataset_path, snr_ratio: int = 0, sequence_length: int = None, selected_modulations=None): 
        super(Radioml_18, self).__init__()
        h5py_file = h5py.File(dataset_path, 'r')
        self.data = h5py_file['X']
        self.modulations = np.argmax(h5py_file['Y'], axis=1)
        self.snr = h5py_file['Z'][:, 0]
        self.len = self.data.shape[0]
        self.snr_ratio = snr_ratio
        self.sequence_length = sequence_length if sequence_length is not None else self.data.shape[1]

        # Default to BPSK and QPSK if no modulations are specified
        if selected_modulations is None:
            selected_modulations = ['BPSK', 'QPSK']

        self.mod_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
                            'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

        self.snr_classes = np.arange(-20., 31., 2)

        # Create a list of modulation indices to include based on the selected modulations
        mod_indices_to_include = [self.mod_classes.index(mod) for mod in selected_modulations]

        # Filter the data based on selected modulations
        data_masking = np.isin(self.modulations, mod_indices_to_include)
        self.data = self.data[data_masking]
        self.modulations = self.modulations[data_masking]
        self.snr = self.snr[data_masking]

        # Remap the modulation labels to sequential values
        self.label_mapping = {original_label: new_label for new_label, original_label in enumerate(mod_indices_to_include)}
        self.modulations = np.array([self.label_mapping[mod] for mod in self.modulations])

        np.random.seed(2018)
        train_indices = []
        validation_indices = []
        test_indices = []

        # Iterate over the selected modulation indices
        for new_mod_label in range(len(mod_indices_to_include)):
            mod_mask = self.modulations == new_mod_label
            mod_indices = np.where(mod_mask)[0]

            for snr_idx in range(0, 26):  # All signal to noise ratios from (-20, 30) dB
                snr_mask = self.snr[mod_indices] == self.snr_classes[snr_idx]
                indices_subclass = mod_indices[snr_mask]

                if len(indices_subclass) == 0:
                    continue

                np.random.shuffle(indices_subclass)
                train_indicies_sublcass = indices_subclass[:int(0.7 * len(indices_subclass))]
                validation_indices_subclass = indices_subclass[int(0.7 * len(indices_subclass)):int(0.85 * len(indices_subclass))]
                test_indicies_subclass = indices_subclass[int(0.85 * len(indices_subclass)):]

                if snr_idx >= snr_ratio:
                    train_indices.extend(train_indicies_sublcass)
                    validation_indices.extend(validation_indices_subclass)
                    test_indices.extend(test_indicies_subclass)

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(validation_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

        print('Filtered dataset shape:', self.data.shape)
        print("Train indices for selected SNRs:", len(train_indices))
        print("Validation indices for selected SNRs:", len(validation_indices))
        print("Test indices for selected SNRs:", len(test_indices))

        input_length = self.data.shape[1]
        print("Input length:", input_length)

    def __getitem__(self, index):
        assert self.sequence_length <= self.data.shape[1], \
            f"Sequence length {self.sequence_length} exceeds data length {self.data.shape[2]}"

        sequence = self.data[index, :self.sequence_length, :]
        label = self.modulations[index]
        return torch.tensor(sequence.transpose(), dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(self.snr[index])

    def __len__(self): 
        return self.data.shape[0]

    def get_original_label(self, new_label):
        """Get the original modulation label from the new label."""
        for orig_label, mapped_label in self.label_mapping.items():
            if mapped_label == new_label:
                return self.mod_classes[orig_label]
            


# Below is the original Dataloader adapted from the RadioML repository provided by XILINX 

class Radioml_18_original(Dataset):
    def __init__(self, dataset_path): 
        super(Radioml_18, self).__init__()
        h5py_file = h5py.File(dataset_path, 'r')
        self.data = h5py_file['X']
        self.modulations  = np.argmax(h5py_file['Y'], axis=1) 
        self.snr = h5py_file['Z'][:, 0]
        self.len = self.data.shape[0] 

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']

        self.snr_classes = np.arange(-20.,31.,2) 
        
        train_indices = []
        validation_indices = []
        test_indices = []
        for mod in range(0, 24): # All  24 modulationa
            for snr_idx in range(0, 26): # All signal to noise ratios from (-20, 30) Db
                start_index = 26*4096*mod + 4069*snr_idx 
                # Because X holds frames srticktly ordered by modulation and snr  
                indices_subclass = list(range(start_index, start_index+4096))


                # Splitting the data into 80% training and 20% testing 
                split = int(np.ceil(0.1*4096))
                np.random.shuffle(indices_subclass) 
                train_indicies_sublcass = indices_subclass[:int(0.7*len(indices_subclass))]
                validation_indices_subclass = indices_subclass[int(0.7*len(indices_subclass)):int(0.9*len(indices_subclass))]
                test_indicies_subclass = indices_subclass[int(0.9*len(indices_subclass)):] 
                
                # to choose a specific SNR valaue or range is here 
                if snr_idx >= 25: 
                    train_indices.extend(train_indicies_sublcass)
                    validation_indices.extend(validation_indices_subclass)
                    test_indices.extend(test_indicies_subclass)

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(validation_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

        print('Dataset shape:', self.data.shape)
        print("Train indices for SNRs 28, 30 dB:", len(train_indices))
        print("Validation indices for SNRs 28, 30 dB:", len(validation_indices))
        print("Test indices for SNRs 28, 30 dB:", len(test_indices)) 

        # Print input length
        input_length = self.data.shape[1]
        print("Input length:", input_length)

    
    def __getitem__(self, index):
        #Transform frame into pytorch channels-first format 
        return self.data[index].transpose(), self.modulations[index], self.snr[index]
    

    def __len__(self): 
        return self.len 
    