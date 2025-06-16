# This file is Part of Conv-LUT
# Conv-LUT is based on LogicNets  

'''
This file contains the data loader for the RadioML dataset. 
The class is created in a way that specific modulations can be imported and their relative sequence length can also be chosen. 
This is done to reduce the complexity to ensure that the concept of processing 1D-CNN via LogicNets Framework is properly working. 
- The dataset full content is to be added and processed in the future '''

import os
import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split

import sys 
sys.path.append('../../src')

# Dataloader Class 
class MITBIHDataset(Dataset):
    def __init__(self, data_path, sequence_length=200, test_size=0.2, seed=20):
        """
        MIT-BIH Arrhythmia Dataset Loader.
        :param records_path: Path to the MIT-BIH dataset folder.
        :param sequence_length: Length of each ECG sample (default: 200 samples per beat).
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.data = []
        self.labels = []

        # Available records (manually filtered to exclude missing ones)
        available_records = [
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 
            117, 118, 119, 121, 122, 123, 124, 
            200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 
            219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234
        ]  

        label_mapping = {
            'N': 0, 'L': 0, 'R': 0,  
            'V': 1, '[': 1, '!': 1, ']': 1, 'E': 1, 
            'A': 2, 'a': 2, 'e': 2, 'j': 2,  
            'J': 3, 'S': 3, 'F': 3, 'f': 3,  
            '/': 4, '"': 4, 'x': 4, 'Q': 4, '|': 4, '~': 4, '+': 4
            }
        
        for record_num in available_records:
            record_path = os.path.join(data_path, str(record_num))
            
            try:
                # Load ECG signal
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, "atr")

                signal = record.p_signal[:, 0]  # Use Lead II (MLII)
                beats = annotation.sample  # R-peak locations
                labels = annotation.symbol  # Beat labels

                # Extract ECG segments around R-peaks
                for i in range(len(beats)):
                    start = max(0, beats[i] - sequence_length // 2)
                    end = start + sequence_length
                    
                    # Ensure fixed length
                    if end > len(signal):
                        continue  

                    self.data.append(signal[start:end])
                    self.labels.append(labels[i])

            except FileNotFoundError:
                print(f"Skipping missing file: {record_num}")

        # Encode labels (convert to numbers)
        # Map the labels to the 5 classes
        self.labels = [label_mapping.get(label, 4) for label in self.labels]  # Default to 4 (Unclassified) if not found
        print(np.unique(self.labels))
        self.data = np.array(self.data)

        # Split dataset into train, val, and test
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        train_indices, test_val_indices = train_test_split(indices, test_size=test_size, random_state=seed)
        val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, random_state=seed)

        # Create samplers
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)
    