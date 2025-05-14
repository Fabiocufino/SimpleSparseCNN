import torch
from torchvision import datasets, transforms
import pickle as pk
import numpy as np
import torch
import os
import sys
from glob import glob
from torch.utils.data import Dataset
import MinkowskiEngine as ME


class XYProjectionFASERCALDataset(Dataset):
    def __init__(self, args):
        """
        Initializes the XYProjectionFASERCALDataset class.

        Args:
        root (str): Root directory containing the data files.
        shuffle (bool): Whether to shuffle the dataset (default: False).
        """
        self.root = args.dataset_path
        self.data_files = self.processed_file_names
        self.train = args.train
        self.is_v5 = True if 'v5' in args.dataset_path else False 
        self.total_events = self.__len__


        with open(self.root + "/metadata.pkl", "rb") as fd:
            self.metadata = pk.load(fd)
            self.metadata['x'] = np.array(self.metadata['x'])
            self.metadata['y'] = np.array(self.metadata['y'])
            self.metadata['z'] = np.array(self.metadata['z'])

            
    @property
    def processed_dir(self):
        """
        Returns the processed directory path.

        Returns:
        str: Path to the processed directory.
        """
        return f'{self.root}'
    
    
    @property
    def processed_file_names(self):
        """
        Returns a list of processed file names.

        Returns:
        list: List of file names.
        """
        return glob(f'{self.processed_dir}/*.npz')
    
    
    def __len__(self):
        """
         Returns the total number of data files.

         Returns:
         int: Number of data files.
         """
        return len(self.data_files)
        
        
    def pdg2label(self, pdg, iscc, name=False):
        """
        PDG to label.
        """
        if iscc:
            if pdg in [-12, 12]:
                label = "CC nue" if name else 0
            elif pdg in [-14, 14]:
                label = "CC numu" if name else 1
            elif pdg in [-16, 16]:
                label = "CC nutau" if name else 2
        else:
            label = "NC" if name else 3

        return label
    
    
    
    def divide_by_std(self, x, param_name):
        return x / self.metadata[param_name]['std']
    
    
    def get_param(self, data, param_name, preprocess=False):
        if param_name not in data:
            return None

        param = data[param_name]
        if param.ndim == 0:
            param = param.reshape(1,) if preprocess else param.item()
        param = self.divide_by_std(param, param_name) if preprocess else param
        
        return param

    
    def __getitem__(self, idx):
        """
        Retrieves a data sample by index.

        Args:
        idx (int): Index of the data sample.

        Returns:
        dict: Data sample with filename, coordinates, features, and labels.
        """
        data = np.load(self.data_files[idx], allow_pickle=True)
        
        run_number = self.get_param(data, 'run_number')
        event_id = self.get_param(data, 'event_id')
        
        '''This is toy model'''
        xz_proj = self.get_param(data,'xz_proj')
        '''This is toy model'''

        in_neutrino_pdg = self.get_param(data, 'in_neutrino_pdg')
        is_cc = self.get_param(data, 'is_cc')


        rear_cal_energy = self.get_param(data, 'rear_cal_energy', preprocess=True)
        rear_hcal_energy = self.get_param(data, 'rear_hcal_energy', preprocess=True)
        rear_mucal_energy = self.get_param(data, 'rear_mucal_energy', preprocess=True)
        faser_cal_energy = self.get_param(data, 'faser_cal_energy', preprocess=True)
        
        # prepare labels 
        flavour_label = self.pdg2label(in_neutrino_pdg, is_cc)

        coords = xz_proj[:,:2]
        q = self.divide_by_std(xz_proj[:,2].reshape(-1, 1), 'q')
        feats = q
    
        feats_global = np.concatenate([rear_cal_energy, rear_hcal_energy,
                                        rear_mucal_energy, faser_cal_energy])   
                                        

        # output
        output = {}
        output['flavour_label'] = torch.tensor([flavour_label]).long()

        if not self.train:
            output['run_number'] = run_number
            output['event_id'] = event_id
            output['is_cc'] = is_cc
            output['in_neutrino_pdg'] = in_neutrino_pdg
    
        output['coords'] = torch.from_numpy(coords.reshape(-1, 2)).float()
        output['feats'] = torch.from_numpy(feats).float()
        output['feats_global'] = torch.from_numpy(feats_global).float()

        return output
