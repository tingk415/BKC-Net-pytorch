import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json

class CCRCC3D(Dataset):
    def __init__(self, num_samples, kfold, is_train):
        self.num_samples = num_samples
        self.kfold = kfold
        self.is_train = is_train
        self.patient_ids = [f"kits_19_case_{str(i).zfill(5)}" for i in range(210)]  # List of patient IDs

        self.params = os.path.abspath(os.path.join('utils','RadiomicsFeaturesSettings','radiomicsFeatures.yaml'))
        self.rdm_matrix = np.load(os.path.abspath(os.path.join('utils','RadiomicsFeaturesOutput','normalized_radiomics_features.npy')))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]

        rdmfea = self.rdm_matrix[index]

        return patient_id, dlfea, rdmfea, ccrcc_cls

