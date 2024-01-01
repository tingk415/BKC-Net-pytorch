import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import json
from model.ConAttUnet import ConAttUNet
from monai.transforms import (
    Compose,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)

class RCC3D(Dataset):
    def __init__(self, num_samples, kfold, is_train):
        self.num_samples = num_samples
        self.kfold = kfold
        self.is_train = is_train
        self.patient_ids = [f"kits_19_case_{str(i).zfill(5)}" for i in range(210)]  # List of patient IDs
        self.image_paths = [f"G:\\kits\\kits19\\case_{str(i).zfill(5)}\\imaging.nii.gz" for i in range(210)]

        self.params = os.path.abspath(os.path.join('utils','RadiomicsFeaturesSettings','radiomicsFeatures.yaml'))
        self.rdm_matrix = np.load(os.path.abspath(os.path.join('utils','RadiomicsFeaturesOutput','normalized_radiomics_features.npy')))

        # Load labels from JSON file
        with open('G:\\kits\\kits19\\kits19.json', 'r') as f:
            labels_data = json.load(f)

        label_to_value = {
            "clear_cell_rcc": 0,
            "clea_cell_papillary_rcc":1, 
            "papillary":2,
            "chromophobe":3, 
            "urothelial":4, 
            "rcc_unclassified":5,
            "multilocular_cystic_rcc":6,
            "wilms":7, 
            "oncocytoma":8,
            "angiomyolipoma":9, 
            "mest":10, 
            "spindle_cell_neoplasm":11
        }
        # 12 classes, convert to one hot encoding
        self.ccrcc_class = F.one_hot(torch.from_numpy([label_to_value[obj['tumor_histologic_subtype']] for obj in labels_data]), 12)

        self.model= ConAttUNet(n_channels=1, n_classes=1)
        
        self.model.load_state_dict(torch.load('results/weights/fpl_epoch_100_best_a.pth'))
        # self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.val_transform =  Compose(
            [
                LoadImaged(keys="image", image_only=True,),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        patient_id = self.patient_ids[index]

        rdmfea = torch.from_numpy(self.rdm_matrix[index])
        ccrcc_cls = self.ccrcc_class[index]

        image_tensor = self.transform({"image": image_path})["image"]

        dlfea = self.model(image_tensor)

        return patient_id, dlfea, rdmfea, ccrcc_cls

