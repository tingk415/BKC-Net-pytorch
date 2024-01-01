import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)


class RCC3D(Dataset):
    def __init__(self, num_samples, kfold, is_train):
        self.num_samples = num_samples
        self.kfold = kfold
        self.is_train = is_train
        self.patient_ids = [f"kits_19_case_{str(i).zfill(5)}" for i in range(210)] 
        self.image_paths = [f"G:\\kits\\kits19\\case_{str(i).zfill(5)}\\imaging.nii.gz" for i in range(210)]  # List of image file paths
        self.mask_paths = [f"G:\\kits\\kits19\\case_{str(i).zfill(5)}\\segmentation.nii.gz" for i in range(210)]  # List of mask file paths
        self.train_transform = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys="image", image_only=True,),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                RandSpatialCropd(keys="image", roi_size=[224, 224, 144], random_size=False),
                RandFlipd(keys="image", prob=0.5, spatial_axis=0),
                RandFlipd(keys="image", prob=0.5, spatial_axis=1),
                RandFlipd(keys="image", prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        self.val_transform = Compose(
            [
                LoadImaged(keys="image", image_only=True,),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

        if self.is_train:
            self.transform = self.train_transform
        else:
            self.transform = self.val_transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        patient_id = self.patient_ids[index]

        # Load the CT image and mask using MONAI's NibabelReader
        image_tensor = self.transform({"image": image_path})["image"]
        mask_tensor = self.transform({"image": mask_path})["image"]

        return patient_id, image_tensor, mask_tensor

