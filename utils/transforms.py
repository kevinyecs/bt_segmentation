import torch
import torch.nn as nn

from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    CropForegroundd,
    Resized,
    SpatialPadd,
    DivisiblePadd,
    RandGaussianSmoothd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandAdjustContrastd,
    ToTensord,
    RandGaussianSmoothd
)


class DatasetTransforms(nn.Module):
    def __init__(self, task):
        self.task = task
    
    def get_transforms(self, tr_type = None):
        if self.task == 'Task01_BrainTumour':
            if tr_type == "nnUnet":
                train_transform = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys="image"),
                        EnsureTyped(keys=["image", "label"]),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                        Spacingd(
                            keys=["image", "label"],
                            pixdim=(1.0, .77, .77),
                            mode=("bilinear", "nearest"),
                        ),
                        CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        k_divisible=[128,128,128],),   
                        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
                        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                        RandRotated(keys=["image", "label"], prob=0.2, range_x=[0.3, 0.3]),
                        RandRotated(keys=["image", "label"], prob=0.2, range_y = [0.3, 0.3]),
                        RandRotated(keys=["image", "label"], prob=0.2, range_z = [0.3, 0.3]),
                        RandZoomd(keys=["image", "label"], prob = .2,  min_zoom=0.7,  max_zoom=1.4),
                        RandAdjustContrastd(keys="image", prob=.15, gamma=(0.65, 1.5)),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                        RandGaussianNoised(keys="image", prob=.15, mean= 0, std = 0.1),
                        RandGaussianSmoothd(keys="image", prob=.2 ,sigma_x=(0.5, 1.5)),
                        RandGaussianSmoothd(keys="image", prob=.2 ,sigma_y=(0.5, 1.5)),
                        RandGaussianSmoothd(keys="image", prob=.2 ,sigma_z=(0.5, 1.5)),
                    ]
                )

                val_transform = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys="image"),
                        EnsureTyped(keys=["image", "label"]),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                        Spacingd(
                            keys=["image", "label"],
                            pixdim=(1.0, .77, .77),
                            mode=("bilinear", "nearest"),
                        ),
                        CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        k_divisible=[128,128,128],),   
                        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
                        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                        RandRotated(keys=["image", "label"], prob=0.2, range_x=[0.3, 0.3]),
                        RandRotated(keys=["image", "label"], prob=0.2, range_y = [0.3, 0.3]),
                        RandRotated(keys=["image", "label"], prob=0.2, range_z = [0.3, 0.3]),
                        RandZoomd(keys=["image", "label"], prob = .2,  min_zoom=0.7,  max_zoom=1.4),
                        RandAdjustContrastd(keys="image", prob=.15, gamma=(0.65, 1.5)),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                        RandGaussianNoised(keys="image", prob=.15, mean= 0, std = 0.1),
                        RandGaussianSharpend(keys="image", prob=.2, sigma1_x=(0.5, 1.5), sigma1_y=(0.5, 1.5), sigma1_z=(0.5, 1.5)),
                    ]
                )
                
            else:
                train_transform = Compose(
                    [
                            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=[128,128,128],),   
            Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
            SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
            #RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            #RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            #RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

                    ]
                )

                val_transform = Compose(
                    [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys="image"),
                    EnsureTyped(keys=["image", "label"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                        ),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=[128,128,128],),
                    Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
                    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),

                    ]
                )
    
        return train_transform, val_transform

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
