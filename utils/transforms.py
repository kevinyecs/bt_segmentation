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
)


class DatasetTransforms(nn.Module):
    def __init__(self, task: str):
        self.task = task

    def get_transforms(self, tr_type=None):
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
                            pixdim=(1.0, 0.77, 0.77),
                            mode=("bilinear", "nearest"),
                        ),
                        CropForegroundd(
                            keys=["image", "label"],
                            source_key="image",
                            k_divisible=[128, 128, 128],
                        ),
                        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
                        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                        # Augmentation (train only)
                        RandRotated(keys=["image", "label"], prob=0.2, range_x=[0.3, 0.3]),
                        RandRotated(keys=["image", "label"], prob=0.2, range_y=[0.3, 0.3]),
                        RandRotated(keys=["image", "label"], prob=0.2, range_z=[0.3, 0.3]),
                        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.7, max_zoom=1.4),
                        RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.65, 1.5)),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                        RandGaussianNoised(keys="image", prob=0.15, mean=0, std=0.1),
                        RandGaussianSmoothd(keys="image", prob=0.2, sigma_x=(0.5, 1.5)),
                        RandGaussianSmoothd(keys="image", prob=0.2, sigma_y=(0.5, 1.5)),
                        RandGaussianSmoothd(keys="image", prob=0.2, sigma_z=(0.5, 1.5)),
                    ]
                )

                # Validation: deterministic only — no random augmentation
                val_transform = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys="image"),
                        EnsureTyped(keys=["image", "label"]),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                        Spacingd(
                            keys=["image", "label"],
                            pixdim=(1.0, 0.77, 0.77),
                            mode=("bilinear", "nearest"),
                        ),
                        CropForegroundd(
                            keys=["image", "label"],
                            source_key="image",
                            k_divisible=[128, 128, 128],
                        ),
                        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
                        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    ]
                )

            else:
                train_transform = Compose(
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
                        CropForegroundd(
                            keys=["image", "label"],
                            source_key="image",
                            k_divisible=[128, 128, 128],
                        ),
                        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
                        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                    ]
                )

                # Validation: deterministic only — no random augmentation
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
                            k_divisible=[128, 128, 128],
                        ),
                        Resized(keys=["image", "label"], spatial_size=(128, 128, 128), mode=("area", "nearest")),
                        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                    ]
                )

        return train_transform, val_transform


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on BraTS classes:
      label 1 = peritumoral edema
      label 2 = GD-enhancing tumor
      label 3 = necrotic and non-enhancing tumor core

    Output channels:
      TC (Tumor Core)      = label 2 | label 3
      WT (Whole Tumor)     = label 1 | label 2 | label 3
      ET (Enhancing Tumor) = label 2
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
