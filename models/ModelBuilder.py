import torch
import torch.nn as nn
from ranger import Ranger

from monai.networks.nets import UNet, SwinUNETR, DynUNet, SegResNet,SegResNetVAE
from monai.networks.nets.attentionunet import AttentionUnet
from .convnext import ConvNeXt

from models import ResNet3dVAE, ResNet3d, nnUnet

def build_model(arch, num_classes,in_channels=4, out_channels=3, device=torch.device("cuda:0")):
    roi = (128, 128, 128) #Before Crop (224, 224, 144)
    device = device
    model = None
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    
    if arch == "SegResNet":
        model = SegResNet(
            spatial_dims = 3,
            init_filters= 32,
            in_channels=in_channels,
            out_channels = out_channels
        ).to(device)
    if arch == "SegNetVAE":
            model = SegResNetVAE(
            input_image_size=[128, 128, 128],
            init_filters= 32,
            in_channels=in_channels,
            out_channels = out_channels,

                
                

        ).to(device)
        
        
    if arch == "ConvNeXt":
        model = ConvNeXt(
        #dims=[32, 64, 128, 256, 320]
        dims=[96, 192, 384, 768]
        ).to(device)
    if arch == "nnUNeXt":
        model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=kernels,
            upsample_kernel_size=strides[1:],
            deep_supervision=False,
            deep_supr_num=2,
            act_name=("GELU"),
            res_block=True
            
        ).to(device)
        model.to(device)
   
    if arch == "nnUNet":
        model = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        kernel_size=kernels,
        upsample_kernel_size=strides[1:],


    ).to(device)
        
        

    if arch == "nUNet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=0,
            act="leakyrelu",
            norm=("layer", {"normalized_shape":(128)})
        ).to(device)
    
    if arch == "UNETR":
        model = SwinUNETR(
            img_size=(128,128,128), 
            in_channels=in_channels, 
            out_channels=out_channels, 
            depths=(2,4,2,2)
            ).to(device)

    if arch == "UNet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2,),
            act="RELU",
            

            
            
        ).to(device)
    #,

    if arch == "AttUnet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        ).to(device)


    if arch == "ResNetVAE":
        model = ResNet3dVAE(
            in_channels=in_channels,
            classes=num_classes,
            dim=[128,128,128]
        ).to(device)
        VAE = True

    if arch == "ResNet3d":
        model = ResNet3d(
            in_channels=in_channels,
            classes=num_classes,
            dim=[128,128,128]
        ).to(device)
        
    return model      

def get_optimizer(model, optim, cfg):
    optimizer = None
    if optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)
    if optim == "ranger":
        optimizer = Ranger(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        rangered = True
    if optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    return optimizer

def get_scheduler(optimizer, cfg):
    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=0, verbose=True)
    if cfg.scheduler == "cycle":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.lr, max_lr=cfg.max_lr, step_size_up=cfg.step_size_up, step_size_down=cfg.step_size_down, mode="triangular2", cycle_momentum=False, base_momentum=0.85, max_momentum=0.95)
    if cfg.scheduler == "poly":
        scheduler = PolynomialLR(optimizer, total_iters=cfg.n_iters, power=0.9, verbose=True)
    return scheduler

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> # xdoctest: +SKIP("undefined vars")
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, total_iters=19400, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]