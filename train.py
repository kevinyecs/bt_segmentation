"""
PyTorch Lightning training module for brain tumor segmentation.

Replaces the notebook-based training loop with a proper CLI-driven trainer
that supports: mixed precision, gradient accumulation, W&B logging,
checkpoint callbacks, PEFT methods, and multi-GPU.

Usage:
    # Full fine-tuning
    python train.py --arch SegResNet --max_epochs 100 --batch_size 2

    # LoRA fine-tuning (resource-constrained)
    python train.py --arch SwinUNETR --peft lora --lora_rank 4 --batch_size 1

    # Frozen encoder
    python train.py --arch SegResNet --peft freeze_encoder --lr 1e-3

    # Resume from checkpoint
    python train.py --arch SegResNet --resume checkpoints/last.ckpt
"""

import os
import argparse
import logging
from typing import Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from models.ModelBuilder import build_model, get_optimizer, get_scheduler, MODEL_REGISTRY
from utils.configurator import Config
from utils.transforms import DatasetTransforms
from utils.losses import DiceFocalLoss, CompoundBraTSLoss
from utils.metrics import BraTSMetrics
from utils.postprocess import postprocess_brats
from utils.peft import apply_lora, freeze_encoder, apply_adapters, print_trainable_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BraTSSegModule(L.LightningModule):
    """Lightning module for BraTS brain tumor segmentation.

    Handles training, validation, loss computation, metric logging,
    and optional PEFT (LoRA / adapters / frozen encoder).
    """

    def __init__(self, cfg: Config, peft: Optional[str] = None, lora_rank: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Build model
        self.model = build_model(
            arch=cfg.arch,
            num_classes=cfg.num_classes,
            in_channels=4,
            out_channels=cfg.num_classes,
            device=torch.device("cpu"),  # Lightning handles device placement
        )

        # Apply PEFT if requested
        if peft == "lora":
            apply_lora(self.model, rank=lora_rank)
            print_trainable_params(self.model)
        elif peft == "freeze_encoder":
            freeze_encoder(self.model)
            print_trainable_params(self.model)
        elif peft == "adapters":
            apply_adapters(self.model, bottleneck_dim=16)
            print_trainable_params(self.model)

        # Loss
        self.criterion = DiceFocalLoss(dice_weight=1.0, focal_weight=1.0)

        # Validation metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.post_pred = AsDiscrete(threshold=0.5)
        self.post_label = AsDiscrete(threshold=0.5)
        self.brats_metrics = BraTSMetrics()

        self.best_mean_dice = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = self.criterion(outputs, labels)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Compute Dice per class
        preds = [self.post_pred(torch.sigmoid(o)) for o in decollate_batch(outputs)]
        targets = [self.post_label(t) for t in decollate_batch(labels)]
        self.dice_metric(preds, targets)

    def on_validation_epoch_end(self):
        dice_values = self.dice_metric.aggregate()
        self.dice_metric.reset()

        class_names = ["TC", "WT", "ET"]
        mean_dice = dice_values.mean().item()

        for i, name in enumerate(class_names):
            if i < len(dice_values):
                self.log(f"val/dice_{name}", dice_values[i].item(), prog_bar=False)

        self.log("val/mean_dice", mean_dice, prog_bar=True)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice

        self.log("val/best_mean_dice", self.best_mean_dice, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model, self.cfg.optim, self.cfg)
        config = {"optimizer": optimizer}

        if self.cfg.scheduler is not None:
            scheduler = get_scheduler(optimizer, self.cfg)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/mean_dice",
            }

        return config


def get_data_loaders(cfg: Config, tr_type: Optional[str] = None):
    """Build train and validation data loaders from the BraTS dataset."""
    from monai.data import load_decathlon_datalist

    data_dir = cfg.data_dir
    dataset_json = os.path.join(data_dir, cfg.task, "dataset.json")

    # Load file lists from the Decathlon dataset.json
    train_files = load_decathlon_datalist(dataset_json, is_segmentation=True, data_list_key="training")
    val_files = load_decathlon_datalist(dataset_json, is_segmentation=True, data_list_key="validation")

    # If no explicit validation split, use last 20% of training
    if not val_files:
        split = int(0.8 * len(train_files))
        val_files = train_files[split:]
        train_files = train_files[:split]

    # Prefix data_dir to paths
    for f in train_files + val_files:
        for key in ["image", "label"]:
            if not os.path.isabs(f[key]):
                f[key] = os.path.join(data_dir, cfg.task, f[key])

    transforms = DatasetTransforms(task=cfg.task)
    train_transform, val_transform = transforms.get_transforms(tr_type=tr_type)

    train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=0.5)
    val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=1.0)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="BraTS Brain Tumor Segmentation Training")
    parser.add_argument("--arch", type=str, default="SegResNet", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adam", "sgd", "ranger", "adamw"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "cycle", "poly", "onecycle"])
    parser.add_argument("--loss", type=str, default="dice_focal", choices=["dice", "dice_focal", "dice_ce"])
    parser.add_argument("--grad_acc", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--val_interval", type=int, default=3)

    # PEFT options
    parser.add_argument("--peft", type=str, default=None, choices=["lora", "freeze_encoder", "adapters"])
    parser.add_argument("--lora_rank", type=int, default=4)

    # Data
    parser.add_argument("--data_dir", type=str, default=None, help="Override DATA_DIR env var")
    parser.add_argument("--transform_type", type=str, default=None, choices=["nnUnet", None])

    # Training
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="bt_segmentation")
    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    # Set data dir
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir

    # Build config
    cfg = Config(
        arch=args.arch,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optim=args.optim,
        scheduler=args.scheduler,
        loss=args.loss,
        grad_acc=args.grad_acc,
        val_interval=args.val_interval,
    )

    # W&B
    if args.wandb_key:
        import wandb
        wandb.login(key=args.wandb_key)

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.exp_name or cfg.exp_name,
        log_model=False,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{args.arch}_{{epoch}}_{{val/mean_dice:.4f}}",
            monitor="val/mean_dice",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val/mean_dice", patience=20, mode="max"),
        RichProgressBar(),
    ]

    # Lightning model
    model = BraTSSegModule(cfg=cfg, peft=args.peft, lora_rank=args.lora_rank)

    # Data
    train_loader, val_loader = get_data_loaders(cfg, tr_type=args.transform_type)

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        accumulate_grad_batches=args.grad_acc,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    logger.info("Training complete. Best mean Dice: %.4f", model.best_mean_dice)


if __name__ == "__main__":
    main()
