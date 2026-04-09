"""
K-Fold Cross-Validation for BraTS segmentation.

Every competitive BraTS solution uses 5-fold cross-validation:
  - Split training data into 5 folds
  - Train 5 models (each uses 4 folds for training, 1 for validation)
  - At inference, ensemble all 5 models for improved accuracy (+1-2% Dice)

Usage:
    from utils.cross_val import KFoldBraTS, train_kfold

    # Generate fold splits
    kfold = KFoldBraTS(data_list=train_files, n_folds=5, seed=42)
    for fold_idx, (train_files, val_files) in enumerate(kfold):
        train_model(train_files, val_files, fold_idx)

    # Or use the full pipeline
    train_kfold(cfg, n_folds=5)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class KFoldBraTS:
    """K-Fold cross-validation splitter for BraTS datasets.

    Produces deterministic, non-overlapping folds. Each fold is a
    (train_files, val_files) tuple.

    Args:
        data_list: List of data dicts (each with "image" and "label" keys).
        n_folds: Number of folds.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_list: List[Dict[str, str]],
        n_folds: int = 5,
        seed: int = 42,
    ):
        self.data_list = list(data_list)
        self.n_folds = n_folds
        self.seed = seed

        rng = np.random.RandomState(seed)
        indices = np.arange(len(self.data_list))
        rng.shuffle(indices)
        self.folds = np.array_split(indices, n_folds)

    def __iter__(self) -> Iterator[Tuple[List[dict], List[dict]]]:
        for fold_idx in range(self.n_folds):
            yield self.get_fold(fold_idx)

    def __len__(self) -> int:
        return self.n_folds

    def get_fold(self, fold_idx: int) -> Tuple[List[dict], List[dict]]:
        """Get train/val split for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_folds-1).

        Returns:
            (train_files, val_files) tuple.
        """
        val_indices = self.folds[fold_idx]
        train_indices = np.concatenate(
            [self.folds[i] for i in range(self.n_folds) if i != fold_idx]
        )

        train_files = [self.data_list[i] for i in train_indices]
        val_files = [self.data_list[i] for i in val_indices]

        logger.info(
            "Fold %d/%d: train=%d, val=%d",
            fold_idx + 1, self.n_folds, len(train_files), len(val_files),
        )
        return train_files, val_files

    def save_splits(self, output_dir: str) -> None:
        """Save fold splits to JSON for reproducibility.

        Args:
            output_dir: Directory to save fold_X.json files.
        """
        os.makedirs(output_dir, exist_ok=True)
        for fold_idx in range(self.n_folds):
            train_files, val_files = self.get_fold(fold_idx)
            split = {
                "fold": fold_idx,
                "n_folds": self.n_folds,
                "seed": self.seed,
                "train": train_files,
                "val": val_files,
            }
            path = os.path.join(output_dir, f"fold_{fold_idx}.json")
            with open(path, "w") as f:
                json.dump(split, f, indent=2)
            logger.info("Saved fold %d split to %s", fold_idx, path)

    @staticmethod
    def load_fold(path: str) -> Tuple[List[dict], List[dict]]:
        """Load a previously saved fold split.

        Args:
            path: Path to fold_X.json.

        Returns:
            (train_files, val_files) tuple.
        """
        with open(path) as f:
            split = json.load(f)
        return split["train"], split["val"]


def train_kfold(
    cfg,
    n_folds: int = 5,
    seed: int = 42,
    fold_subset: Optional[List[int]] = None,
) -> List[str]:
    """Train models across all folds using PyTorch Lightning.

    Args:
        cfg: Config dataclass with arch, lr, max_epochs, etc.
        n_folds: Number of folds.
        seed: Random seed.
        fold_subset: If provided, only train these fold indices (e.g., [0, 1]).
            Useful for distributing folds across multiple machines.

    Returns:
        List of best checkpoint paths (one per fold).
    """
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import WandbLogger
    from monai.data import DataLoader, CacheDataset, load_decathlon_datalist

    from train import BraTSSegModule
    from utils.transforms import DatasetTransforms

    data_dir = cfg.data_dir
    dataset_json = os.path.join(data_dir, cfg.task, "dataset.json")
    all_files = load_decathlon_datalist(dataset_json, is_segmentation=True, data_list_key="training")

    # Prefix data_dir to paths
    for f in all_files:
        for key in ["image", "label"]:
            if not os.path.isabs(f[key]):
                f[key] = os.path.join(data_dir, cfg.task, f[key])

    kfold = KFoldBraTS(all_files, n_folds=n_folds, seed=seed)

    # Save splits for reproducibility
    splits_dir = os.path.join("checkpoints", f"{cfg.arch}_kfold_splits")
    kfold.save_splits(splits_dir)

    transforms = DatasetTransforms(task=cfg.task)
    train_transform, val_transform = transforms.get_transforms()

    folds_to_train = fold_subset if fold_subset is not None else list(range(n_folds))
    best_checkpoints = []

    for fold_idx in folds_to_train:
        logger.info("=" * 60)
        logger.info("Training fold %d / %d", fold_idx + 1, n_folds)
        logger.info("=" * 60)

        train_files, val_files = kfold.get_fold(fold_idx)

        train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=0.5)
        val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=1.0)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        # Fresh model per fold
        model = BraTSSegModule(cfg=cfg)

        wandb_logger = WandbLogger(
            project="bt_segmentation",
            name=f"{cfg.arch}_fold{fold_idx}",
            group=f"{cfg.arch}_{n_folds}fold",
            log_model=False,
        )

        ckpt_dir = os.path.join("checkpoints", f"{cfg.arch}_fold{fold_idx}")
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"fold{fold_idx}_{{epoch}}_{{val/mean_dice:.4f}}",
                monitor="val/mean_dice",
                mode="max",
                save_top_k=1,
                save_last=True,
            ),
            EarlyStopping(monitor="val/mean_dice", patience=20, mode="max"),
        ]

        trainer = L.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            accumulate_grad_batches=cfg.grad_acc,
            check_val_every_n_epoch=cfg.val_interval,
            logger=wandb_logger,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
        )

        trainer.fit(model, train_loader, val_loader)
        best_checkpoints.append(callbacks[0].best_model_path)
        logger.info("Fold %d best checkpoint: %s", fold_idx, callbacks[0].best_model_path)

    logger.info("All folds complete. Best checkpoints:")
    for i, path in enumerate(best_checkpoints):
        logger.info("  Fold %d: %s", i, path)

    return best_checkpoints


def load_kfold_ensemble(
    cfg,
    checkpoint_paths: List[str],
    device: str = "cuda",
) -> List:
    """Load all fold models for ensemble inference.

    Args:
        cfg: Config (same architecture used for all folds).
        checkpoint_paths: Paths to per-fold checkpoints.
        device: Target device.

    Returns:
        List of loaded models, ready for `ensemble_predict()`.
    """
    from models.ModelBuilder import build_model
    import torch

    models = []
    for path in checkpoint_paths:
        model = build_model(cfg.arch, cfg.num_classes, device=torch.device(device))
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        # Lightning wraps state_dict with "model." prefix
        cleaned = {k.replace("model.", "", 1): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        model.eval()
        models.append(model)
        logger.info("Loaded fold model from %s", path)

    return models
