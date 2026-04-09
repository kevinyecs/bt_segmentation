import os
import torch
from datetime import datetime
from dataclasses import dataclass, field

cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')


@dataclass
class Config:
    arch: str = "SegResNet"
    max_epochs: int = 42
    batch_size: int = 1
    lr: float = 1e-4
    scheduler: str = None
    wd: float = 0.0
    backbone: str = None
    optim: str = "ranger"
    loss: str = "dice"
    swa: bool = False
    grad_acc: int = 4
    task: str = "Task01_BrainTumour"
    val_interval: int = 3
    momentum: float = 0.9
    resume: bool = False

    # Derived / computed fields
    device: str = field(init=False)
    data_dir: str = field(init=False)
    exp_name: str = field(init=False)
    seed: int = field(default=42, init=False)
    min_lr: float = field(default=1e-6, init=False)
    num_classes: int = field(default=3, init=False)
    n_fold: int = field(default=5, init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dir = os.environ.get("DATA_DIR", "/notebooks/shared/data")
        self.exp_name = (
            f"{cur_time}_{self.arch}_{self.max_epochs}_{self.batch_size}"
            f"_{self.lr}_{self.wd}_{self.backbone}_{self.optim}"
            f"_{self.loss}_TASK:{self.task}_RESUME:{self.resume}"
        )
        self.T_max = int(30000 / self.batch_size * self.max_epochs) + 50
        self.T_0 = 25
        self.n_accumulate = max(1, 64 / self.batch_size)
        self.max_lr = self.lr * 10
        self.step_size_up = 500
        self.step_size_down = 500
        self.n_iters = self.T_max
