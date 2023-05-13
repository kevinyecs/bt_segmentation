import torch
import torch.nn as nn
import datetime.datetime

#Config class for hyper params and model construction

cur_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")

class Config(nn.Module):
    def __init__(self , max_epochs=42, batch_size=1, lr=1e-4, \
                scheduler=None, wd=0.0, backbone=None, optim="ranger", arch = "SegResNet", loss="dice", swa=False, grad_acc=4, task='brain', val_interval=3,momentum=0.9, resume=False):
        self.exp_name       = cur_time + "_"+ str(arch)+ "_"+ str(max_epochs) + "_"+ str(batch_size) + \
                            "_"+ str(lr) + "_"+ str(wd) + "_"+ str(backbone) + "_"+ str(optim)  + "_"+ str(loss)+ "_TASK: " + str(task) +" _RESUME:"+str(resume)
        self.loss = loss
        self.optim          = optim
        self.max_epochs     = max_epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.scheduler      = scheduler
        self.wd             = wd
        self.backbone       = backbone
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed           = 42
        self.data_dir       = "/notebooks/shared/data"
        self.min_lr         = 1e-6
        self.T_max          = int(30000/batch_size* max_epochs)+50
        self.T_0            = 25
        self.n_accumulate   = max(1,64/batch_size)
        self.n_fold         = 5
        self.num_classes    = 3
        self.arch           = arch
        self.swa            =swa
        self.grad_acc       =grad_acc
        self.task           = 'Task01_BrainTumour'
        self.val_interval   = val_interval
        self.momentum       = momentum
        self.resume         = resume
