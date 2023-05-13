import matplotlib.pyplot as plt
import numpy as np
#import graphviz


class Visualizer():
    def __init__(self, in_ch, out_ch, org_data):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.data = org_data
        #self.tr_data = tr_data
        self.modalities = ['T1', 'T1ce', 'T2', 'FLAIR']

    def plot(self):
        print(f"image shape: {self.data['image'].shape}")
        plt.figure("image", (24, 6))
        for i in range(self.in_ch):
            plt.subplot(1, 4, i + 1)
            plt.title(f"image channel {self.modalities[i]}")
            #print(type(self.data["image"][i, :, :, self.data['image'].shape[-1]//2].detach().cpu()))
            #print(min(self.data["image"][i, :, :, self.data['image'].shape[-1]//2].detach().cpu()))
            #print(max(self.data["image"][i, :, :, self.data['image'].shape[-1]//2].detach().cpu()))
            plt.imshow(self.data["image"][i, :, :, self.data['image'].shape[-1]//2].detach().cpu(), cmap="gray") #vmin vmax
        plt.show()
        # also visualize the 3 channels label corresponding to this image
        print(f"label shape: {self.data['label'].shape}")
        plt.figure("label", (18, 6))
        for i in range(self.out_ch):
            plt.subplot(1, 3, i + 1)
            plt.title(f"label channel {i}")
            plt.imshow(self.data["label"][i, :, :, self.data['image'].shape[-1]//2].detach().cpu())
        plt.show()
        ''''
        plt.figure("diff",(18,6))
        for i in range(self.out_ch):
            plt.subplot(1, 3, i + 1)
            plt.title(f"label channel {i}")
            diff_img = self.data["image"]- self.tr_data["image"]
            plt.imshow(np.abs(diff_img[i, :, :, self.data['image'].shape[-1]//2].detach().cpu()))
        plt.show()
        '''
        
class TrainingResults():
    def __init__(self, epoch_loss_values, val_interval, metric_values,metric_values_tc,  metric_values_wt, metric_values_et):
        self.epoch_loss_values = epoch_loss_values
        self.val_interval = val_interval
        self.metric_values = metric_values
        self.metric_values_tc = metric_values_tc
        self.metric_values_wt = metric_values_wt
        self.metric_values_et = metric_values_et
        
    def plot(self):
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")

        x = [i + 1 for i in range(len(self.epoch_loss_values))]
        y = self.epoch_loss_values

        data = [[x, y] for (x, y) in zip(x, y)]
        table = wandb.Table(data=data, columns=["Epoch", "Loss"])
        wandb.log({"Epoch Average Loss": wandb.plot.line(table, "Epoch", "Loss", title="Epoch Average Loss")})

        plt.xlabel("epoch")
        plt.plot(x, y, color="red")
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")

        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values))]
        y = self.metric_values
        data = [[x, y] for (x, y) in zip(x, y)]
        table = wandb.Table(data=data, columns=["Epoch", "Dice"])
        wandb.log({"Val Mean Dice": wandb.plot.line(table, "Epoch", "Dice", title="Val Mean Dice")})

        plt.xlabel("epoch")
        plt.plot(x, y, color="green")
        plt.show()

        plt.figure("train", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Val Mean Dice TC")

        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values_tc))]
        y = self.metric_values_tc

        data = [[x, y] for (x, y) in zip(x, y)]
        table = wandb.Table(data=data, columns=["Epoch", "Dice"])
        wandb.log({"Val Mean Dice TC": wandb.plot.line(table, "Epoch", "Dice", title="Val Mean Dice TC")})

        plt.xlabel("epoch")
        plt.plot(x, y, color="blue")
        plt.subplot(1, 3, 2)
        plt.title("Val Mean Dice WT")

        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values_wt))]
        y = self.metric_values_wt

        data = [[x, y] for (x, y) in zip(x, y)]
        table = wandb.Table(data=data, columns=["Epoch", "Dice"])
        wandb.log({"Val Mean Dice WT": wandb.plot.line(table, "Epoch", "Dice", title="Val Mean Dice WT")})

        plt.xlabel("epoch")
        plt.plot(x, y, color="brown")
        plt.subplot(1, 3, 3)
        plt.title("Val Mean Dice ET")

        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values_et))]
        y = self.metric_values_et

        data = [[x, y] for (x, y) in zip(x, y)]
        table = wandb.Table(data=data, columns=["Epoch", "Dice"])
        wandb.log({"Val Mean Dice ET": wandb.plot.line(table, "Epoch", "Dice", title="Val Mean Dice ET")})

        plt.xlabel("epoch")
        plt.plot(x, y, color="purple")
        plt.show()
