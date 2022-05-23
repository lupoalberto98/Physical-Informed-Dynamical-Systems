import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.train_loss_log = []
        self.val_loss_log = []
        self.params_loss_log= []
        self.train_reg_log = []
        self.val_reg_log = []
        
        
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics["train_loss"].cpu()
        self.train_loss_log.append(train_loss)
        if "params_loss" in trainer.logged_metrics.keys():
            params_loss = trainer.logged_metrics["params_loss"].cpu()
            self.params_loss_log.append(args_loss)
        if "train_reg_loss" in trainer.logged_metrics.keys():
            train_reg_loss =  trainer.logged_metrics["train_reg_loss"].cpu()
            self.train_reg_log.append(train_reg_loss)
        
    def on_validation_epoch_end(self,trainer, pl_module):
        val_loss = trainer.logged_metrics["val_loss"].cpu()
        if "val_reg_loss" in trainer.logged_metrics.keys():
            val_reg_loss =  trainer.logged_metrics["val_reg_loss"].cpu()
            self.val_reg_log.append(val_reg_loss)
            
        self.val_loss_log.append(val_loss)
        
    