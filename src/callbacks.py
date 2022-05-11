import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.train_loss_log = []
        self.val_loss_log = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics["train_loss"].cpu() 
        self.train_loss_log.append(train_loss)
        
    def on_validation_epoch_end(self,trainer, pl_module):
        val_loss = trainer.logged_metrics["val_loss"].cpu()
        self.val_loss_log.append(val_loss)
        
    