from pytorch_lightning.callbacks import ModelCheckpoint
from typing import *
import pytorch_lightning as pl
import wandb
from src.logger import logger


class WANDBCheckpointCallback(ModelCheckpoint):
    def save_checkpoint(self, trainer: "pl.Trainer", unused: Optional["pl.LightningModule"] = None) -> None:
        if self.last_model_path:
            wandb.save(self.last_model_path, overwrite=True)
        else:
            logger.error('Model was not found')
