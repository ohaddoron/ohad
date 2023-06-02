from typing import Any
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torchmetrics.functional import precision_recall, accuracy
import torch.nn.functional as F
from surv_pred.datasets import ModalitiesDataset
import pytorch_lightning as pl

from surv_pred.models import ClinicalNet


class ClassificationNet(nn.Module):
    def __init__(self, num_patients):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(1024),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(256),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.LazyLinear(num_patients),

        )

    def forward(self, x):
        return self.net.forward(x)


output_patients = dict(
    Clinical=10965,
    miRNA=9691,
    mRNA=9706,
    DNAm=10394
)


class ClinicalClassificationNet(ClinicalNet):
    def __init__(self, num_patients, *args, **kwargs):
        self.num_patients = num_patients
        super().__init__(net_params={})
        # self.embedding_layers = nn.ModuleList([nn.Embedding(36, 1024) for _ in range(9)])
        self.map_continuous = nn.LazyLinear(42)
        self.map_embeddings = nn.Sequential(nn.LazyLinear(42),
                                            )

        self.bn_layer = nn.BatchNorm1d(42)

    def _init_net(self, **kwargs) -> nn.Module:
        return ClassificationNet(num_patients=self.num_patients)

    def forward(self, x):
        continuous_x, categorical_x = x[:, 9:], x[:, :9]
        categorical_x = categorical_x.to(torch.int64)

        x = [emb_layer(categorical_x[:, i])
             for i, emb_layer in enumerate(self.embedding_layers)]
        x = self.map_embeddings(torch.cat(x, 1))
        # x = self.embedding_dropout(x)

        continuous_x = F.leaky_relu(self.bn_layer(self.map_continuous(continuous_x)))

        x = x + continuous_x

        # continuous_x, categorical_x = x[:, :1], x[:, 1:]
        # categorical_x = categorical_x.to(torch.int64)
        #
        # x = [emb_layer(categorical_x[:, i])
        #      for i, emb_layer in enumerate(self.embedding_layers)]
        # x = torch.cat(x, 1)
        #
        # continuous_x = self.bn_layer(continuous_x)
        # x = torch.cat([x, continuous_x], 1)

        out = self.output_layer(x)

        return out


class ClassificationDataset(ModalitiesDataset):
    def __init__(self, modality, patients, db_params):
        super().__init__(modality=modality, patients=patients, db_params=db_params)
        self.data = self.data.drop_duplicates()
        self._patients_random_mat = np.random.rand(len(self), 41).astype(np.float32)

    def __getitem__(self, index):
        return {'features': self.data.iloc[index].to_numpy().astype(np.float32), 'target': index}

    def __len__(self):
        return len(self.data)


class ClinicalClassificationModel(pl.LightningModule):
    def __init__(self, modality: str = 'Clinical'):
        super().__init__()
        self.save_hyperparameters()

        self.modality = modality
        if modality == 'Clinical':
            self.net = ClinicalClassificationNet(num_patients=output_patients[modality])
        else:
            self.net = ClassificationNet(num_patients=output_patients[modality])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=ClassificationDataset(modality=self.modality,
                                          patients=pd.read_csv(f'{self.modality}.csv').dropna().set_index(
                                              'patient').index.tolist(),
                                          db_params={
                                              'mongodb_connection_string': 'mongodb://admin:mimp1lab@132.66.207.18:80',
                                              'db_name': 'TCGAOmics'}),
            batch_size=32,
            num_workers=0,
            shuffle=True
        )

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return self.train_dataloader()

    def configure_optimizers(self) -> Any:
        optimizer = Adam(self.net.parameters(), lr=1e-4)
        return {'optimizer': optimizer, 'scheduler': ReduceLROnPlateau(optimizer, 'train/accuracy'),
                'monitor': 'train/accuracy'}

    def step(self, batch, purpose: str):
        logits = self.net(batch['features'])

        loss = CrossEntropyLoss()(logits, batch['target'])

        precision, recall = precision_recall(logits, batch['target'], num_classes=logits.shape[1])
        acc = accuracy(logits, batch['target'])

        self.log(f'{purpose}/loss', loss)
        self.log(f'{purpose}/precision', precision, prog_bar=True)
        self.log(f'{purpose}/recall', recall, prog_bar=True)
        self.log(f'{purpose}/accuracy', acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(purpose='train', batch=batch)


if __name__ == '__main__':
    trainer = Trainer(gpus=None, log_every_n_steps=1, max_epochs=-1)
    trainer.fit(ClinicalClassificationModel())
