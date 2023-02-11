import os
import random
import tempfile
from pathlib import Path
import typing as tp

import lifelines.utils
from loguru import logger as LOGGER
import hydra
import lifelines
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from pymongo import MongoClient
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import Parameter, TransformerEncoderLayer
from torch.optim import AdamW
from torch.utils.data import DataLoader

from surv_pred import models
from surv_pred.datasets import MultiModalityDataset


class MultiModalitiesDataModule(LightningDataModule):
    def __init__(self, batch_size: int, database_params: dict, modalities: tp.List[str],
                 fetch_all_modalities: bool = False, survival_resolution: int = 100, **kwargs):
        super().__init__()
        self.survival_resolution = survival_resolution
        self.fetch_all_modalities = fetch_all_modalities
        self.batch_size = batch_size
        self.database_params = database_params
        with MongoClient(database_params['db_params']['mongodb_connection_string']) as client:
            db = client[database_params['db_params']['db_name']]
            col = db['metadata']

            train_patients = col.find({'split': 'train'}).distinct('patient')

        self.modalities = modalities

        self.train_patients, self.val_patients = train_test_split(train_patients, test_size=0.1)

        self.test_patients = list(set(col.find({'split': 'test'}).distinct('patient')))

    @staticmethod
    def get_patients_in_modality(modality: str, **kwargs):
        df = pd.read_csv(Path(__file__).parent.joinpath(f'{modality}.csv')).set_index('patient').dropna()
        return df.index.tolist()

    def setup(self, stage: tp.Optional[str] = None) -> None:
        self.train_dataset = MultiModalityDataset(patients=self.train_patients,
                                                  modalities=self.modalities,
                                                  **self.database_params,
                                                  fetch_all_modalities=self.fetch_all_modalities,
                                                  survival_resolution=self.survival_resolution)
        self.val_dataset = MultiModalityDataset(patients=self.val_patients,
                                                modalities=self.modalities,
                                                **self.database_params, fetch_all_modalities=self.fetch_all_modalities,
                                                labtrans={modality: self.train_dataset.datasets[modality].labtrans for
                                                          modality in self.modalities},
                                                scaler={modality: self.train_dataset.datasets[modality].scaler for
                                                        modality in self.modalities},
                                                survival_resolution=self.survival_resolution
                                                )
        self.test_dataset = MultiModalityDataset(patients=self.test_patients,
                                                 modalities=self.modalities,
                                                 **self.database_params, fetch_all_modalities=self.fetch_all_modalities,
                                                 labtrans={modality: self.train_dataset.datasets[modality].labtrans for
                                                           modality in self.modalities},
                                                 scaler={modality: self.train_dataset.datasets[modality].scaler for
                                                         modality in self.modalities},
                                                 survival_resolution=self.survival_resolution
                                                 )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.train_dataset.collate_fn,
            # multiprocessing_context='fork'
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=0,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=0
        )

    @property
    def patients(self):
        return dict(train=self.train_patients, val=self.val_patients, test=self.test_patients)


class MultiModalitiesModel(LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters(params)

        self.nets = torch.nn.ModuleDict({modality: getattr(models, self.hparams.model_configs[modality].name)(
            **self.hparams.model_configs[modality].net_params) for
            modality in self.hparams.modalities})

        self.survival_threshold = Parameter(torch.tensor(0.5), requires_grad=False)

        self.deep_sets_phi = getattr(models, self.hparams.deep_sets_phi['name'])(**self.hparams.deep_sets_phi['params'])

        self.surv_head = getattr(models, self.hparams.surv_head['name'])(**self.hparams.surv_head['params'])

        self.modality_embedding = nn.Embedding(num_embeddings=len(self.hparams.modalities), embedding_dim=32)

        # self.transformer_encoder_layer = TransformerEncoderLayer(d_model=32, )

    def prepare_data(self) -> None:
        return

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        return self.step(batch, purpose='train')

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='val')

    def test_step(self, batch, batch_idx, *args, **kwargs) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='test')

    def step(self, batch, purpose, *args, **kwargs):
        surv_outputs, interm_out, brier_score, concordance_index, loss, recon_loss, c_loss = [{}, {}, {}, {}, {}, {},
                                                                                              {}]
        log_dict = {}
        patients_interm_outputs = {}
        patients_surv_fn = {}
        patients_events = {}
        patients_durations = {}
        patients_event_index = {}
        for modality in batch.keys():
            try:
                features, durations, events, surv_fns, event_indices, patients = batch[modality]['features'], \
                                                                                 batch[modality]['duration'], \
                                                                                 batch[modality]['event'], \
                                                                                 batch[modality]['surv_fn'], \
                                                                                 batch[modality]['event_index'], \
                                                                                 batch[modality]['patient']
            except KeyError:
                continue
            if modality == 'DNAm' and purpose == 'train':
                self.log(name=f'{purpose}/DNAm_#_patients', value=len(features))

            if len(features) == 1:
                with torch.no_grad():
                    self.nets[modality] = self.nets[modality].eval()
                    surv_outputs, interm_out = self.nets[modality](features)
                    self.nets[modality] = self.nets[modality].train()
            else:
                surv_outputs, interm_out = self.nets[modality](features)

            for _interm_out, patient in zip(interm_out[-2], patients):
                patients_interm_outputs[patient] = {modality: _interm_out,
                                                    **patients_interm_outputs[patient]} if patients_interm_outputs.get(
                    patient) else {modality: _interm_out}

            loss[modality] = []
            brier_score[modality] = []
            for patient, surv_out, surv_fn, event, event_ind, duration in zip(patients, surv_outputs, surv_fns, events,
                                                                              event_indices, durations):
                if event == 0:
                    surv_fn = surv_fn[:event_ind]
                    surv_out = surv_out[:event_ind]
                loss[modality].append(getattr(nn.functional, self.hparams.loss_fn)(surv_out, surv_fn))
                brier_score[modality].append(torch.mean((surv_fn - surv_out) ** 2))

                patients_surv_fn[patient] = surv_fn
                patients_events[patient] = event
                patients_durations[patient] = duration
                patients_event_index[patient] = event_ind

            loss[modality] = sum(loss[modality]) / len(loss[modality])
            brier_score[modality] = torch.mean(torch.tensor(brier_score[modality]))
            self.log(f'{purpose}/{modality}_brier_score', brier_score[modality],
                     batch_size=len(features))
            try:
                concordance_index[modality] = lifelines.utils.concordance_index(durations.cpu().numpy(),
                                                                                torch.tensor(
                                                                                    surv_outputs >= self.survival_threshold).sum(
                                                                                    dim=1).detach().cpu().numpy(),
                                                                                event_observed=np.array(events)
                                                                                )
                self.log(f'{purpose}/{modality}_concordance_index', concordance_index[modality],
                         batch_size=len(features))

            except ZeroDivisionError as zd:
                pass

            recon_loss[modality] = self.compute_reconstruction_loss(
                reconstruction_loss_params=self.hparams.reconstruction_loss_params,
                prediction=interm_out[-1],
                target=features
            )
            if recon_loss[modality] is not None:
                loss[modality] += recon_loss[modality]
                self.log(f'{purpose}/{modality}_reconstruction_loss', recon_loss, batch_size=len(features))
            if len(features) == 1 or modality == 'Clinical':
                c_loss[modality] = 0.
            else:
                c_loss[modality] = self.compute_contrastive_loss_within_modality(interm_out=interm_out,
                                                                                 features=features,
                                                                                 modality=modality,
                                                                                 **self.hparams.contrastive_loss_params)

            if c_loss[modality] is not None:
                self.log(f'{purpose}/{modality}_c_loss', c_loss[modality], batch_size=len(features))
                loss[modality] += c_loss[modality]
            self.log(f'{purpose}/loss', loss[modality], batch_size=len(features))

        multi_modality_contrastive_loss = self.compute_contrastive_loss_multi_modality(
            patients_features_interm_outputs=patients_interm_outputs
        )

        self.log(f'{purpose}/multi_modality_contrastive_loss', multi_modality_contrastive_loss)
        self.log(f'{purpose}/concordance_index', torch.mean(torch.tensor(list(concordance_index.values()))))
        self.log(f'{purpose}/brier_score', torch.mean(torch.tensor(list(brier_score.values()))))

        total_loss = sum([loss[modality] for modality in loss])
        total_loss += multi_modality_contrastive_loss

        if self.hparams.use_surv_head:
            surv_head_out_dict = self.compute_surv_head_loss_concordance(
                patients_interm_outputs=patients_interm_outputs,
                patients_surv_fns=patients_surv_fn,
                patients_events=patients_events,
                patients_durations=patients_durations,
                patients_events_index=patients_event_index

            )

            self.log(f'{purpose}/surv_head_loss', surv_head_out_dict['surv_head_loss'])
            self.log(f'{purpose}/surv_head_concordance', surv_head_out_dict['surv_head_concordance'])
            self.log_dict(
                {
                    f'{purpose}/{key}': value for key, value in
                    surv_head_out_dict['surv_head_modality_concordance'].items()
                }
            )
            total_loss += surv_head_out_dict['surv_head_loss']

        self.log(f'{purpose}/total_loss', total_loss)

        return {'loss': total_loss,
                **log_dict,
                'surv_outputs': surv_outputs.detach(),
                }

    def surv_head_inference_single_modality_based(self,
                                                  patients_interm_outputs,
                                                  patients_surv_fns,
                                                  patients_durations, patients_events,
                                                  patients_events_index
                                                  ):
        surv_fns = []
        durations = []
        events = []
        event_indices = []
        loss = []
        brier_score = []
        modalities = []
        for patient in patients_interm_outputs.keys():
            surv_fns.extend(len(patients_interm_outputs[patient]) * [patients_surv_fns[patient]])
            durations.extend(len(patients_interm_outputs[patient]) * [patients_durations[patient]])
            events.extend(len(patients_interm_outputs[patient]) * [patients_events[patient]])
            event_indices.extend(len(patients_interm_outputs[patient]) * [patients_events_index[patient]])
            modalities.extend(list(patients_interm_outputs[patient].keys()))
        if self.hparams.use_modalities_embeddings:
            embeddings = []
            for item in patients_interm_outputs.values():
                _embs = torch.stack(list(item.values()))
                emb_pos = [self.hparams.modalities.index(key) for key in item.keys()]
                modalities_embs = self.modality_embedding(torch.tensor(emb_pos, device=self.device))
                _embs += modalities_embs
                embeddings.append(_embs)
            embeddings_stacked = torch.cat(embeddings)
        else:
            embeddings_stacked = torch.cat(
                list([torch.stack(list(item.values())) for item in patients_interm_outputs.values()]))
        if self.hparams.use_separate_optimizer_for_surv_head:
            embeddings_stacked = embeddings_stacked.detach()
        surv_outputs = self.surv_head(embeddings_stacked)[0]
        for patient, surv_out, surv_fn, event, event_ind, duration in zip(patients_interm_outputs.keys(), surv_outputs,
                                                                          surv_fns, events,
                                                                          event_indices, durations):
            if event == 0:
                surv_fn = surv_fn[:event_ind]
                surv_out = surv_out[:event_ind]
            loss.append(getattr(nn.functional, self.hparams.loss_fn)(surv_out, surv_fn))
            brier_score.append(torch.mean((surv_fn - surv_out) ** 2))
        concordance = {}
        for modality in self.hparams.modalities:
            indices = np.stack(modalities) == modality
            modality_durations = list(
                torch.tensor(durations).cpu().numpy()[indices].astype(float))
            modality_surv_outputs = surv_outputs[indices]
            modality_events = np.array(events)[indices]

            concordance[f'surv_head_{modality}_concordance_index'] = lifelines.utils.concordance_index(
                modality_durations,
                torch.tensor(
                    modality_surv_outputs >= self.survival_threshold).sum(
                    dim=1).detach().cpu().numpy(),
                event_observed=modality_events
            )
        surv_head_concordance = lifelines.utils.concordance_index(torch.tensor(durations).cpu().numpy(),
                                                                  torch.tensor(
                                                                      surv_outputs >= self.survival_threshold).sum(
                                                                      dim=1).detach().cpu().numpy(),
                                                                  event_observed=np.array(events)
                                                                  )
        surv_head_loss = sum(loss) / len(loss)
        brier_score = torch.mean(torch.tensor(brier_score))
        return dict(surv_head_loss=surv_head_loss,
                    surv_head_concordance=surv_head_concordance,
                    durations=durations,
                    brier_score=brier_score,
                    events=events,
                    surv_head_modality_concordance=concordance)

    def surv_head_deep_sets_embedding_averaging(self,
                                                patients_interm_outputs,
                                                patients_surv_fns,
                                                patients_durations, patients_events,
                                                patients_events_index
                                                ):
        surv_fns = []
        durations = []
        events = []
        event_indices = []
        loss = []
        brier_score = []
        modalities = []
        embeddings = []
        for patient in patients_interm_outputs.keys():
            surv_fns.append(patients_surv_fns[patient])
            durations.append(patients_durations[patient])
            events.append(patients_events[patient])
            event_indices.append(patients_events_index[patient])

            _embs = torch.mean(torch.stack(list(patients_interm_outputs[patient].values())), axis=0, keepdim=True)
            if self.hparams.use_modalities_embeddings:
                emb_pos = [self.hparams.modalities.index(item) for item in
                           list(patients_interm_outputs[patient].keys())]
                modalities_embs = self.modality_embedding(torch.tensor(emb_pos, device=self.device))
                _embs += torch.mean(modalities_embs, keepdim=True, dim=0)
                embeddings.append(_embs)

        embeddings_stacked = torch.cat(embeddings)

        if self.hparams.use_separate_optimizer_for_surv_head:
            embeddings_stacked = embeddings_stacked.detach()

        embeddings_after_phi = self.deep_sets_phi(embeddings_stacked)

        surv_outputs = self.surv_head(embeddings_after_phi)[0]
        for patient, surv_out, surv_fn, event, event_ind, duration in zip(patients_interm_outputs.keys(), surv_outputs,
                                                                          surv_fns, events,
                                                                          event_indices, durations):
            if event == 0:
                surv_fn = surv_fn[:event_ind]
                surv_out = surv_out[:event_ind]
            loss.append(getattr(nn.functional, self.hparams.loss_fn)(surv_out, surv_fn))
            brier_score.append(torch.mean((surv_fn - surv_out) ** 2))
        concordance = {}
        surv_head_concordance = lifelines.utils.concordance_index(torch.tensor(durations).cpu().numpy(),
                                                                  torch.tensor(
                                                                      surv_outputs >= self.survival_threshold).sum(
                                                                      dim=1).detach().cpu().numpy(),
                                                                  event_observed=np.array(events)
                                                                  )
        surv_head_loss = sum(loss) / len(loss)
        brier_score = torch.mean(torch.tensor(brier_score))
        return dict(surv_head_loss=surv_head_loss,
                    surv_head_concordance=surv_head_concordance,
                    durations=durations,
                    brier_score=brier_score,
                    events=events,
                    surv_head_modality_concordance={})

    def surv_head_transformer_with_deep_sets_embedding_averaging(self,
                                                                 patients_interm_outputs,
                                                                 patients_surv_fns,
                                                                 patients_durations, patients_events,
                                                                 patients_events_index
                                                                 ):
        surv_fns = []
        durations = []
        events = []
        event_indices = []
        loss = []
        brier_score = []

        embeddings = torch.empty((0, len(self.hparams.modalities), 32), device=self.device)
        src_key_padding_mask = torch.empty((0, len(self.hparams.modalities), 32), device=self.device)
        for patient in patients_interm_outputs.keys():
            surv_fns.append(patients_surv_fns[patient])
            durations.append(patients_durations[patient])
            events.append(patients_events[patient])
            event_indices.append(patients_events_index[patient])

            embs = torch.zeros((1, len(self.hparams.modalities), 32), device=self.device)

            inds = []
            p_mask = torch.ones((1, len(self.hparams.modalities), 32), device=src_key_padding_mask.device,
                                dtype=self.dtype)
            for modality, _emb in patients_interm_outputs[patient].items():
                ind = self.hparams.modalities.index(modality)
                inds.append(ind)
                if self.hparams.use_modalities_embeddings:
                    emb_pos = self.hparams.modalities.index(modality)

                    modality_emb = self.modality_embedding(torch.tensor(emb_pos, device=self.device))
                    _emb = _emb + modality_emb
                embs[:, ind] = _emb

            p_mask[:, inds] = 0
            src_key_padding_mask = torch.cat([src_key_padding_mask, p_mask])

            embeddings = torch.cat([embeddings, embs])

        if self.hparams.use_separate_optimizer_for_surv_head:
            embeddings = embeddings.detach()

        embeddings_post_transformer = torch.mean(self.deep_sets_phi(embeddings), axis=1)
        surv_outputs = self.surv_head(embeddings_post_transformer)[0]
        for patient, surv_out, surv_fn, event, event_ind, duration in zip(patients_interm_outputs.keys(), surv_outputs,
                                                                          surv_fns, events,
                                                                          event_indices, durations):
            if event == 0:
                surv_fn = surv_fn[:event_ind]
                surv_out = surv_out[:event_ind]
            loss.append(getattr(nn.functional, self.hparams.loss_fn)(surv_out, surv_fn))
            brier_score.append(torch.mean((surv_fn - surv_out) ** 2))
        concordance = {}
        surv_head_concordance = lifelines.utils.concordance_index(torch.tensor(durations).cpu().numpy(),
                                                                  torch.tensor(
                                                                      surv_outputs >= self.survival_threshold).sum(
                                                                      dim=1).detach().cpu().numpy(),
                                                                  event_observed=np.array(events)
                                                                  )
        surv_head_loss = sum(loss) / len(loss)
        brier_score = torch.mean(torch.tensor(brier_score))
        return dict(surv_head_loss=surv_head_loss,
                    surv_head_concordance=surv_head_concordance,
                    durations=durations,
                    brier_score=brier_score,
                    events=events,
                    surv_head_modality_concordance={})

    def compute_surv_head_loss_concordance(
            self,
            patients_interm_outputs: dict,
            patients_surv_fns: tp.Dict[str, torch.Tensor],
            patients_durations: tp.Dict[str, int],
            patients_events: tp.Dict[str, int],
            patients_events_index: tp.Dict[str, int],
    ) -> dict:
        """
        This function computes the survival head loss and concordance index for a given set of patients. The input
        includes a dictionary of patients' intermediate outputs, survival functions, durations, events,
        and event indices.

        The function first initializes empty lists for the survival functions, durations, events, event indices,
        loss, and Brier score. For each patient in the input dictionary, it extends the survival functions,
        durations, events, and event indices lists by the appropriate number of copies of the patient's corresponding
        values.

        It then concatenates the intermediate outputs for each patient and pass it through a survival head (
        self.surv_head) to get the survival outputs. It then calculates the loss for each patient using the specified
        loss function (self.hparams.loss_fn) by comparing the survival outputs with the survival functions. It also
        calculates the Brier score for each patient.

        Finally, it uses the lifelines.utils.concordance_index function to calculate the concordance index for all
        the patients, using the durations, survival outputs, and events as input. The function returns a dictionary
        containing the survival head loss, concordance index, durations, Brier score, and events.

        """
        if self.hparams.latent_space_modality_concatenation_method == 'single_modality':
            return self.surv_head_inference_single_modality_based(patients_interm_outputs=patients_interm_outputs,
                                                                  patients_surv_fns=patients_surv_fns,
                                                                  patients_events=patients_events,
                                                                  patients_events_index=patients_events_index,
                                                                  patients_durations=patients_durations)
        elif self.hparams.latent_space_modality_concatenation_method == 'deep_sets_embedding_averaging':
            return self.surv_head_deep_sets_embedding_averaging(patients_interm_outputs=patients_interm_outputs,
                                                                patients_surv_fns=patients_surv_fns,
                                                                patients_events=patients_events,
                                                                patients_events_index=patients_events_index,
                                                                patients_durations=patients_durations)
        elif self.hparams.latent_space_modality_concatenation_method == 'deep_sets_embedding_transformer_averaging':
            return self.surv_head_transformer_with_deep_sets_embedding_averaging(
                patients_interm_outputs=patients_interm_outputs,
                patients_surv_fns=patients_surv_fns,
                patients_events=patients_events,
                patients_events_index=patients_events_index,
                patients_durations=patients_durations
            )

    def compute_contrastive_loss_multi_modality(self, patients_features_interm_outputs: tp.Dict[
        str, tp.Dict[str, torch.Tensor]]):
        """
        This function computes the contrastive loss for multi-modality data. The input is a dictionary of patient
        features, where the keys are patient IDs and the values are dictionaries with keys as modality names and
        values as the intermediate outputs for that modality. The function first initializes empty lists for the
        embeddings and targets.

        For each patient in the input dictionary, the function selects a random patient as the negative sample. For
        the positive sample, it selects the first two modalities from the current patient's modality list,
        and appends the intermediate outputs for those modalities to the embeddings lists (embs_1 and embs_2) and
        appends a target of 1 to the target list. For the negative sample, it selects the first modality from the
        negative patient's modality list, and appends the intermediate output for that modality to the embeddings
        lists and appends a target of 0 to the target list. If the negative patient has more than one modality and
        the current patient also has more than one modality, it will also append the second modality from each
        patient's modality list to the embeddings list and a target of 0 to the target list.

        Finally, the function stacks the embeddings and targets along the 0th dimension and calculates the cosine
        embedding loss between the embeddings using a margin of 0.5. The function returns the calculated loss.
        """
        embs_1, embs_2, target = [], [], []
        for patient in patients_features_interm_outputs.keys():
            patient_modalities = list(patients_features_interm_outputs[patient].keys())
            for modality in list(patients_features_interm_outputs[patient].keys()):
                negative_patient = random.choice(list(set(list(patients_features_interm_outputs.keys())) - {patient}))
                remaining_modalities = set(patient_modalities) - {modality}
                if remaining_modalities:
                    selected_modality = random.choice(list(remaining_modalities))
                    embs_1.append(patients_features_interm_outputs[patient][modality])
                    embs_2.append(patients_features_interm_outputs[patient][selected_modality])
                    target.append(torch.tensor(1, dtype=torch.long, device=embs_1[-1].device))

                negative_modalities = list(patients_features_interm_outputs[negative_patient].keys())
                embs_1.append(patients_features_interm_outputs[patient][modality])
                embs_2.append(patients_features_interm_outputs[negative_patient][negative_modalities[0]])
                target.append(torch.tensor(0, dtype=torch.long, device=embs_1[-1].device))

        embs_1 = torch.stack(embs_1, dim=0)
        embs_2 = torch.stack(embs_2, dim=0)
        target = torch.stack(target)

        return nn.CosineEmbeddingLoss(margin=0.5)(embs_1, embs_2, target)

    def compute_contrastive_loss_within_modality(self, features, interm_out, use: bool, dropout_rate: float,
                                                 modality: str):
        if use:
            repr_1 = interm_out[-2]
            repr_2 = self.nets[modality](nn.functional.dropout(features, dropout_rate))[1][-2]
            embs_1, embs_2, target = [], [], []
            for i in range(len(repr_1)):
                # pos
                embs_1.append(repr_1[i])
                embs_2.append(repr_2[i])
                target.append(torch.tensor(1, dtype=torch.long, device=repr_1.device))
                # neg
                embs_1.append(repr_1[1])
                embs_2.append(repr_2[random.choice(list(set(range(len(repr_1))) - {i}))])
                target.append(torch.tensor(0, dtype=torch.long, device=repr_1.device))
            embs_1 = torch.stack(embs_1, dim=0)
            embs_2 = torch.stack(embs_2, dim=0)
            target = torch.stack(target)

            return nn.CosineEmbeddingLoss(margin=0.5)(embs_1, embs_2, target)
        return

    @staticmethod
    def compute_reconstruction_loss(reconstruction_loss_params, prediction, target):
        if not reconstruction_loss_params['use']:
            return
        return getattr(nn.functional, reconstruction_loss_params['method'])(prediction, target)

    def configure_optimizers(self):
        if self.hparams.use_separate_optimizer_for_surv_head:
            return [
                AdamW(params=self.nets.parameters(recurse=True), lr=1e-4, weight_decay=self.hparams.weight_decay),
                AdamW(params=list(self.surv_head.parameters()) + list(self.modality_embedding.parameters()) + list(
                    self.deep_sets_phi.parameters()), lr=5e-3, weight_decay=self.hparams.weight_decay)
            ]
        else:
            AdamW(params=list(self.nets.parameters(recurse=True)) + list(self.surv_head.parameters()) + list(
                self.modality_embedding.parameters()), lr=1e-4, weight_decay=self.hparams.weight_decay)

    def forward(self, x) -> tp.Any:
        return self.net(x)


@hydra.main(version_base=None, config_path='config/multi_modality_config', config_name='config')
def main(config: DictConfig):
    dm = MultiModalitiesDataModule(database_params=config.db, batch_size=config.batch_size,
                                   modalities=config.modalities, fetch_all_modalities=config.fetch_all_modalities,
                                   survival_resolution=config.survival_resolution)

    print(OmegaConf.to_yaml(config))

    config.train_patients = dm.train_patients
    config.val_patients = dm.val_patients
    config.test_patients = dm.test_patients

    model = MultiModalitiesModel(params=dict(config)
                                 )

    if config.debug:
        ml_logger = TensorBoardLogger(save_dir=tempfile.gettempdir(),
                                      name=f'MultiModality={"_".join(config.modalities)}'
                                      )

    else:
        ml_logger = WandbLogger(project=config.project,
                                name=f'MultiModality',
                                log_model=config.log_model
                                )
        # ml_logger.watch(model, log_graph=True)
    callbacks = [EarlyStopping(**config.early_stop_monitor)]

    trainer = Trainer(gpus=[config.gpu],
                      logger=ml_logger,
                      callbacks=callbacks,
                      limit_train_batches=0.2,
                      log_every_n_steps=20,
                      gradient_clip_val=config.trainer_params.gradient_clip_val

                      )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
