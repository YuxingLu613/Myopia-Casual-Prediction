from collections import defaultdict
from functools import reduce
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.nn import functional as F
from torchmetrics import *
from torchmetrics.utilities.data import dim_zero_cat
import torch
import torchmetrics.functional as MF
from ddp_Utils import *
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from ehrformer import EHRFormer

 
class EHRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.data_type = config.get('data_type', '')
        self.momentum = config.get('momentum', 0.9)
        self.wd = config.get('wd', 1e-6)
        self.lr = config.get('lr', 1e-3)
        self.n_nodes = config.get('n_nodes', 1)
        self.n_gpus = config.get('n_gpus', 1)
        if isinstance(self.n_gpus, list):
            self.n_gpus = len(self.n_gpus)
        self.n_epoch = config.get('n_epoch', None)
        self.reg_label_cols = config.get('reg_label_cols', [])
        self.model = torch.compile(EHRFormer(config))

        self.output_dir = config.get('output_dir', None)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_folder = 'pred'

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                  first_cycle_steps=self.n_epoch,
                                                  max_lr=self.lr, 
                                                  min_lr=1e-8, 
                                                  warmup_steps=int(self.n_epoch * 0.1))
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step()

    def on_train_start(self):
        self.loggers[0].log_hyperparams(self.config)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def training_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        valid_mask = data['valid_mask']
        reg_label = label['reg']['values']
        reg_mask = valid_mask.unsqueeze(1).expand_as(label['reg']['masks']) & label['reg']['masks']

        cat_feats = data['cat_feats']
        float_feats = data['float_feats']
        valid_mask = data['valid_mask']
        time_index = data['time_index']
        # Get medication_type from batch if present
        medication_type = data.get('medication_type', None)

        y_cls, adv_loss = self.model(cat_feats, float_feats, valid_mask, time_index, medication_type=medication_type, grl_lambda=1.0)

        n_reg = reg_label.shape[1]
        reg_loss = 0
        for i in range(n_reg):
            l = reg_label[:, i, :].reshape(-1)
            m = reg_mask[:, i, :].reshape(-1)
            y = y_cls[i].reshape(-1)
            reg_losses = F.mse_loss(y, l, reduction='none')
            reg_loss = reg_loss + (reg_losses * m).sum() / (m.sum().clip(1))

        # Add adversarial loss (with negative sign)
        alpha = 1.0  # You can tune this hyperparameter or set from config
        if adv_loss is not None:
            loss = reg_loss - alpha * adv_loss
        else:
            loss = reg_loss

        self.log("train_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.preds = defaultdict(list)
        self.targets = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        valid_mask = data['valid_mask']
        reg_label = label['reg']['values']
        reg_mask = valid_mask.unsqueeze(1).expand_as(label['reg']['masks']) & label['reg']['masks']

        cat_feats = data['cat_feats']
        float_feats = data['float_feats']
        valid_mask = data['valid_mask']
        time_index = data['time_index']
        y_cls = self.model(cat_feats, float_feats, valid_mask, time_index)

        n_reg = reg_label.shape[1]
        
        reg_loss = 0
        reg_labels = []
        reg_masks = []
        reg_preds = []
        for i in range(n_reg):
            l = reg_label[:, i, :].reshape(-1)
            m = reg_mask[:, i, :].reshape(-1)
            y = y_cls[i].reshape(-1)
            reg_preds.append(y)
            reg_labels.append(l)
            reg_masks.append(m)
        reg_labels = torch.cat(reg_labels, dim=0)
        reg_masks = torch.cat(reg_masks, dim=0)
        reg_preds = torch.cat(reg_preds, dim=0)
        reg_losses = F.mse_loss(reg_preds, reg_labels, reduction='none')
        reg_loss = reg_loss + (reg_losses * reg_masks).sum() / (reg_masks.sum().clip(1))

        tensor_to_gather = [
            reg_label.contiguous(), reg_mask.contiguous()
        ] + y_cls
        tensor_gathered = [x.cpu() for x in all_gather(tensor_to_gather)]
        reg_label = tensor_gathered[0]
        reg_mask = tensor_gathered[1]
        y_cls = tensor_gathered[2:]

        for i in range(len(self.reg_label_cols)):
            mask = reg_mask[:, i, :]
            pp = y_cls[i].squeeze(-1)
            pp = pp[mask]

            yy = reg_label[:, i, :]
            yy = yy[mask]

            if len(pp) > 0:
                self.preds[f'reg_{i}'].append(pp)
                self.targets[f'reg_{i}'].append(yy)
        self.log("val_loss", reg_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return reg_loss

    def on_validation_epoch_end(self) -> None:
        mpcc = []
        for i, k in enumerate(self.reg_label_cols):
            if len(self.preds[f'reg_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'reg_{i}']).double()
                target = dim_zero_cat(self.targets[f'reg_{i}']).double()
                self.log(f"val_{k}_mse", MF.mean_squared_error(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_pcc", MF.pearson_corrcoef(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_r2", MF.r2_score(pred, target), prog_bar=False, rank_zero_only=True)
                mpcc.append(MF.pearson_corrcoef(pred, target))
        mpcc = sum(mpcc) / len(mpcc) if len(mpcc) != 0 else 0
        self.log(f"val_mpcc", mpcc, prog_bar=False, rank_zero_only=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        data = batch['data']
        pid = batch['pid']
        label = batch['label']
        valid_mask = data['valid_mask']
        reg_label = label['reg']['values']

        valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, label['reg']['masks'].size(1), -1)
        reg_mask = valid_mask_expanded & label['reg']['masks']

        cat_feats = data['cat_feats']
        float_feats = data['float_feats']
        valid_mask = data['valid_mask']
        time_index = data['time_index']

        y_cls = self.model(cat_feats, float_feats, valid_mask, time_index)
        self.test_outputs.append({
            'preds': y_cls, 'pid': pid, 
            'reg_label': reg_label, 'reg_mask': reg_mask
        })

    def on_test_epoch_end(self):
        output_dir = Path(self.output_dir) / self.pred_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        n_reg = len(self.reg_label_cols)

        pred_reg = [[] for _ in range(n_reg)]
        reg_mask = [[] for _ in range(n_reg)]
        reg_label = [[] for _ in range(n_reg)]
        for i in range(len(self.test_outputs)):
            for j in range(n_reg):
                pred_reg[j].append(self.test_outputs[i]['preds'][j])
                reg_mask[j].append(self.test_outputs[i]['reg_mask'][:, j, :])
                reg_label[j].append(self.test_outputs[i]['reg_label'][:, j, :])
        pred_reg = [torch.cat(x, dim=0).to('cpu') for x in pred_reg]
        reg_mask = [torch.cat(x, dim=0).to('cpu') for x in reg_mask]
        reg_label = [torch.cat(x, dim=0).to('cpu') for x in reg_label]

        pid = reduce(lambda x, y: x + y, [x['pid'] for x in self.test_outputs])
        
        n_sample = len(pid)
        df = pd.DataFrame(pid, columns=['pid'])

        for i, col in enumerate(self.reg_label_cols):
            tmp1 = pd.DataFrame([{f"{col}_prob_0": pred_reg[i][j, :, 0].numpy()} for j in range(n_sample)])
            tmp2 = pd.DataFrame([{f"{col}_mask": reg_mask[i][j, :].numpy()} for j in range(n_sample)])
            tmp3 = pd.DataFrame([{col: reg_label[i][j, :].numpy()} for j in range(n_sample)])
            df = pd.concat([df, tmp1, tmp2, tmp3], axis=1)
        output_path = output_dir / f'test_pred.{self.global_rank}.{self.config["file_name"]}.parquet'
        df.to_parquet(output_path)
        self.test_outputs.clear()