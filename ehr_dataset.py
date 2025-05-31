from pathlib import Path
import random
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pyarrow.parquet as pq
import diskcache


def save_parquet(df, path):
    tmp_df = df.copy()
    for col in tmp_df.columns:
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and len(tmp_df.iloc[0][col].shape) == 2:
            tmp_df[col] = tmp_df[col].map(lambda x: list(x))
    tmp_df.to_parquet(path)


def load_parquet(path):
    parquet_file = pq.ParquetFile(path)
    tmp_df = []
    for sub_df in parquet_file.iter_batches(batch_size=100000):
        tmp_df.append(sub_df.to_pandas())
    tmp_df = pd.concat(tmp_df, axis=0).reset_index(drop=True)
    for col in tmp_df.columns:
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and isinstance(tmp_df.iloc[0][col][0], np.ndarray):
            tmp_df[col] = tmp_df[col].map(lambda x: np.array(list(x)))
    return tmp_df


def gen_future_index(n, roll_prob=0.9):
    if random.randint(0, 10) < roll_prob * 10:
        result = np.roll(np.arange(n), -1)
    else:
        result = np.zeros(n, dtype=np.int32)
        for i in range(n):
            result[i] = random.randint(i+1, n)
        result[result == n] = 0
    return result


@dataclass
class EHRDataset(Dataset):
    data: pd.DataFrame
    mode: str
    config: dict

    def __post_init__(self):
        n_reg = []
        n_reg += [1 for _ in range(len(self.config['current_label']['reg_label_cols']))]

        self.config['n_reg'] = n_reg
        self.config['reg_label_cols'] = self.config['current_label']['reg_label_cols']

        self.seq_max_len = self.config['seq_max_len']
        self.roll_prob = self.config['roll_prob']

    def read_sample(self, idx):
        sample = {
            'cat_feats': torch.tensor(self.data.loc[self.data.index[idx], 'tokenized_category_feats'].astype(np.int64), dtype=torch.long),
            'float_feats': torch.tensor(self.data.loc[self.data.index[idx], 'tokenized_float_feats'].astype(np.int64), dtype=torch.long),
            'valid_mask': torch.tensor(self.data.loc[self.data.index[idx], 'valid_mask'].astype(bool), dtype=torch.bool),
            'time_index': torch.tensor(self.data.loc[self.data.index[idx], 'visit_index'].astype(np.int64), dtype=torch.long),
        }
        # Use config for medication type column
        med_col = self.config.get('medication_type_col', None)
        if med_col is not None and med_col in self.data.columns:
            sample['medication_type'] = torch.tensor(self.data.loc[self.data.index[idx], med_col], dtype=torch.long)
        sample['cat_feats'] = sample['cat_feats'].view(-1,self.config['seq_max_len'])
        sample['float_feats'] = sample['float_feats'].view(-1,self.config['seq_max_len'])
        return sample
    
    def read_label(self, idx):
        labels = {
            'reg': {
                'values': [],
                'masks': []
            },
        }
        time_index = self.data.loc[self.data.index[idx], 'visit_index'].astype(np.int8)

        for col in self.config['current_label']['reg_label_cols']:
            raw_value = self.data.loc[self.data.index[idx], col]
            value = np.nan_to_num(raw_value, nan=-1)
            labels['reg']['values'].append(value)
            mask = (value != -1) & ~np.isnan(value)
            labels['reg']['masks'].append(mask)

        values = torch.tensor(np.stack(labels['reg']['values'], axis=0), dtype=torch.float32)
        labels['reg']['values'] = values
        labels['reg']['masks'] = torch.tensor(np.stack(labels['reg']['masks'], axis=0), dtype=torch.bool)
 
        return labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = {
            'pid': self.data.loc[self.data.index[idx], 'patient_sn'],
            'data': self.read_sample(idx),
            'label': self.read_label(idx)
        }
        return result
    
class EHRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.df_path = config.get('df_path', None)
        self.use_cache = config.get('use_cache', False)

        self.dataset_col = config.get('dataset_col', None)
        self.batch_size = config.get('batch_size', None)
        self.train_folds = config.get('train_folds', None)
        self.valid_folds = config.get('valid_folds', None)
        self.test_folds = config.get('test_folds', None)
        
    def setup(self, stage=None):
        if isinstance(self.df_path, pd.DataFrame):
            df = self.df_path
        else:
            if self.use_cache:
                df = load_parquet(Path(self.df_path) / 'metadata.parquet')
            else:
                df = load_parquet(Path(self.df_path))
        df_train = df[df[self.dataset_col].isin(self.train_folds)].copy()
        df_train = df_train.head(df_train.shape[0]//self.batch_size*self.batch_size)
        df_valid = df[df[self.dataset_col].isin(self.valid_folds)].copy()
        df_valid = df_valid.head((df_valid.shape[0]//self.batch_size-1)*self.batch_size)
        self.ds_train = EHRDataset(df_train, 'train', self.config)
        self.ds_valid = EHRDataset(df_valid, 'valid', self.config)
        if self.config.get('test_df', '') != '':
            df_test = pd.read_csv(self.config['test_df'])
        else:
            df_test = df[df[self.dataset_col].isin(self.test_folds)].copy()
        self.ds_test = EHRDataset(df_test, 'test', self.config)
        

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=4, 
                          pin_memory=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=4, 
                          pin_memory=False, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=4,
                          pin_memory=False, shuffle=False)

    def teardown(self, stage=None):
        pass
