import os
from ehr_dataset import EHRDataModule
from ehr_model_module_pretrain import EHRModule
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from lr_monitor2 import LearningRateMonitor
from transformers import GPT2Config
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from Utils import *


config = json_load('configs/train_reg.json')
config['transformer'] = GPT2Config.from_pretrained('gpt2')

debug = config['debug']
output_dir = Path(config['output_dir'])
n_gpus = config['n_gpus']
n_epoch = config['n_epoch']

# data module
data_module = EHRDataModule(config)
data_module.setup()

# model module
model_module = EHRModule(config)

# trainer
log_dir = output_dir / 'log'
if not config['train'] and config['test']:
    version = get_max_version(str(log_dir))
else:
    version = None
logger_csv = CSVLogger(str(log_dir), version=version)
version_dir = Path(logger_csv.log_dir)
loggers = [logger_csv]
if not debug:
    logger_wandb = WandbLogger(project=config['project'])
    loggers.append(logger_wandb)
callbacks = [
    ModelCheckpoint(dirpath=(version_dir / 'checkpoint'), filename='{epoch}-{val_loss:.3f}',
                    monitor='val_loss', mode='min'),
    TQDMProgressBar(refresh_rate=1),
]
if not debug:
    callbacks.append(LearningRateMonitor(logging_interval='epoch', logger_indexes=1))


trainer = Trainer(
    accelerator='gpu', devices=[0],
    max_epochs=n_epoch,
    logger=loggers,
    callbacks=callbacks,
    strategy='ddp_find_unused_parameters_true',
    precision='bf16-mixed',
    sync_batchnorm=True
)
trainer.fit(
    model_module, 
    datamodule=data_module, 
)
if config['test']:
    trainer = Trainer(
        inference_mode=True,
        accelerator ='gpu', devices=[0]
    )
    ckpt = torch.load(os.path.join(version_dir,'/SEAL_combine_all.ckpt'))
    model_module.load_state_dict(ckpt['state_dict'], strict=False)
    trainer.test(model_module, datamodule=data_module)