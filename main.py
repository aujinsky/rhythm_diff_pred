import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torchmetrics
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.nn.utils.rnn import pad_sequence
import argparse
import torch.optim
import random
import os

from rhythm_rae import RhythmRAE
from utils import dataset

temp_dict = {}



def gen_src_mask(total_len, len_list):
    batch_len = len(len_list)
    zero = torch.zeros(batch_len, total_len)
    for tens, t in zip(zero, len_list):
        mask = torch.ones(total_len-t) 
        tens[t:] = mask
    ret = zero.bool()
    return ret

def save_args(args, filename):
    args_dict = vars(args)
    with open(filename, 'w') as file:
        json.dump(args_dict, file, indent=4)


class TrainerModule(pl.LightningModule):
    def __init__(self, dataset_type="osumania", **kwargs):
        super().__init__()
        self.save_hyperparameters({'ddataset_type': dataset_type, 'kwargs': kwargs})
        #self.save_hyperparameters(kwargs)
    
        self.dataset_type = dataset_type
        self.train_loss_list = []
        self.test_acc_list = []
        self.alpha = 0.0001
        self.num_keys = 4

        self.model = RhythmRAE(num_keys = self.num_keys)

    def training_step(self, batch, batch_idx):
        pattern = batch[0]
        len_list = batch[1][0]
        max_len = batch[1][1]
        target = batch[2]

        mask = gen_src_mask(max_len, len_list).to('cuda')
        
        pred = self.model(pattern, mask)

        loss = F.mse_loss(pred, target)
        
        if loss.isnan().any() == True:
            import IPython; IPython.embed(); exit(1)    
        self.log('train_loss_step', loss)
        self.train_loss_list.append(loss)
        return loss
    def on_train_epoch_end(self):
        train_loss_tensor = torch.tensor(self.train_loss_list)
        self.train_loss_list = []
        average_train_loss = train_loss_tensor.mean()
        if hasattr(self.model, 'scale'):
            self.log('scale', self.model.scale)
        self.log('avg_train_loss', average_train_loss)

    def validation_step(self, batch, batch_idx):
        pattern = batch[0]
        len_list = batch[1][0]
        max_len = batch[1][1]
        target = batch[2]

        mask = gen_src_mask(max_len, len_list).to('cuda')
        
        pred = self.model(pattern, mask)

        loss = F.mse_loss(pred, target)
        
        if loss.isnan().any() == True:
            import IPython; IPython.embed(); exit(1)    
        self.log('validation_loss_step', loss)
        return loss

    def test_step(self, batch, batch_idx):
        pattern = batch[0]
        len_list = batch[1][0]
        max_len = batch[1][1]
        target = batch[2]

        mask = gen_src_mask(max_len, len_list).to('cuda')
        
        pred = self.model(pattern, mask)

        loss = F.mse_loss(pred, target)
        
        if loss.isnan().any() == True:
            import IPython; IPython.embed(); exit(1)    
        self.log('test_loss_step', loss)
        return loss

    def on_test_epoch_end(self):
        test_acc_tensor = torch.tensor(self.test_acc_list[0::2])
        test_samples_tensor = torch.tensor(self.test_acc_list[1::2])
        self.test_acc_list = []
        average_test_acc = torch.sum(test_acc_tensor * test_samples_tensor) / torch.sum(test_samples_tensor)
        
        self.log('avg_test_acc', average_test_acc)

    def configure_optimizers(self):
        """
        optimizer = torch.optim.Adam(model.parameters(), 
                           lr=0.001,  # Initial learning rate
                           betas=(0.9, 0.999),  # Setting beta1 as momentum
                           weight_decay=0.0001)  # Setting the weight decay
    
        # Creating a learning rate scheduler that reduces the learning rate by a factor of 0.1 every 20 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)    
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "avg_train_loss"}   
    

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, dataset_type = "osumania"):
        super().__init__()
        self.dataset_type = dataset_type
        self.save_hyperparameters()
        self.batch_size = batch_size
                
    def setup(self, stage=None):
        if self.dataset_type == "osumania":
            self.osumania_dataset = dataset.OsuManiaDataset()
            
            self.dataset_train, self.dataset_validation, self.dataset_test = torch.utils.data.random_split(self.osumania_dataset, [10076, 500, 500])
            #self.nia_train = torch.utils.data.ConcatDataset([org_train] + aug_train_list)
        else:
            NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size = self.batch_size, num_workers=2, collate_fn=self.osumania_dataset.collate_fn, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size = self.batch_size, num_workers=2, collate_fn=self.osumania_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size = self.batch_size, num_workers=2, collate_fn=self.osumania_dataset.collate_fn)
    
def get_arguments():
    return 



if __name__=="__main__":
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)    
    max_epochs = 30
    torch.set_float32_matmul_precision('medium')
    custom_callback = []

    logger = TensorBoardLogger("rrae_logs", name= "osumania")

   
    trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, devices=1, logger = logger, callbacks=custom_callback)
        
    model = TrainerModule()
    
    data = DataModule()
    
    trainer.test(model, data)
