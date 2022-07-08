#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pytorch_lightning')


# In[2]:


get_ipython().system('pip install Lightning')


# In[3]:


from pathlib import Path
from pytorch_lightning import LightningModule, Trainer


# In[4]:


get_ipython().system('pip install pydicom')


# In[5]:


get_ipython().system('pip install torchmetrics')


# In[6]:


get_ipython().system('pip install pytorch_lightning')


# In[7]:


import pydicom


# In[8]:


import numpy as np


# In[9]:


import cv2


# In[10]:


import pandas as pd


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


from tqdm.notebook import tqdm


# In[13]:


import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as ptl


# In[14]:


labels = pd.read_csv("train.csv")


# In[15]:


labels.head(100)


# In[16]:


ROOT_PATH = Path("Data/train")


# In[17]:


SAVE_PATH = Path("Analysed")


# In[18]:


fig, axis = plt.subplots(3,3, figsize=(9,9))
c = 0
for i in range(3):
    for j in range(3):
        StudyInstance_UID = labels.StudyInstanceUID.iloc[c]
        SeriesInstance_UID = labels.SeriesInstanceUID.iloc[c]
        SOPInstance_UID = labels.SOPInstanceUID.iloc[c]
        dcm_path = ROOT_PATH/StudyInstance_UID/SeriesInstance_UID/SOPInstance_UID
        dcm_path = dcm_path.with_suffix(".dcm")
        dcm = pydicom.read_file(dcm_path).pixel_array
        
        
        label = labels["pe_present_on_image"].iloc[c]
        axis[i][j].imshow(dcm, cmap="gray")
        axis[i][j].set_title(label)
        c+=1


# In[19]:


get_ipython().system('pip install ipywidgets')


# In[20]:


from tqdm.notebook import tqdm


# In[21]:


sums, sums_squared = 0, 0

for c, SOPInstance_UID in enumerate(labels.SOPInstanceUID):
    StudyInstance_UID = labels.StudyInstanceUID.iloc[c]
    SeriesInstance_UID = labels.SeriesInstanceUID.iloc[c]
    SOPInstance_UID = labels.SOPInstanceUID.iloc[c]
    dcm_path = ROOT_PATH/StudyInstance_UID/SeriesInstance_UID/SOPInstance_UID
    dcm_path = dcm_path.with_suffix(".dcm")
    dcm = pydicom.read_file(dcm_path).pixel_array / 255
    
    dcm_array = cv2.resize(dcm, (224,224))
    label = labels.pe_present_on_image.iloc[c]
    train_or_val = "train" if c<800000 else "val"
    current_save_path = SAVE_PATH/train_or_val/str(label)
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path/SOPInstance_UID, dcm_array)
    
    normalizer = 224*224
    if train_or_val == "train":
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (dcm_array **2).sum() /normalizer
    
    
 


# In[22]:


print(sums)


# In[23]:


mean = sums / 24000
std = np.sqrt((sums_squared / 24000) - mean**2)


# mean, std

# In[24]:


mean


# In[25]:


std


# In[26]:


def load_file(path):
    return np.load(path).astype(np.float32)


# In[27]:


train_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(0.49, 0.248),
                    transforms.RandomAffine(degrees=(-5,5), translate=(0,0.05), scale=(0.9,1.1)), 
                    transforms.RandomResizedCrop((224, 224), scale=(0.35,1))
])

val_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(0.49, 0.248),
])


# In[28]:


train_dataset = torchvision.datasets.DatasetFolder("Analysed/train/",loader=load_file, extensions="npy",transform=train_transforms)
val_dataset = torchvision.datasets.DatasetFolder("Analysed/val/",loader=load_file, extensions="npy",transform=val_transforms)


# In[29]:


fig, axis = plt.subplots(2,2, figsize=(9,9))
for i in range(2):
    for j in range(2):
        random_index = np.random.randint(0,24000)
        x_ray, label = train_dataset[random_index]
        axis[i][j].imshow(x_ray[0], cmap="gray")
        axis[i][j].set_title(label)


# In[30]:


batch_size = 64
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, num_workers=num_workers, shuffle=False)


# In[ ]:





# In[31]:


np.unique(train_dataset.targets, return_counts=True)


# In[32]:


torchvision.models.resnet18()


# In[45]:


class CtpaAnalysis(pl.LightningModule):
    def __init_(self):
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self,batch, batch_idx):
        ctpa, label = batch
        label = label.float()
        pred = self(ctpa)[:,0]
        loss = self.loss_fn(pred, label)
        
        self.log("Train Loss", loss)
        self.log("Step Train ACC", self.train_acc(torch.sigmoid(pred), label.int()))
        
        return loss
    
    def training_epoch_end(self, outs):
        self.log("Train ACC", self.train_acc.compute())
        
    def validation_step(self,batch, batch_idx):
        ctpa, label = batch
        label = label.float()
        pred = self(ctpa)[:,0]
        loss = self.loss_fn(pred, label)
        
        self.log("Val Loss", loss)
        self.log("Step Val ACC", self.val_acc(torch.sigmoid(pred), label.int()))
        
    
    def validation_epoch_end(self, outs):
        self.log("Val ACC", self.val_acc.compute())
        
    def configure_optimizers(self):
        return [self.optimizer]
        


# In[46]:


model = CtpaAnalysis()


# In[47]:


checkpoint_callback = ModelCheckpoint(
        monitor="Val ACC", 
        save_top_k=10,
        mode="max"
)


# In[48]:


import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


# In[49]:


gpus = 1
trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=35)


# In[50]:


trainer.fit(model, train_loader, val_loader)


# In[ ]:


optimizer


# In[ ]:





# In[ ]:




