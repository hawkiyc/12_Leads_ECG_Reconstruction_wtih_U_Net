#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 17:11:58 2023

@author: hawkiyc
"""

#%%
"Import Library"

from datetime import datetime
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

mpl.style.use('seaborn')

#%%
"Setting GPU"

if torch.cuda.is_available():
    device = torch.device('cuda')

elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was "
              "NOT built with MPS enabled.")
    else:
        print("MPS not available because this MacOS version is NOT 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
print(device)

#%%
'Load Data'

root_path = '../data/data_for_12_leads_reconstructions/'
f_path = ["normal_ecg", 'other_abnormal']

for p in f_path:
    f_list = [f for f in os.listdir(root_path + p) if ".npz" in f]
    
    for i in range(len(f_list)):
        
        if i == 0 and f_path.index(p) == 0:
            _ = np.load(root_path+p+"/"+f_list[i], allow_pickle=True)
            p_id,ecg, = _['p_id'],_['ecg'][:,:,648:3448],
            
        else:
            _ = np.load(root_path+p+"/"+f_list[i], allow_pickle=True)
            _p_id,_ecg, = _['p_id'],_['ecg'][:,:,648:3448],
            p_id=np.concatenate((p_id,_p_id))
            ecg=np.concatenate((ecg,_ecg))

x = ecg[:,1,:].reshape(ecg.shape[0],1,ecg.shape[2])
y = np.concatenate((ecg[:,0,:].reshape(ecg.shape[0],1,ecg.shape[2]),
                    ecg[:,2:,:]),axis=1)

normal_ecg_train, normal_ecg_test, normal_ecg_val = \
    pd.read_csv("../data/normal_ecg_split_idx.csv", 
                usecols = ['train'],).convert_dtypes().dropna(),\
        pd.read_csv("../data/normal_ecg_split_idx.csv", 
                    usecols = ['test'],).convert_dtypes().dropna(),\
            pd.read_csv("../data/normal_ecg_split_idx.csv", 
                        usecols = ['val'],).convert_dtypes().dropna()

normal_ecg_train, normal_ecg_test, normal_ecg_val = \
    list(normal_ecg_train.iloc[:,0]), \
        list(normal_ecg_test.iloc[:,0]), \
            list(normal_ecg_val.iloc[:,0])
     
other_abnormal_train, other_abnormal_test, other_abnormal_val = \
    pd.read_csv("../data/other_abnormal_split_idx.csv", 
                usecols = ['train'],).convert_dtypes().dropna(),\
        pd.read_csv("../data/other_abnormal_split_idx.csv", 
                    usecols = ['test'],).convert_dtypes().dropna(),\
            pd.read_csv("../data/other_abnormal_split_idx.csv", 
                        usecols = ['val'],).convert_dtypes().dropna()

other_abnormal_train, other_abnormal_test, other_abnormal_val = \
    list(other_abnormal_train.iloc[:,0]), \
        list(other_abnormal_test.iloc[:,0]), \
            list(other_abnormal_val.iloc[:,0])
     
single_arrhythmia_train, single_arrhythmia_test, single_arrhythmia_val = \
    pd.read_csv("../data/single_arrhythmia_split_idx.csv", 
                usecols = ['train'],).convert_dtypes().dropna(),\
        pd.read_csv("../data/single_arrhythmia_split_idx.csv", 
                    usecols = ['test'],).convert_dtypes().dropna(),\
            pd.read_csv("../data/single_arrhythmia_split_idx.csv", 
                        usecols = ['val'],).convert_dtypes().dropna()

single_arrhythmia_train, single_arrhythmia_test, single_arrhythmia_val = \
    list(single_arrhythmia_train.iloc[:,0]), \
        list(single_arrhythmia_test.iloc[:,0]), \
            list(single_arrhythmia_val.iloc[:,0])

multi_arrhythmia_train, multi_arrhythmia_test, multi_arrhythmia_val = \
    pd.read_csv("../data/multi_arrhythmia_split_idx.csv", 
                usecols = ['train'],).convert_dtypes().dropna(),\
        pd.read_csv("../data/multi_arrhythmia_split_idx.csv", 
                    usecols = ['test'],).convert_dtypes().dropna(),\
            pd.read_csv("../data/multi_arrhythmia_split_idx.csv", 
                        usecols = ['val'],).convert_dtypes().dropna()

multi_arrhythmia_train, multi_arrhythmia_test, multi_arrhythmia_val = \
    list(multi_arrhythmia_train.iloc[:,0]), \
        list(multi_arrhythmia_test.iloc[:,0]), \
            list(multi_arrhythmia_val.iloc[:,0])

train_id = normal_ecg_train+\
    other_abnormal_train+\
        single_arrhythmia_train+\
            multi_arrhythmia_train

test_id = normal_ecg_test+\
    other_abnormal_test+\
        single_arrhythmia_test+\
            multi_arrhythmia_test

val_id = normal_ecg_val+\
    other_abnormal_val+\
        single_arrhythmia_val+\
            multi_arrhythmia_val

train_indices = np.where(np.in1d(p_id, train_id))[0]
test_indices = np.where(np.in1d(p_id, test_id))[0]
val_indices = np.where(np.in1d(p_id, val_id))[0]

train_p_id, train_x, train_y = \
    p_id[train_indices], x[train_indices], y[train_indices]
test_p_id, test_x, test_y = \
    p_id[test_indices], x[test_indices], y[test_indices]
val_p_id, val_x, val_y = \
    p_id[val_indices], x[val_indices], y[val_indices]


print(set(train_p_id) & set(test_p_id))
print(set(train_p_id) & set(val_p_id))
print(set(val_p_id) & set(test_p_id))

#%%
'Data Check'

data_check = False
leads = ["DI", 'DIII', 'AVR', 'AVL', 'AVF', 
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def verify_data():
    idx = np.random.choice(a = np.array(range(len(x))), 
                           size =10, replace = False)
    fig = plt.figure(figsize=(15, 1600))
    
    for i in idx:
        ax = fig.add_subplot(10, 1, int(np.where(idx == i)[0])+1, 
                             xticks=[], yticks=[])
        plt.plot(range(x[i].shape[1]), x[i].flatten())
        ax.set_title(f'ID: {p_id[i]}')
    
    for i in idx:
        fig2 = plt.figure(figsize=(15, 1600))
        plt.suptitle(f'ID: {p_id[i]}')
        for l in range(11):
            ax2 = fig2.add_subplot(11, 1, l+1, 
                                   xticks=[], yticks=[])
            plt.plot(range(y[i].shape[1]), y[i,l].flatten())
            ax2.set_title(f"Lead: {leads[l]}")
    
if data_check:
    verify_data()

#%%
'Torch Data Loader'

train_x = torch.from_numpy(train_x)
train_x = train_x.view(-1,train_x.shape[1],train_x.shape[2])

val_x = torch.from_numpy(val_x)
val_x = val_x.view(-1,val_x.shape[1],val_x.shape[2])

test_x = torch.from_numpy(test_x)
test_x = test_x.view(-1,test_x.shape[1],test_x.shape[2])

train_set = TensorDataset(
    train_x, torch.from_numpy(train_y))
val_set = TensorDataset(
    val_x, torch.from_numpy(val_y))
test_set = TensorDataset(
    test_x, torch.from_numpy(test_y))

batch_size = 128

train_loader = DataLoader(dataset=train_set, 
                          batch_size = batch_size, 
                          shuffle=True)
val_loader = DataLoader(dataset=val_set, 
                        batch_size = batch_size, 
                        shuffle=True)
test_loader = DataLoader(dataset=test_set, 
                         batch_size = batch_size, 
                         shuffle=False)
    
#%%
'Build U-Net Model And Other Function We Need'

class DoubleConv(nn.Module):
    #  Conv -> BN -> LReLU -> Conv -> BN -> LReLU
    def __init__(self, in_ch, out_ch, droprate):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 19, padding='same'),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(droprate),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, 19, padding='same'),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(droprate),
            nn.LeakyReLU(0.2, inplace=True),)
    def forward(self, x):
        x = self.f(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, droprate):
        super().__init__()
        self.f = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_ch, out_ch, droprate),)
    def forward(self, x):
        x = self.f(x)
        return x


class Up(nn.Module):
    # upsample and concat
    def __init__(self, in_ch, out_ch, droprate):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, droprate)
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.f = nn.Conv1d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.f(x)
        return x

class Unet(nn.Module):
    def __init__(self, droprate):
        super().__init__()
        self.inc = DoubleConv(1, 32, droprate)
        self.d1 = Down(32, 64, droprate)
        self.d2 = Down(64, 128, droprate)
        self.d3 = Down(128, 256, droprate)
        self.d4 = Down(256, 512, droprate)

        self.u1 = Up(512, 256, droprate)
        self.u2 = Up(256, 128, droprate)
        self.u3 = Up(128, 64, droprate)
        self.u4 = Up(64, 32, droprate)
        self.outc = OutConv(32, 11)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)#out = 1400
        x3 = self.d2(x2)#out = 700
        x4 = self.d3(x3)#out = 350
        x5 = self.d4(x4)#out = 175
        x = self.u1(x5, x4)#out = 350
        x = self.u2(x, x3)#out = 700
        x = self.u3(x, x2)#out = 1400
        x = self.u4(x, x1)#out =2800
        x = self.outc(x)
        return x

#%%
'Training the Model'

model = Unet(droprate = .05)
model.to(device)
loss_fn = nn.L1Loss()

opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.7, patience=3)

n_epochs = 80
step_per_epoch = math.ceil(len(train_set)/batch_size)
train_loss = []
val_loss = []

os.mkdir('val_reconstruction') if os.path.exists(
    'val_reconstruction') is False else None

for epoch in range(n_epochs):
    t0 = datetime.now()
    loss_per_step = []
    model.train()
    with tqdm(train_loader, unit='batch',) as per_epoch:
        for x,y in per_epoch:
            opt.zero_grad()
            per_epoch.set_description(f"Epoch: {epoch+1}/{n_epochs}")
            x,y = x.to(device,
                       torch.float32), y.to(device,
                                            torch.float32)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss_per_step.append(loss.item())
            loss.backward()
            opt.step()
            per_epoch.set_postfix(train_loss=loss.item(), 
                                  )
        if scheduler.__module__ == 'torch.optim.lr_scheduler':
            scheduler.step(1)
    train_loss.append(sum(loss_per_step)/len(loss_per_step))
    
    val_loss_per_step = []
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, unit='batch',) as per_val_epoch:
            for x_val,y_val in per_val_epoch:
                per_val_epoch.set_description("Model Evaluation: ")
                x_val,y_val = x_val.to(device,
                                       torch.float32), y_val.to(device,
                                                                torch.float32)
                y_hat_val = model(x_val)
                loss_val = loss_fn(y_hat_val, y_val)
                val_loss_per_step.append(loss_val.item())
                per_val_epoch.set_postfix(val_loss=loss_val.item(), 
                                          )
    val_loss.append(sum(val_loss_per_step)/len(val_loss_per_step))
    if (epoch + 1) % 5 == 0:
        
        for i in range(5):
            temp = np.concatenate((x_val[i].cpu().numpy(),
                                   y_hat_val[i].cpu().numpy()),
                                  axis=0)
            fig = plt.figure(figsize=(15, 1600))
            plt.suptitle(f'Epoch: {epoch+1:03d}_{i}')
            
            for l in range(12):
                ax2 = fig.add_subplot(12, 1, l+1, 
                                      xticks=[], yticks=[])
                plt.plot(temp[l].flatten(), label='Reconstructed')
                if l == 0:
                    ax2.set_title('Lead: DII')
                else:
                    plt.plot(y_val[i,l-1,:].cpu().numpy().flatten(), 
                             label='Ground Truth')
                    ax2.set_title(f"Lead: {leads[l-1]}")
                    plt.legend()
            plt.savefig(f'val_reconstruction/epoch_{epoch+1:03d}_{i}.png')
            plt.close()
    
    dt = datetime.now() - t0
    print(f'train_loss: {train_loss[-1]:.6f}, val_loss: {val_loss[-1]:.6f}, ',
          f'Time Duration: {dt}')

#%%
'Results Visualization'

'Plot Loss'
plt.figure()
plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.legend()
plt.show()
plt.savefig('U-Net_Loss_Result.png', )

os.mkdir('results') if os.path.exists(
    'results') is False else None

'Reconstruct 12 Leads ECG'
predictions = np.random.randn(1,11,2800)
model.eval()
with torch.no_grad():
    temp_test_loss = []
    for x,y in test_loader:
        x,y = x.to(device,torch.float32), y.to(device,torch.float32)
        output = model(x)
        iter_test_loss = loss_fn(output, y)
        temp_test_loss.append(iter_test_loss.item())
        predictions = np.concatenate((predictions,
                                      output.cpu().numpy()))
test_loss = sum(temp_test_loss)/len(temp_test_loss)
print(f"test loss: {test_loss:.6f}")
predictions = predictions[1:]
np.savez('results/Reconstructed_12_Leads_ECG',
         DII = test_x.cpu().numpy(),
         Ground_Truth = test_y,
         Reconstructed = predictions)

'Plot Re-Construction Results'

for i in range(len(test_p_id)):
    x_ = test_x[i].cpu().numpy()
    y_ = test_y[i]
    fig = plt.figure(figsize=(15, 1600))
    plt.suptitle(f'ID: {test_p_id[i]}')
    for l in range(12):
        ax2 = fig.add_subplot(12, 1, l+1, 
                               xticks=[], yticks=[])
        if l == 0:
            plt.plot(x_.flatten())
            ax2.set_title('Lead: DII')
        else:            
            plt.plot(y_[l-1].flatten(), label='Ground Truth')
            plt.plot(predictions[i,l-1].flatten(), label='Reconstructed')
            ax2.set_title(f"Lead: {leads[l-1]}")
            plt.legend()
    plt.savefig(f'results/ID_{p_id[i]}.png')
    plt.close()

#%%
'Save the Model'

torch.save(model, 'whole_model.pt')
