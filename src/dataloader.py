import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import pathlib
import consts as c

SQ_SIZE = 1000
N_DIM = 8
RANDOM_SEED = 8

class MyDataset(Dataset):
    def __init__(self, X, Y = None):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.Y is not None:
            y = self.Y[index]
            return x, y
        else:
            return x

def readfile(path, mode):
    pathx = path / (str(c.NOISE) + 'x' + mode + "ingM.csv")
    pathy = path / (str(c.NOISE) + 'y' + mode + "ingM.csv")

    with open(pathx, encoding='utf-8') as f:
        x = torch.FloatTensor(np.loadtxt(f, delimiter=",",skiprows=0, dtype=np.float32))
    
    with open(pathy, encoding='utf-8') as f:
        y = torch.FloatTensor(np.loadtxt(f, delimiter=",",skiprows=0, dtype=np.float32))

    y = y / 1000
    # x = x.reshape((x.shape[0], c.N_CHANNEL, c.LEN))
    # x = x.permute(1, 0, 2)
    
    return x, y

def getdataloader(path = r'D:\photon_counting_radar\dataset2', batch_size = 16, mode = 'train'):
    path = pathlib.Path(path) / ("num=" + str(c.N_CHANNEL))
    X, Y = readfile(path, mode) 
    Y = Y.T
    print(X.shape)
    if mode != 'train':
        test_set = MyDataset(X, Y)
        test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, drop_last = False)
        return test_loader

    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size = .15, random_state = RANDOM_SEED)
    train_set = MyDataset(train_X, train_Y)
    val_set = MyDataset(val_X, val_Y)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, drop_last = True)

    return train_loader, val_loader
