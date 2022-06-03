import consts as c
import torch
from TCN import TemporalConvNet
from train import train
from test import test

model = TemporalConvNet(c.N_CHANNEL, [c.N_CHANNEL, 8, 16, 8, 1])
train(model, c.N_EPOCHS, c.BATCH_SIZE)
test(model, c.BATCH_SIZE)