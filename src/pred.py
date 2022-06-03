import torch
import time
import pandas as pd
import numpy as np
from dataloader import getdataloader
from TCN import TemporalConvNet
import consts as c

model = TemporalConvNet(1, [1, 8, 16, 8, 1])

USE_CUDA = torch.cuda.is_available

def test(model, batch_size):
    # Model test
    st = time.time()
    
    model.load_state_dict(torch.load("_UNet_best_state_dict.pt"))
    model = model.cuda() if USE_CUDA else model
    model.eval()

    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    # Create test dataset and dataloader
    test_loader = getdataloader(batch_size = batch_size, mode = 'test')
    
    with torch.no_grad():
        cnt, loss_sum = 0, 0
        for i, (batch_X, batch_Y) in enumerate(test_loader):
            if USE_CUDA:
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
            batch_pred = model(batch_X)
            loss = loss_fn(batch_Y, batch_pred)
            loss_sum += loss
            cnt += 1
            if i == 0:
                gt, pred = batch_Y, batch_pred
            else:
                gt, pred = torch.cat((gt, batch_Y), dim = 0), torch.cat((pred, batch_pred), dim = 0)
    final_loss = loss_sum / cnt
    ed = time.time()
    print("Inference Time consumption: {}s, Test_Loss: {}.".format(ed - st, final_loss))

    gt, pred = gt.cpu().numpy().reshape(-1, 1), pred.cpu().numpy().reshape(-1, 1)
    data = np.concatenate((gt, pred), axis = 1)

    np.savetxt('result.csv', data, delimiter=",")
test(model, c.BATCH_SIZE)