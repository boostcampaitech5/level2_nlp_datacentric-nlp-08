import datetime

import torch
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
def time_check():
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string