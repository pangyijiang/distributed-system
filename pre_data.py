# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from utils import SaveToPickleFile, LoadFromPickleFile
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

seed = 0#np.random.randint(0, 1000000)

if seed is not None:
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def trans_class_rate(num_class, n_local_pickle = "data.pickle"):
    if not os.path.exists(n_local_pickle):
        df = pd.read_csv('./newFormatData.csv', sep=',')
        df.drop(['date'], axis=1, inplace=True)
        df.sort_values(by=["userID"], inplace=True)
        df = df.reset_index(drop=True)
        dataset_rating = df.pivot_table(index="userID", values="rating", columns="movieID")
        dataset_rating = dataset_rating.to_numpy()
        dataset_rating = dataset_rating[:, : - (dataset_rating.shape[1]%num_class)]
        dataset_rating = np.stack(np.split(dataset_rating, num_class, axis = 1), axis = 1)
        dataset_rating = np.nanmean(dataset_rating, axis=2)
        dataset_rating = np.nan_to_num(dataset_rating, nan = 0.0)
        dataset_rating = np.round(dataset_rating, 1)
        SaveToPickleFile(dataset_rating, n_local_pickle)
    else:
        dataset_rating = LoadFromPickleFile(n_local_pickle)
    return dataset_rating


class CustomImageDataset(Dataset):
    def __init__(self, dataset_rating, label_rep = -0.5):
        self.dataset_rating = dataset_rating
        self.n_class = dataset_rating.shape[1]
        self.label_rep = label_rep

    def __len__(self):
        return len(self.dataset_rating)

    def __getitem__(self, idx):
        data = self.dataset_rating[idx]

        label_hid_number = np.random.randint(1, int(self.n_class/2))
        label_hid_loc = np.random.randint(0, self.n_class, label_hid_number)
        label_hid_onehot = np.ones(self.n_class)
        label_hid_onehot[label_hid_loc] = -1
        data_hidden = data*label_hid_onehot
        data_hidden[data_hidden < 0 ] = -1

        return data_hidden, data


def pre_dataloader(train_rate, num_class):
    dataset_rating = trans_class_rate(num_class)
    data_train = dataset_rating[:int(train_rate*len(dataset_rating))]
    data_test = dataset_rating[int(train_rate*len(dataset_rating)):]
    training_data = CustomImageDataset(data_train)
    test_data = CustomImageDataset(data_test)


    train_sampler = DistributedSampler(dataset=training_data)

    train_dataloader = DataLoader(training_data, sampler=train_sampler, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader