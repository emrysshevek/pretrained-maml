import torch
from torch.utils import data
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class Dataset(data.Dataset):

    def __init__(self, dataframe, device=None, encoder=None):
        self.labels = dataframe[dataframe.columns[-1]]
        self.instances = dataframe[dataframe.columns[:-1]]
        self.device = device
        self.encoder = encoder

        if not self.encoder:
            self.encoder = LabelEncoder()
            self.encoder.fit(self.labels)
        self.labels = pd.Series(self.encoder.transform(self.labels))

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, index):
        X = torch.tensor(self.instances.iloc[index], dtype=torch.float32, device=self.device)
        # print(self.labels.iloc[index], type(self.labels.iloc[index]))
        Y = torch.tensor((self.labels.iloc[index]), dtype=torch.int64, device=self.device)
        return X, Y

# def __init__(self, list_IDs, labels):
# 	self.labels = labels
# 	self.list_IDs = list_IDs

# def __len__(self):
# 	return len(self.list_IDs)

# def __getitem__(self, index):
# 	ID = self.list_IDs[index]
# 	X = torch.load('data/' + ID + '.pt')
# 	y = self.labels[ID]
# 	return X, y
