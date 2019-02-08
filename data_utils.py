import pandas as pd
import numpy as np
from dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder


def load_data(data_path, batch_size, device, test_split=.1, val_split=.1, shuffle=True):
    df = pd.read_csv(data_path, header=0)
    train_df, test_df = partition_data(df, test_split)
    train_df, val_df = partition_data(train_df, val_split)

    encoder = LabelEncoder()
    encoder.fit(df[df.columns[-1]].unique())

    train_set = Dataset(train_df, device, encoder)
    validation_set = Dataset(val_df, device, encoder)
    test_set = Dataset(test_df, device, encoder)

    train_generator = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_generator = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle)
    test_generator = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_generator, validation_generator, test_generator


def get_task(data_path, batch_size, device, test_split=.1, val_split=.1, shuffle=True):
    df = pd.read_csv(data_path, header=0)
    classes = np.random.choice(df[df.columns[-1]].unique(), 2, replace=False)
    df = df.loc[df[df.columns[-1]].isin(classes)]

    train_df, test_df = partition_data(df, test_split)
    train_df, val_df = partition_data(train_df, val_split)

    encoder = LabelEncoder()
    encoder.fit(df[df.columns[-1]].unique())

    train_set = Dataset(train_df, device, encoder)
    validation_set = Dataset(val_df, device, encoder)
    test_set = Dataset(test_df, device, encoder)

    train_generator = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_generator = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle)
    test_generator = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_generator, validation_generator, test_generator


def partition_data(df, split=.2):
    mask = np.random.rand(df.shape[0]) < split
    return df[~mask], df[mask]