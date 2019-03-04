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


def sample_tasks(df, n_tasks, classes, n_instances, n_way, device=None):
    # TODO: is it ok if there is some overlap between train and test instances?
    tasks = []
    for x in range(n_tasks):
        task_classes = np.random.choice(classes, n_way, replace=False)
        # task_df = pd.concat([df.loc[df[df.columns[-1]] == x].sample(n=num_instances) for x in task_classes])
        # tasks.append(DataLoader(Dataset(task_df, device), batch_size=n_way*num_instances, shuffle=True))
        samples = [df.loc[df[df.columns[-1]] == x].sample(n=n_way * n_instances) for x in task_classes]
        task_train_df = pd.concat([samples[x][:n_instances] for x in range(len(task_classes))])
        task_test_df = pd.concat([samples[x][n_instances:] for x in range(len(task_classes))])
        tasks.append(
            (DataLoader(Dataset(task_train_df, device), batch_size=n_way * n_instances, shuffle=True),
             DataLoader(Dataset(task_test_df, device), batch_size=n_way * n_instances, shuffle=True))
        )
    return tasks


def split_data(data, target, n_way):
    classes = data[target].unique()
    test_classes = np.random.choice(classes, n_way, replace=False)
    train_classes = [x for x in classes if x not in test_classes]

    test_df = data.loc[data[target].isin(test_classes)]
    train_df = data.loc[data[target].isin(train_classes)]

    return train_classes, train_df, test_classes, test_df


def partition_data(df, split=.2):
    mask = np.random.rand(df.shape[0]) < split
    return df[~mask], df[mask]