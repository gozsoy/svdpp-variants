import os
import logging
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from model import RegularizedSVD, SVDPP, Bayesian_SVDPP
from dataset import SVD_Dataset, SVDPP_Dataset


def get_logger(cfg):

    logger = logging.getLogger(cfg['experiment_name']+cfg['model'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  - %(message)s', 
                                  '%d-%m-%Y %H:%M')
    file_handler = logging.FileHandler(os.path.join(
        cfg['log_dir'], cfg['model'], cfg['experiment_name']))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_data(cfg):

    # read MovieLens 1M dataset
    ratings_df = pd.read_csv(cfg['data_path'], sep="::", header=None)

    # rename columns
    ratings_df = ratings_df[[0, 1, 2]].rename(
        columns={0: 'user_id', 1: 'movie_id', 2: 'rating'})

    # split into train, valid and test sets
    train_valid_df, test_df = train_test_split(
        ratings_df, test_size=cfg['test_size'], 
        random_state=cfg['test_split_random_state'])
    train_df, valid_df = train_test_split(
        train_valid_df, test_size=cfg['valid_size'], 
        random_state=cfg['valid_split_random_state'])

    global_mean = np.mean(train_df.rating.values)

    # user rated items necessary for svdpp
    user_rated_items_df = train_df.groupby('user_id')['movie_id'].agg(
        lambda row: list(row)).reset_index()
    user_rated_items_df['counter'] = user_rated_items_df['movie_id'].apply(
        lambda row: len(row))

    return train_df, valid_df, test_df, global_mean, user_rated_items_df


# performance metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_model(cfg, global_mean):

    if cfg['model'] == 'regularized_svd':
        return RegularizedSVD(
            num_users=cfg['num_users'], num_items=cfg['num_items'],
            global_mean=global_mean, embedding_dim=cfg['embedding_dim'])
    elif cfg['model'] == 'svdpp':
        return SVDPP(
            num_users=cfg['num_users'], num_items=cfg['num_items'],
            global_mean=global_mean, embedding_dim=cfg['embedding_dim'])
    elif cfg['model'] == 'bayesian_svdpp':
        return Bayesian_SVDPP(
            num_users=cfg['num_users'], num_items=cfg['num_items'],
            global_mean=global_mean, embedding_dim=cfg['embedding_dim'])
    else:
        raise NotImplementedError()


def get_dataloader(cfg, df, user_rated_items_df):

    if cfg['model'] == 'regularized_svd':
        dataset = SVD_Dataset(df)
        return DataLoader(
            dataset=dataset, batch_size=cfg['batch_size'], shuffle=True)

    elif cfg['model'] == 'svdpp' or cfg['model'] == 'bayesian_svdpp':
        dataset = SVDPP_Dataset(df, user_rated_items_df)
        
        # pad short rated_movies list with 0
        def collate_fn(batch):

            rated_item_max_len = np.max(list(
                map(lambda row: row[3], batch)))

            def aux_f(sample):
                temp_sample = list(sample)
                temp_sample[2] = temp_sample[2] + [0] * \
                    (rated_item_max_len - len(temp_sample[2]))
                return tuple(temp_sample)

            batch = list(map(aux_f, batch))

            u_id_tensor = torch.tensor(list(
                map(lambda row: row[0], batch)), dtype=torch.int64)
            m_id_tensor = torch.tensor(list(
                map(lambda row: row[1], batch)), dtype=torch.int64)
            rates_items_tensor = torch.tensor(list(
                map(lambda row: row[2], batch)), dtype=torch.int64)
            rated_count = torch.tensor(list(
                map(lambda row: row[3], batch)), dtype=torch.float32)
            r_tensor = torch.tensor(list(
                map(lambda row: row[4], batch)), dtype=torch.int64)
            return (u_id_tensor, m_id_tensor,
                    rates_items_tensor, rated_count), r_tensor

        return DataLoader(dataset=dataset, batch_size=cfg['batch_size'], 
                          shuffle=True, collate_fn=collate_fn, num_workers=5)

    else:
        raise NotImplementedError()