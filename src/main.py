import os
import yaml
import logging
import argparse

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from model import RegularizedSVD
from dataset import CF_Dataset

# performance metric
rmse = lambda y_true,y_pred: np.sqrt(mean_squared_error(y_true, y_pred))


def train(cfg, train_df, valid_df, global_mean):

    train_dataset = CF_Dataset(train_df)
    valid_dataset = CF_Dataset(valid_df)
    
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=cfg['batch_size'],shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=cfg['batch_size'])

    model = RegularizedSVD(num_users=cfg['num_users'], num_items=cfg['num_items'], global_mean=global_mean, embedding_dim=cfg['embedding_dim']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = nn.MSELoss()

    lowest_val_rmse = float('inf')

    # zero the parameters' gradients
    optimizer.zero_grad()


    for epoch in range(cfg['epochs']):  # loop over dataset

        logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}')

        # training
        model.train()
        
        batch_train_loss_array=[]
        batch_train_rmse_array=[]
        batch_train_reg_loss_array=[] # temporary

        for batch_data in train_dataloader: # loop over train batches
            
            x, y_true = batch_data[0], batch_data[1]
            y_true = y_true.to(torch.float32)

            optimizer.zero_grad()

            # forward pass
            y_pred = model(x)

            # compute loss
            mse_loss = loss_fn(y_true.to(device),y_pred)

            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.norm(param,'fro')**2
            
            loss = mse_loss + cfg['beta'] * reg_loss.to(device)

            # backpropagation
            loss.backward()

            # gradient descent with optimizer
            optimizer.step()
                
            # save batch metrics
            batch_train_loss_array.append(mse_loss.detach().cpu().item())
            batch_train_rmse_array.append(rmse(y_true, y_pred.detach().cpu()))
            batch_train_reg_loss_array.append(reg_loss.detach().cpu().item()) # temporary

        # validation
        model.eval()
        with torch.no_grad():

            batch_valid_rmse_array=[]
            
            for valid_batch_data in valid_dataloader: # loop over valid batches
                
                valid_x, valid_y_true = valid_batch_data[0], valid_batch_data[1]
                valid_y_true = valid_y_true.to(torch.float32)

                # forward pass
                valid_y_pred = model(valid_x)

                # save batch metrics
                batch_valid_rmse_array.append(rmse(valid_y_true, valid_y_pred.detach().cpu()))


        # display metrics at end of epoch
        epoch_train_loss, epoch_train_rmse = np.mean(batch_train_loss_array), np.mean(batch_train_rmse_array)
        epoch_val_rmse = np.mean(batch_valid_rmse_array)

        logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, train_loss: {epoch_train_loss:.4f}, train_rmse: {epoch_train_rmse:.4f}, val_rmse: {epoch_val_rmse:.4f}\n')

        if lowest_val_rmse > epoch_val_rmse:
            lowest_val_rmse = epoch_val_rmse

            save_dict = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss}
            torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_best.pt'))

    return


# estimating test set performance
def evaluate(cfg, test_df, global_mean):
    test_dataset = CF_Dataset(test_df)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=cfg['batch_size'])

    model = RegularizedSVD(num_users=cfg['num_users'], num_items=cfg['num_items'], global_mean=global_mean, embedding_dim=cfg['embedding_dim']).to(device)

    checkpoint = torch.load(os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_best.pt'),map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():

        batch_test_rmse_array=[]
        
        for test_batch_data in test_dataloader: # loop over test batches
            
            test_x, test_y_true = test_batch_data[0], test_batch_data[1]
            test_y_true = test_y_true.to(torch.float32)

            # forward pass
            test_y_pred = model(test_x)

            # save batch metrics
            batch_test_rmse_array.append(rmse(test_y_true, test_y_pred.detach().cpu()))


    # display metrics
    epoch_test_rmse = np.mean(batch_test_rmse_array)
    logger.info(f'------------')
    logger.info(f'TEST RESULTS (RMSE): {epoch_test_rmse:.4f}')
    logger.info(f'------------')

    return


if __name__ == '__main__':

    # load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    with open(_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # create logger
    logger = logging.getLogger(cfg['experiment_name']+cfg['model'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  - %(message)s','%d-%m-%Y %H:%M')
    file_handler = logging.FileHandler(os.path.join(cfg['log_dir'],cfg['model'],cfg['experiment_name']))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # print settings
    logger.info(f'cfg: {cfg}')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')


    # read MovieLens 1M dataset
    ratings_df = pd.read_csv(cfg['data_path'],sep="::",header=None)

    # rename columns
    ratings_df = ratings_df[[0,1,2]].rename(columns={0:'user_id',1:'movie_id',2:'rating'})

    # split into train, valid and test sets
    train_valid_df, test_df = train_test_split(ratings_df, test_size=cfg['test_size'], random_state=cfg['test_split_random_state'])
    train_df, valid_df = train_test_split(train_valid_df, test_size=cfg['valid_size'], random_state=cfg['valid_split_random_state'])

    global_mean = np.mean(train_df.rating.values)

    if cfg['evaluate']:
        evaluate(cfg, test_df, global_mean)
    else:
        train(cfg, train_df, valid_df, global_mean)