import os
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import get_model, get_dataloader, rmse, get_logger, load_data


def train(cfg, train_df, valid_df, global_mean, user_rated_items_df):

    train_dataloader = get_dataloader(cfg, train_df, user_rated_items_df)
    valid_dataloader = get_dataloader(cfg, valid_df, user_rated_items_df)

    net = get_model(cfg, global_mean).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', verbose=True)

    train_writer = SummaryWriter(log_dir=os.path.join(
        cfg['log_dir'], cfg['experiment_name'], 'train'))
    valid_writer = SummaryWriter(log_dir=os.path.join(
        cfg['log_dir'], cfg['experiment_name'], 'validation'))
    lowest_val_rmse = float('inf')

    # zero the parameters' gradients
    optimizer.zero_grad()

    for epoch in range(cfg['epochs']):  # loop over dataset

        logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}')

        # training
        net.train()
        
        batch_train_total_loss_array = []
        batch_train_reg_loss_array = []  # either l2-reg or kl-reg
        batch_train_mse_loss_array = []
        batch_train_rmse_array = []
        
        for batch_data in train_dataloader:  # loop over train batches
            
            x, y_true = batch_data[0], batch_data[1]
            y_true = y_true.to(torch.float32)

            optimizer.zero_grad()

            # forward pass
            y_pred = net(x)

            # compute loss
            mse_loss = loss_fn(y_true.to(device), y_pred)

            reg_loss = 0
            if cfg['model'] == 'regularized_svd' or cfg['model'] == 'svdpp':
                for param in net.parameters():
                    reg_loss += torch.norm(param, 'fro')**2
                reg_loss *= cfg['beta']
            else:
                reg_loss = net.compute_total_kl_loss().cpu() * cfg['kl_coef']

            loss = mse_loss + reg_loss.to(device)

            # backpropagation
            loss.backward()

            # gradient descent with optimizer
            optimizer.step()
                
            # save batch metrics
            batch_train_mse_loss_array.append(mse_loss.detach().cpu().item())
            batch_train_reg_loss_array.append(reg_loss.detach().item())
            batch_train_total_loss_array.append(loss.detach().cpu().item())
            batch_train_rmse_array.append(rmse(y_true, y_pred.detach().cpu()))
            
        # validation
        net.eval()
        with torch.no_grad():

            batch_valid_rmse_array = []
            
            for valid_batch_data in valid_dataloader:  # loop over valid batch
                
                valid_x = valid_batch_data[0]
                valid_y_true = valid_batch_data[1]
                valid_y_true = valid_y_true.to(torch.float32)

                # forward pass
                valid_y_pred = net(valid_x)

                # save batch metrics
                batch_valid_rmse_array.append(
                    rmse(valid_y_true, valid_y_pred.detach().cpu()))

        # record metrics at end of epoch
        epoch_train_loss = np.mean(batch_train_total_loss_array)
        epoch_train_rmse = np.mean(batch_train_rmse_array)
        train_mse_loss = np.mean(batch_train_mse_loss_array)
        train_reg_loss = np.mean(batch_train_reg_loss_array)
        epoch_val_rmse = np.mean(batch_valid_rmse_array)

        logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, '
                    f'train_total_loss: {epoch_train_loss:.4f}, '
                    f'train_mse_loss: {train_mse_loss:.4f}, '
                    f'train_reg_loss: {train_reg_loss:.4f}, '
                    f'train_rmse: {epoch_train_rmse:.4f}, '
                    f'val_rmse: {epoch_val_rmse:.4f}\n')
        train_writer.add_scalar('epoch_loss', epoch_train_loss, epoch)
        train_writer.add_scalar('epoch_mse_loss', train_mse_loss, epoch)
        train_writer.add_scalar('epoch_reg_loss', train_reg_loss, epoch)
        train_writer.add_scalar('epoch_rmse', epoch_train_rmse, epoch)
        valid_writer.add_scalar('epoch_rmse', epoch_val_rmse, epoch)

        if lowest_val_rmse > epoch_val_rmse:
            lowest_val_rmse = epoch_val_rmse

            save_dict = {'epoch': epoch,
                         'model_state_dict': net.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': epoch_train_loss}
            torch.save(save_dict, os.path.join(
                cfg['checkpoint_dir'], cfg['experiment_name']+'_best.pt'))

        # val_rmse regulates learning rate
        scheduler.step(epoch_val_rmse)

    return


# estimating test set performance
def evaluate(cfg, test_df, global_mean, user_rated_items_df):

    test_dataloader = get_dataloader(cfg, test_df, user_rated_items_df)

    net = get_model(cfg, global_mean).to(device)

    checkpoint = torch.load(os.path.join(
        cfg['checkpoint_dir'], cfg['experiment_name']+'_best.pt'), 
        map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    net.eval()
    with torch.no_grad():

        batch_test_rmse_array = []
        
        for test_batch_data in test_dataloader:  # loop over test batches
            
            test_x = test_batch_data[0]
            test_y_true = test_batch_data[1]
            test_y_true = test_y_true.to(torch.float32)

            # forward pass
            test_y_pred = net(test_x)

            # save batch metrics
            batch_test_rmse_array.append(
                rmse(test_y_true, test_y_pred.detach().cpu()))

    # display metrics
    epoch_test_rmse = np.mean(batch_test_rmse_array)
    logger.info('------------')
    logger.info(f'TEST RESULTS (RMSE): {epoch_test_rmse:.4f}')
    logger.info('------------')

    return


if __name__ == '__main__':

    # parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    with open(_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # create logger
    logger = get_logger(cfg)

    # print settings
    logger.info(f'cfg: {cfg}')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # load movielens 1M dataset
    train_df, valid_df, test_df, \
        global_mean, user_rated_items_df = load_data(cfg)

    if cfg['mode'] == 'train':
        train(cfg, train_df, valid_df, global_mean, user_rated_items_df)
    elif cfg['mode'] == 'evaluate':
        evaluate(cfg, test_df, global_mean, user_rated_items_df)
    elif cfg['mode'] == 'both':
        train(cfg, train_df, valid_df, global_mean, user_rated_items_df)
        evaluate(cfg, test_df, global_mean, user_rated_items_df)
    else:
        raise NotImplementedError()