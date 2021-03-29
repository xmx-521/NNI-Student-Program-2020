import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model import Net

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import EVimageDataset
from torch.utils.data import DataLoader

import nni

dir_img_train = 'data_after/train_dataset.hdf5'
dir_img_test = 'data_after/test_dataset.hdf5'
dir_checkpoint = 'checkpoints/'


def train_net(net, device, epochs=50, batch_size=64, lr=0.01, save_cp=True):

    train = EVimageDataset(dir_img_train)
    val = EVimageDataset(dir_img_test)
    n_train = len(train)
    n_val = len(val)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
    best_acc = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max',
                                                           patience=2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train,
                  desc=f'Epoch {epoch + 1}/{epochs}',
                  unit='img') as pbar:
            for batch in train_loader:
                imgs, labels = batch

                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device)

                output = net(imgs)
                loss = criterion(output, labels)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // batch_size) == 0:

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate',
                                      optimizer.param_groups[0]['lr'],
                                      global_step)

                    logging.info('Validation Acurracy: {}'.format(val_score))
                    writer.add_scalar('Acurracy/test', val_score, global_step)
                    if best_acc < val_score:
                        best_acc = val_score
                    nni.report_intermediate_result(val_score)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    nni.report_final_result(best_acc)
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(
        description='Tran the CNN on event images and labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e',
                        '--epochs',
                        metavar='E',
                        type=int,
                        default=50,
                        help='Number of epochs',
                        dest='epochs')
    parser.add_argument('-b',
                        '--batch-size',
                        metavar='B',
                        type=int,
                        nargs='?',
                        default=50,
                        help='Batch size',
                        dest='batchsize')
    parser.add_argument('-l',
                        '--learning-rate',
                        metavar='LR',
                        type=float,
                        nargs='?',
                        default=0.000003,
                        help='Learning rate',
                        dest='lr')
    parser.add_argument('-f',
                        '--load',
                        dest='load',
                        type=str,
                        default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    nni_args = nni.get_next_parameter()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = Net()
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=nni_args['batch_size'],
                  lr=nni_args['lr'],
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
