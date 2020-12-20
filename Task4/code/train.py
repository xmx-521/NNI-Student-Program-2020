

import os
import time
import numpy as np
import argparse
import logging
import torch
import torch.nn as nn
import matplotlib.image as mpimg
from torch import optim
from torch.utils.data import DataLoader, random_split

# 提供的网络类和数据集类
from evnet import EVnet
from dataset import EVimageDataset

# 可视化
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 这里记得改路径
dir_train_img = ""
dir_train_truth = ""
dir_test_img = ""
dir_test_truch=""


def train(net, device, epochs=6, batch_size=32, learning_rate=0.0001, save_cp=True):
    #装载数据集
    train_dataset = EvimageDataset(dir_train_img, dir_train_truth)
    n_train = int(len(train_dataset))
    
    test_dataset = EvimageDataset(dir_test_img, dir_test_truth)
    n_test= int(len(test_dataset))

    # num_workers是进程数，跑不了可以改一改
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    global_step=0
    # tensorboard记录一下参数(后面写)
    writer=SummaryWriter(comment=f'LR_{learning_rate}_BS_{batch_size}')
    # log记录一下
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_test}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,betas=(0.9,0,999),eps=1e-8,weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer，'max', patience=2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                labels = batch['label']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device)

                output = net(imgs)
                loss = criterion(output, labels)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss=0
        # 进度条可视化
        with tqdm(total=n_train, desc=f'Epoch{epoch}/{epochs}', unit='img') as pbar:
        
            for batch_idx, data in enumerate(train_loader):
                imgs, truth = data['image'], data['label']
                imgs, truth= imgs.to(device), truth.to(device)
                output = net(imgs)
                loss = nn.functional.cross_entropy(output, truth)
                epoch_loss += loss
                
                # loss可视化
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                # 可视化
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # 测试
                    val_score = eval_net(net, test_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    writer.add_images('images', imgs, global_step)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')



def get_args():
    parser = argparse.ArgumentParser(description='Train the EVnet on images and truth',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = Net()
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,device=device,epochs=args.epochs,batch_size=args.batchsize,learning_rate=args.lr,)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
