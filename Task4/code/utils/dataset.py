from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import h5py
import tqdm
import numpy as np


class EVimageDataset(Dataset):

    def __init__(self, path):
        # input:the path of hdf5 file
        self.path = path

        # TODO:get dataset len

    def __len__(self):
        return 2100

    def __getitem__(self, i):
        # input an index of 0-2099,representing
        person_index = (int)(i/100)

        # get person label tensor
        label = torch.tensor(person_index, dtype=torch.int64)

        # open hdf5 file
        f1 = h5py.File(self.path, 'r')

        # get group index
        grp = f1[str(person_index)]

        # get dataset with image_index
        image_index = i % 100+1
        dset = grp[str(image_index)]

        # get dataset tensor
        image = dset[()]

        return image, label


# dataset run info
# dataset = EVimageDataset("../data_after/train_dataset.hdf5")
# image, label = dataset.__getitem__(300)
# print(label)
# print(image[1, 127, 23])
