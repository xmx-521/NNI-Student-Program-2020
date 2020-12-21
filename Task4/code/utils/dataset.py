from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import h5py


def preprocess(EVlimageDataset,train_or_test):
    #train_or_test = 'test'
    # set it to train/test to generate train/test_dataset.hfd5 files
    f = h5py.File(train_or_test + "_dataset.hfd5", "w")
    data_path = './' + train_or_test + '/'

    # convert raw data to hfd5 file
    for label in range(21):
        grp = f.create_group(str(label), "w")
        for i in range(100):
            dataset = grp.create_dataset(str(i + 1), (128, 128, 2), dtype='i')
            _data = np.loadtxt(data_path + str(label) + '/' + str(i + 1) + '.txt')
            for j in range(len(_data)):
                if _data[i, 3] == 0:
                    dataset[_data[i, 1], _data[i, 2], 0] += 1
                else:
                    dataset[_data[i, 1], _data[i, 2], 1] += 1


class EVimageDataset(Dataset):

    def __init__(self,img_dir,train_or_test):
        self.img_dir=img_dir
        self.train_or_test=train_or_test
        self.ids=[splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):

        '''


        idx = self.ids[i]
        suffix_mark='*'
        frame_file=glob(self.frame_dir+idx+suffix_mark)
        label_file=glob(self.label_dir+idx+suffix_mark)


        '''

        return{
            'image':torch.from_numpy(frame).type(torch.FloatTensor),
            'label':torch.from_numpy(label).type(torch.FloatTensor)
        }