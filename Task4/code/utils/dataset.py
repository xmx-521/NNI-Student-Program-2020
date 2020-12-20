from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import h5py


def preprocess(dataset,frame_dir):
    print("to be done")


class EVimageDataset(Dataset):

    def __init__(self,frame_dir,label_dir):

        self.frame_dir=frame_dir
        self.label_dir=label_dir
        self.ids = [splitext(file)[0] for file in listdir(frame_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')



    def __len__(self):
        return len(self.ids)



    def __getitem__(self, i):
        idx = self.ids[i]
        suffix_mark='*'
        frame_file=glob(self.frame_dir+idx+suffix_mark)
        label_file=glob(self.label_dir+idx+suffix_mark)

        #TODO:error process

        frame=preprocess(self,frame_dir)
        #TODO:label处理情况

        return{
            'image':torch.from_numpy(frame).type(torch.FloatTensor),
            'label':torch.from_numpy(label).type(torch.FloatTensor)
        }


