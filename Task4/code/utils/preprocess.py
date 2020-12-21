import numpy as np
import h5py
from tqdm import tqdm




def preprocess(path, f):
    for label in tqdm(range(21)):
        grp = f.create_group(str(label), "w")
        for i in tqdm(range(1, 101)):
            image = np.zeros([128, 128, 2])
            _data = np.loadtxt(path + str(label) + '/' + str(i) + '.txt')
            for j in range(len(_data)):
                if _data[j, 3] == 1:
                    image[(int)(_data[j, 1]), (int)(_data[j, 2]), 0] += 1
                else:
                    image[(int)(_data[j, 1]), (int)(_data[j, 2]), 1] += 1
            grp.create_dataset(name=str(i), data=image)

'''
print("making the train set")
preprocess(train_path, f1)
print("making the test set")
preprocess(test_path, f2)

f1.close()
f2.close()
'''