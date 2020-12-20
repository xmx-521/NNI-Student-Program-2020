import numpy as np
import h5py

train_or_test = 'test' #set it to train/test to generate train/test_dataset.hfd5 files
f = h5py.File(train_or_test + "_dataset.hfd5", "w")
data_path = './'+ train_or_test+ '/'

#convert raw data to hfd5 file
for label in range(21):
    grp = f.create_group(str(label),"w")
    for i in range(100):
        dataset = grp.create_dataset(str(i+1), (128,128,2), dtype = 'i')
        _data = np.loadtxt(data_path + str(label) + '/' + str(i+1) + '.txt')
        for j in range(len(_data)):
            if _data[i, 3] == 0:
                dataset[_data[i, 1],_data[i, 2], 0] += 1
            else:
                dataset[_data[i, 1], _data[i, 2], 1] += 1
