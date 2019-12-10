# Load 32 x 32 imagenet images and convert them into grayscale

from skimage.color import rgb2gray

import matplotlib.pyplot as plt
import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# We want to :
# load all the data. batch 1 ~ 10

if __name__ == '__main__':

    for n in range(1,11):
        tdb1 = unpickle('./Imagenet32_train/train_data_batch_' + str(n))
        data = tdb1['data']
        grayscale_batch = np.zeros((np.shape(data)[0], 32, 32))
        for m in range(np.shape(data)[0]):
            image = np.zeros((32,32,3))
            image[:,:,0] = np.reshape(data[m, 0:1024], (32,32))
            image[:,:,1] = np.reshape(data[m, 1024:2048], (32,32))
            image[:,:,2] = np.reshape(data[m, 2048:3072],(32,32))
            grayscale_batch[m,:,:,] = rgb2gray(image/255)
        np.save('imagenet32_batch_'+str(n)+'.npy',grayscale_batch)
        print('Another batch done! ')

