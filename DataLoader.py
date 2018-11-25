import random
import numpy as np
import h5py
import scipy.ndimage
import glob
import os
from utils.data_utils import generate_data
from sklearn.model_selection import train_test_split


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        project_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(project_path + cfg.data_dir):
            os.makedirs(cfg.data_dir)
            generate_data()
        self.data_file = project_path + '/data/zillow_data.h5'
        if cfg.normalize:
            self.get_stats()
        self.get_data(mode=cfg.mode)

    def next_batch(self, start, end, mode='train'):
        if mode == 'train':
            x = self.x_train[start:end]
            y = self.y_train[start:end]
            return x, y
        elif mode == 'valid':
            x = self.x_valid[start:end]
            y = self.y_valid[start:end]
            return x, y
        else:
            x = self.x_test[start:end]
            return x

    def get_data(self, mode='train'):
        if mode == 'train':
            print('counting the number of train samples........')
            h5f = h5py.File(self.data_file, 'r')
            x = h5f['X_train'][:]
            y = h5f['y_train'][:]
            h5f.close()
            if self.cfg.normalize:
                x = normalize(x, self.input_mean, self.input_std)
                y = normalize(y, self.output_mean, self.output_std)
            self.x_train, self.x_valid, self.y_train, self.y_valid = \
                train_test_split(x, y, test_size=0.2, random_state=1337)
            self.num_tr = self.y_train.shape[0]
            self.num_val = self.y_valid.shape[0]
        else:
            h5f = h5py.File(self.data_file, 'r')
            self.x_test = h5f['X_test'][:]
            h5f.close()
            self.num_te = self.x_test.shape[0]
            if self.cfg.normalize:
                self.x_test = normalize(self.x_test, self.input_mean, self.input_std)

    def get_stats(self):
        h5f = h5py.File(self.data_file, 'r')
        x = h5f['X_train'][:]
        y = h5f['y_train'][:]
        h5f.close()
        self.input_mean = np.mean(x, axis=0)
        self.input_std = np.std(x, axis=0)
        self.output_mean = np.mean(y)
        self.output_std = np.std(y)

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        self.x_train = self.x_train[permutation, :]
        self.y_train = self.y_train[permutation]

def normalize(data, mean, std):
    return (data - mean) / std


def denormalize(data, mean, std):
    return (data * std) + mean
