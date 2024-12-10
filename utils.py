'''
Contains functions used for Data processing and loading of the MNISt dataset
'''

import numpy as np
import struct
from array import array
from tqdm import tqdm

class MnistDataloader:
    def __init__(self, train_paths, test_paths, val_split=0.1, shuffle = False, batch_size = 8):
        """
        Initialize the MNISTDataLoader.

        Parameters:
            train_path (str): Path to the training CSV file.
            test_path (str): Path to the testing CSV file.
            val_split (float): Proportion of training data to use for validation.
        """
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.val_split = val_split
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data = {}
        

    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in tqdm(range(size)):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels

    def load_data(self):
        """
        Load the training and testing data from the CSV files 
        Split the training data into training and validation sets.
        """
        # Load the training and testing data
        trainX, trainY = self.read_images_labels(self.train_paths[0], self.train_paths[1])    
        testX, testY = self.read_images_labels(self.test_paths[0], self.test_paths[1])

        # Split the training data into training and validation sets
        num_val = int(len(trainX) * self.val_split)
        valdiationX = trainX[-num_val:]
        valdiationY = trainY[-num_val:]

        self.train_data = np.array(trainX[:-num_val]), np.array(trainY[:-num_val])

        if self.shuffle:
            idx = np.random.permutation(len(self.train_data[0]))
            self.train_data = self.train_data[0][idx], self.train_data[1][idx]

        self.validation_data = np.array(valdiationX), np.array(valdiationY)
        self.test_data = np.array(testX), np.array(testY)
        return self.train_data, self.validation_data, self.test_data

    def get_batch(self, data,labels, batch_size):
        """
        Get a batch of data.

        Parameters:
            data (tuple): Tuple of data and labels.
            batch_size (int): Size of the batch.

        Returns:
            tuple: Tuple of data and labels.
        """
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size], labels[i:i + batch_size]