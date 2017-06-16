import numpy as np
import pickle as pkl
import gzip
import os, urllib

def _get_datafolder_path():
    '''
    Gets the path of the data directory.
    '''
    full_path = os.path.abspath('.')
    path = full_path +'/data'
    return path

def load_mnist_realval():
    '''
    Loads the real valued MNIST dataset
    :param dataset: path to dataset file
    :return: None
    '''
    dataset=_get_datafolder_path()+'/mnist_real/mnist.pkl.gz'
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_mnist_realval(dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f) # Returns a tuple. First element data and second label.
    f.close()
    x_train, targets_train = train_set[0], train_set[1]
    x_valid, targets_valid = valid_set[0], valid_set[1]
    x_test, targets_test = test_set[0], test_set[1]
    return x_train, targets_train, x_valid, targets_valid, x_test, targets_test

def _download_mnist_realval(dataset):
    """
    Download the MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    origin = (
        'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset)
    
# def generate_synthetic_data():
     
    