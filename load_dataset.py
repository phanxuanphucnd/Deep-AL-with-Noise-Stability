import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import *
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, MNIST, SVHN
import pandas as pd
import torch

def get_csv_data(datadir):
    #get train data
    train_data_path = os.path.join(datadir, 'train.csv')
    train = pd.read_csv(train_data_path)
    #get test data
    test_data_path = os.path.join(datadir, 'test.csv')
    test = pd.read_csv(test_data_path)
    return train , test

def get_combined_data(datadir):
  #reading train data
  train , test = get_csv_data(datadir)

  target = train.SalePrice
  train.drop(['SalePrice'],axis = 1 , inplace = True)

  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop(['index', 'Id'], inplace=True, axis=1)
  return combined, target


def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df

#https://colab.research.google.com/drive/1J8ZTI2UIJCwml2nrLVu8Gg0GXEz-7ZK0#scrollTo=Ng_BSoDwrr3G
def load_house_dataset(datadir):
    #Load train and test data into pandas DataFrames
    train_data, _ = get_csv_data(datadir)
    target = train_data.SalePrice
    train_data.drop(['SalePrice'], axis = 1, inplace = True)
    train_data.drop(['Id'], inplace=True, axis=1)
    
    num_cols = get_cols_with_no_nans(train_data, 'num')
    cat_cols = get_cols_with_no_nans(train_data, 'no_num')
    print ('Number of numerical columns with no nan values :',len(num_cols))
    print ('Number of nun-numerical columns with no nan values :',len(cat_cols))
    train_data = train_data[num_cols + cat_cols]
    print('There were {} columns before encoding categorical features'.format(train_data.shape[1]))
    train_data = oneHotEncode(train_data, cat_cols)
    print('There are {} columns after encoding categorical features'.format(train_data.shape[1]))

    num_train = len(train_data) // 2
    trainX = train_data[:num_train]
    trainY = target[:num_train]
    testX = train_data[num_train:]
    testY = target[num_train:]

    #print(trainX)
    #print(trainY)
    #print(testX)
    #print(testY)

    trainX = torch.from_numpy(trainX.to_numpy()).float()
    trainY = torch.from_numpy(trainY.to_numpy()).float()
    testX = torch.from_numpy(testX.to_numpy()).float()
    testY = torch.from_numpy(testY.to_numpy()).float()

    #print(trainX.shape, trainX[:5,:5], trainY.shape, trainY[:5])
    #print(testX.shape, testX[:5,:5], testY.shape, testY[:5])
    return trainX, trainY, testX, testY


class RegressionDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf=None):
        trainX, trainY, textX, testY = load_house_dataset('./benchmark/house_regression')
        if train_flag:
            self.data = trainX, trainY
        else:
            self.data = textX, testY
        print('train:', train_flag, 'data_len', self.__len__())

    def __getitem__(self, index):
        X, Y = self.data
        return X[index], Y[index]

    def __len__(self):
        return len(self.data[1])

class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf):
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.mnist = MNIST('./benchmark', train=train_flag, 
                                download=True, transform=transf)
        elif self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('./benchmark', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100('./benchmark', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "fashionmnist":
            self.fmnist = FashionMNIST('../fashionMNIST', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "svhn":
            self.svhn = SVHN('./benchmark', split="train", 
                                    download=True, transform=transf)
        if self.dataset_name == "house":
            self.house = RegressionDataset('house', True)

    def __getitem__(self, index):
        if self.dataset_name == "mnist":
            data, target = self.mnist[index]
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        if self.dataset_name == "cifar100":
            data, target = self.cifar100[index]
        if self.dataset_name == "fashionmnist":
            data, target = self.fmnist[index]
        if self.dataset_name == "svhn":
            data, target = self.svhn[index]
        if self.dataset_name == "house":
            data, target = self.house[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "fashionmnist":
            return len(self.fmnist)
        elif self.dataset_name == "svhn":
            return len(self.svhn)
        elif self.dataset_name == "mnist":
            return len(self.mnist)
        elif self.dataset_name == "house":
            return len(self.house)
##

data_mean = {'house':[0], 'mnist': [0.1307], 'cifar10': [0.4914, 0.4822, 0.4465], 'cifar100': [0.5071, 0.4867, 0.4408], 'svhn': [0.4310, 0.4302, 0.4463]}
data_std = {'house':[1], 'mnist': [0.3081], 'cifar10': [0.2023, 0.1994, 0.2010], 'cifar100': [0.2675, 0.2565, 0.2761], 'svhn': [0.1965, 0.1984, 0.1992]}

# Data
def load_dataset(dataset):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize(data_mean[dataset], data_std[dataset]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(data_mean[dataset], data_std[dataset]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])


    if dataset == 'cifar10': 
        data_train = CIFAR10('./benchmark', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR10('./benchmark', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        no_train = NUM_TRAIN
    elif dataset == 'cifar10im': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        #data_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
        targets = np.array(data_train.targets)
        #NUM_TRAIN = targets.shape[0]
        classes, _ = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = [500, 5000] * 5
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        # print(NUM_TRAIN)
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        data_unlabeled = MyDataset(dataset[:-2], True, test_transform)
        data_unlabeled.cifar10.targets = targets[imb_class_idx]
        data_unlabeled.cifar10.data = data_unlabeled.cifar10.data[imb_class_idx]
        data_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        no_train = NUM_TRAIN
    elif dataset == 'cifar100':
        data_train = CIFAR100('./benchmark', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR100('./benchmark', train=False, download=True, transform=test_transform)
        NO_CLASSES = 100
        no_train = NUM_TRAIN
    elif dataset == 'mnist':
        data_train = MNIST('./benchmark', train=True, download=True, 
                                    transform=T.Compose([T.ToTensor(), T.Normalize(data_mean[dataset], data_std[dataset])]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor(), T.Normalize(data_mean[dataset], data_std[dataset])]))
        data_test  = MNIST('./benchmark', train=False, download=True, 
                                    transform=T.Compose([T.ToTensor(), T.Normalize(data_mean[dataset], data_std[dataset])]))
        NO_CLASSES = 10
        no_train = NUM_TRAIN
    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../fashionMNIST', train=True, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = FashionMNIST('../fashionMNIST', train=False, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        no_train = NUM_TRAIN
    elif dataset == 'svhn':
        data_train = SVHN('./benchmark', split='train', download=True, 
                                    transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = SVHN('./benchmark', split='test', download=True, 
                                    transform=test_transform)
        NO_CLASSES = 10
        no_train = NUM_TRAIN
    elif dataset == 'house':
        data_train = RegressionDataset(dataset, True)
        data_unlabeled = MyDataset(dataset, True, None)
        data_test = RegressionDataset(dataset, False)
        NO_CLASSES = 1
        no_train = NUM_TRAIN
    return data_train, data_unlabeled, data_test, 0, NO_CLASSES, no_train
