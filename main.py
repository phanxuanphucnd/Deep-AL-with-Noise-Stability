'''
modified from GCN Active Learning: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
'''

# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
import time
# Custom
import backbone 
from query_models import LossNet
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *


parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="cifar100",
                    help="")
parser.add_argument("-m","--method_type", type=str, default="lloss",
                    help="")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-k", "--n_sampling", type=int, default=50,
                    help="Number of sampling for deep Bayesian or Noise Stability")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
print(args)

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def select_init_set(data_train):
    indices = list(range(NUM_TRAIN))
    for _ in range(100000):
        random.shuffle(indices)
        label_cnt = [0] * NO_CLASSES
        for idx in indices[:START]:
            label = data_train[idx][1] 
            label_cnt[label] += 1
        if min(label_cnt) > 0:
            break
    return indices
##
# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['Random', 'BALD', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL', 'Entropy', 'EntropyBayesian', 'BatchBALD', 'BADGE', 'NoiseStability']
    datasets = ['cifar10', 'cifar100','svhn', 'mnist', 'house']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    
    results = open('results_'+str(args.method_type)+"_"+args.dataset +'_main'+str(CYCLES)+str(args.total)+'.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    for trial in range(TRIALS):
        seed_torch(args.seed+trial)
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, _, NO_CLASSES, no_train = load_dataset(args.dataset)
        NUM_TRAIN = no_train
        if args.dataset == 'mnist':
            indices = select_init_set(data_train)
        else:
            indices = list(range(NUM_TRAIN))
            random.shuffle(indices)

        if args.total:
            labeled_set= indices
        else:
            labeled_set = indices[:START]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH, 
                                    sampler=SubsetRandomSampler(labeled_set), 
                                    pin_memory=True, drop_last=(START>BATCH))
        test_loader  = DataLoader(data_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}

        for cycle in range(CYCLES):
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                if SUBSET < 0:
                    subset = unlabeled_set
                else:
                    subset = unlabeled_set[:SUBSET]

            train_repeated = 1
            acc_list = []
            for r in range(train_repeated):
                seed = int(args.seed*1e3+trial*1e2+cycle*1e1+r)
                seed_torch(seed)
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    if args.dataset == "fashionmnist" or args.dataset == 'mnist':
                        resnet18 = backbone.SmallNet(num_classes=NO_CLASSES).cuda()
                    elif args.dataset == 'house':
                        resnet18 = backbone.MLP_Regression(215).cuda()
                    else:
                        resnet18    = backbone.ResNet18(num_classes=NO_CLASSES).cuda()
                        #resnet18    = backbone.vgg16().cuda() 
                        #resnet18    = mobilenet.MobileNetV2(num_classes=NO_CLASSES).cuda()
                    if method == 'lloss':
                        loss_module = LossNet().cuda()
                        #loss_module = LossNet(feature_sizes=[32,16,8,4], num_channels=[64, 128, 256, 1024]).cuda()

                models      = {'backbone': resnet18}
                if method =='lloss':
                    models = {'backbone': resnet18, 'module': loss_module}
                torch.backends.cudnn.benchmark = True
                
                # Loss, criterion and scheduler (re)initialization
                if args.dataset in ['house']:
                    criterion      = nn.MSELoss(reduction='none')
                    optim_backbone = optim.Adam(models['backbone'].parameters(), lr=LR) 
                elif args.dataset in ['mnist']:
                    criterion      = nn.CrossEntropyLoss(reduction='none')
                    optim_backbone = optim.Adam(models['backbone'].parameters(), lr=LR) 
                else:
                    criterion      = nn.CrossEntropyLoss(reduction='none')
                    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                        momentum=MOMENTUM, weight_decay=WDECAY)

                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES, gamma=0.1) #should be 0.5 if use VGG
                optimizers = {'backbone': optim_backbone}
                schedulers = {'backbone': sched_backbone}
                if method == 'lloss':
                    optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                        momentum=MOMENTUM, weight_decay=WDECAY)
                    sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                    optimizers = {'backbone': optim_backbone, 'module': optim_module}
                    schedulers = {'backbone': sched_backbone, 'module': sched_module}
                
                # Training and testing
                class_balance = (args.dataset not in ['house']) and (len(dataloaders['train']) < 2)
                train(models, method, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, args.dataset in ['house'], class_balance)
                acc, loss = test(models, criterion, method, dataloaders, mode='test', regression=(args.dataset in ['house']))
                acc_list.append(acc)
            acc = sum(acc_list) / len(acc_list)
            if args.dataset == 'house':
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test loss {:.6f}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), loss))
            else:
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")

            if cycle == (CYCLES-1):
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            start_time = time.time()
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args, adden=ADDENDUM)
            last_time = 1.0 * (time.time() - start_time) / 60
            print('data selection time %.2f min' % last_time)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())

            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
            if SUBSET < 0:
                unlabeled_set -= listd
            else:
                unlabeled_set = listd + unlabeled_set[SUBSET:]

            #avoid very small batches
            batch_size = BATCH
            if len(labeled_set) > BATCH and len(labeled_set) < BATCH + 16:
                batch_size = BATCH - 16
            elif len(labeled_set) > 2*BATCH and len(labeled_set) < 2*BATCH + 16:
                batch_size = BATCH - 8
            elif len(labeled_set) > 3*BATCH and len(labeled_set) < 3*BATCH + 16:
                batch_size = BATCH - 6
            elif len(labeled_set) > 4*BATCH and len(labeled_set) < 4*BATCH + 16:
                batch_size = BATCH - 4
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH, 
                                            sampler=SubsetRandomSampler(labeled_set), 
                                            pin_memory=True)

    results.close()
