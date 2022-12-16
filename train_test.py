from config import *
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import time
##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def test(models, criterion, method, dataloaders, mode='val', regression=False):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            if regression:
                target_loss = criterion(scores.squeeze(1), labels)
            else:
                target_loss = criterion(scores, labels)
            target_loss = torch.sum(target_loss) / target_loss.size(0)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += torch.sqrt(target_loss).item()
    total_loss /= NUM_TRAIN
    return 100 * correct / total, total_loss


iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, regression, class_balance):

    start_time = time.time()
    models['backbone'].train()
    if method == 'lloss':
        models['module'].train()
    global iters

    if class_balance:
        label_cnt = [0] * NUM_CLASSES
        for data in dataloaders['train']:
            labels = data[1]
            for i in range(labels.shape[0]):
                label_cnt[labels[i]] += 1
        class_weight = 1.0 / np.array(label_cnt)

    for data in dataloaders['train']:
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            labels = data[1].cuda()

        iters += 1
        if class_balance:
            sample_weight = torch.zeros(labels.shape)
            for i in range(labels.shape[0]):
                sample_weight[i] = class_weight[labels[i]]
            sample_weight = sample_weight.cuda()

        optimizers['backbone'].zero_grad()
        if method == 'lloss':
            optimizers['module'].zero_grad()

        scores, _, features = models['backbone'](inputs)
        if regression:
            target_loss = criterion(scores.squeeze(1), labels)
        else:
            target_loss = criterion(scores, labels)

        if method == 'lloss':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 
        elif class_balance:
            #sample_weight = torch.ones(inputs.shape[0]).cuda()
            loss = torch.sum(target_loss * sample_weight).mean()
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss

        #print('iter', iters, 'loss', loss.item())
        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss':
            optimizers['module'].step()
    end_time = time.time()
    print('epoch %d time %.1f min' % (epoch, 1.0 * (end_time - start_time) / 60))
    return loss

def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, regression=False, balance_class=False):
    global iters
    iters = 0
    print('>> Train a Model.')
    best_acc = 0.
    
    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, regression, balance_class)

        schedulers['backbone'].step()
        if method == 'lloss':
            schedulers['module'].step()

    print('>> Finished.')
