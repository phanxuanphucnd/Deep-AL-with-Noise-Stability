import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# Custom
from config import *
from query_models import VAE, Discriminator, GCN
from sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
import torch.nn.functional as F 
from torch.distributions import Categorical
import time

import copy
from scipy import stats
from sklearn.metrics import pairwise_distances
# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def badge_sampling(models, unlabeled_loader, args):
    models['backbone'].eval()
    embDim = models['backbone'].get_embedding_dim()
    embedding = np.zeros([SUBSET, embDim * NUM_CLASSES])

    with torch.no_grad():
        idx = 0
        for inputs, labels, _ in unlabeled_loader:
            x, y = inputs.cuda(), labels.cuda()
            cout, out, _ = models['backbone'](x)
            out = out.data.cpu().numpy()
            batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
            maxInds = np.argmax(batchProbs,1)
            for j in range(len(y)):
                for c in range(NUM_CLASSES):
                    if c == maxInds[j]:
                        embedding[idx][embDim * c : embDim * (c+1)] = copy.deepcopy(out[j]) * (1 - batchProbs[j][c])
                    else:
                        embedding[idx][embDim * c : embDim * (c+1)] = copy.deepcopy(out[j]) * (-1 * batchProbs[j][c])
                idx += 1

    chosen = init_centers(embedding, ADDENDUM)
    arg = [0] * (SUBSET - ADDENDUM) + chosen
    assert len(arg) == SUBSET
    return np.array(arg)
