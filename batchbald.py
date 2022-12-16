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


import math
from toma import toma
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List
from batchbald_redux import joint_entropy


def predict_prob_dropout_split(models, unlabeled_loader, args, log_softmax=False):
    models['backbone'].train()
    models['backbone'].set_dropout(True, args.dropout_rate)
    
    n_drop = args.n_sampling
    probs = torch.zeros([n_drop, SUBSET, NUM_CLASSES])
    probs = probs.cuda()
    iters = 0
    for i in range(n_drop):
        start_time = time.time()
        with torch.no_grad():
            idx_start = 0
            for x, _, _ in unlabeled_loader:
                idxs = np.arange(idx_start, idx_start + x.shape[0])
                idx_start += x.shape[0]
                #print(idxs)
                if iters % 10 == 0:
                    print('dropout sampling %d/%d' % (iters, n_drop*len(unlabeled_loader)))
                iters += 1
                x = x.cuda()
                out = models['backbone'](x)[0]
                if log_softmax:
                    probs[i][idxs] += F.log_softmax(out, dim=1)
                else:
                    probs[i][idxs] += F.softmax(out, dim=1)
        last_time = 1.0 * (time.time() - start_time) / 60
        print('dropout pass %d/%d finished, time %.2f min' % (i, n_drop, last_time))
        
    models['backbone'].set_dropout(False, 0)
    return probs
    

def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N

# Cell


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def batchbald_sampling(models, unlabeled_loader, args):
    batch_num = min(ADDENDUM, SUBSET)
    num_samples = 100000 #len(idxs_unlabeled)
    batch_size = batch_num
    N = SUBSET
    K = args.n_sampling
    C = NUM_CLASSES

    log_probs_N_K_C = predict_prob_dropout_split(models, unlabeled_loader, args, True)
    log_probs_N_K_C = log_probs_N_K_C.permute(1,0,2)
    
    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)        
    #print(log_probs_N_K_C.shape, num_samples, batch_size - 1, K, C)
    #print(log_probs_N_K_C[:5,:5,:5], conditional_entropies_N[:5])
    
    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
            num_samples, batch_size - 1, K, C, dtype=torch.double, device='cuda:0')
        
    candidate_indices = []
    candidate_scores = []
    U = torch.empty(SUBSET, dtype=torch.double)

    for i in range(batch_num):
        start_time = time.time()
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditional_entropies = conditional_entropies_N[candidate_indices].sum()
        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=U)

        U -= conditional_entropies_N + shared_conditional_entropies
        U[candidate_indices] = -float('inf')
        #print('score', i, U)
        candidate_score, candidate_index = U.max(dim=0)
        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())
        last_time = 1.0 * (time.time() - start_time) / 60
        print('BatchBALD %d/%d finished, time %.2f min' % (i, batch_num, last_time))
   
    chosen = candidate_indices
    arg = []
    for x in range(SUBSET):
        if not x in chosen:
            arg.append(x)
    arg += chosen
    assert len(arg) == SUBSET and len(chosen) == ADDENDUM
    return np.array(arg)

