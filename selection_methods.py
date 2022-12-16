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
import backbone
import torch.nn.functional as F 
from torch.distributions import Categorical
import time

from batchbald import batchbald_sampling
from badge import badge_sampling
from noise_stability import noise_stability_sampling

def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    
    vae = models['vae']
    discriminator = models['discriminator']
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
    
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int( (ADDENDUM*cycle+ SUBSET) * EPOCHV / BATCH )

    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
            if iter_count % 100 == 0:
                print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
                
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, _, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()    
    with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    _, features_batch, _ = models['backbone'](inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features #.detach().cpu().numpy()
    return feat

def get_kcg(models, labeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(SUBSET,(SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(SUBSET) if x not in batch]
    return  other_idx + batch


# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args, adden):
    if method == 'Random':
        #arg = np.random.randint(SUBSET, size=SUBSET)
        arg = np.arange(SUBSET)
        np.random.shuffle(arg)

    if (method == 'UncertainGCN') or (method == 'CoreGCN'):
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)
        binary_labels = torch.cat((torch.zeros([SUBSET, 1]),(torch.ones([len(labeled_set),1]))),0)


        features = get_features(model, unlabeled_loader)
        features = nn.functional.normalize(features)
        adj = aff_to_adj(features)

        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=args.hidden_units,
                         nclass=1,
                         dropout=args.dropout_rate).cuda()
                                
        models      = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(SUBSET, SUBSET+(cycle+1)*ADDENDUM, 1)
        nlbl = np.arange(0, SUBSET, 1)
        
        ############
        for _ in range(200):

            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = args.lambda_loss
            #print(outputs.shape, lbl[0], lbl[-1])
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()


        models['gcn_module'].eval()
        with torch.no_grad():
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = features.cuda()
                labels = binary_labels.cuda()
            scores, _, feat = models['gcn_module'](inputs, adj)
            
            if method == "CoreGCN":
                feat = feat.detach().cpu().numpy()
                new_av_idx = np.arange(SUBSET,(SUBSET + (cycle+1)*ADDENDUM))
                sampling2 = kCenterGreedy(feat)  
                batch2 = sampling2.select_batch_(new_av_idx, ADDENDUM)
                other_idx = [x for x in range(SUBSET) if x not in batch2]
                arg = other_idx + batch2
                
                arg_new = [x for x in arg if x < len(subset)]
                arg = arg_new

            else:

                s_margin = args.s_margin 
                scores_median = np.squeeze(torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())
                arg = np.argsort(-(scores_median))

            print("Max confidence value: ",torch.max(scores.data))
            print("Mean confidence value: ",torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[SUBSET:,0] == labels[SUBSET:,0]).sum().item() / ((cycle+1)*ADDENDUM)
            correct_unlabeled = (preds[:SUBSET,0] == labels[:SUBSET,0]).sum().item() / SUBSET
            correct = (preds[:,0] == labels[:,0]).sum().item() / (SUBSET + (cycle+1)*ADDENDUM)
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct)
    
    if method == 'CoreSet':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)

        arg = get_kcg(model, ADDENDUM*(cycle+1), unlabeled_loader)
        arg_new = [x for x in arg if x < len(subset)]
        arg = arg_new

    if method == 'lloss':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)        

    if method == 'VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True)
        if args.dataset == 'fashionmnist':
            vae = VAE(28,1,3)
            discriminator = Discriminator(28)
        else:
            vae = VAE()
            discriminator = Discriminator(32)
        models      = {'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models,optimizers, labeled_loader, unlabeled_loader, cycle+1)
        
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:                       
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds) 
    
    
    if method == 'Entropy':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty_entropy(model, unlabeled_loader, args)
        arg = np.argsort(uncertainty)        
    
    if method == 'EntropyBayesian':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_entropy_mcdropout(model, unlabeled_loader, args)
        print(uncertainty.shape)
        arg = np.argsort(uncertainty)        
    
    if method == 'NoiseStability':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = noise_stability_sampling(model, unlabeled_loader, args)
        #uncertainty = uncertainty / entropy

        arg = np.argsort(uncertainty)

    if method == 'BALD':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty_bald(model, unlabeled_loader, args)
        print(uncertainty.shape)
        arg = np.argsort(uncertainty)
    
    if method == 'BatchBALD':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)

        arg = batchbald_sampling(model, unlabeled_loader, args)

    if method == 'BADGE':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=True)

        arg = badge_sampling(model, unlabeled_loader, args) #last ADDENDUM indexes will be selected 

    #print(arg.shape)        
    return arg

def get_entropy_mcdropout(models, unlabeled_loader, args):
    models['backbone'].train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).cuda()
    models['backbone'].set_dropout(True, args.dropout_rate)

    iters = 0
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            if iters % 10 == 0:
                print('mcdropout %d/%d' % (iters, len(unlabeled_loader)))
            iters += 1
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            entropy = torch.zeros(inputs.shape[0]).cuda()
            for i in range(args.n_sampling):
                out, _, _ = models['backbone'](inputs)
                out = F.softmax(out, dim = 1)
                if iters == 1:
                    print(i, out[0])
                ent = Categorical(probs = out).entropy()
                ent = ent.view(ent.size(0))
                entropy += ent
                #print(entropy.shape)
            uncertainty = torch.cat((uncertainty, entropy), 0)
            print(uncertainty.shape)
    
    models['backbone'].set_dropout(False, 0)
    
    return uncertainty.cpu()

def get_uncertainty_entropy(models, unlabeled_loader, args):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).cuda()
    #models['backbone'].set_dropout(False, 0)

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            out, _, _ = models['backbone'](inputs)
            out = F.softmax(out, dim = 1)
            entropy = Categorical(probs = out).entropy()
            entropy = entropy.view(entropy.size(0))
            uncertainty = torch.cat((uncertainty, entropy), 0)
    
    #models['backbone'].set_dropout(False, 0)
    return uncertainty.cpu()

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
    
def get_uncertainty_bald(models, unlabeled_loader, args):
    probs = predict_prob_dropout_split(models, unlabeled_loader, args, False)
    pb = probs.mean(0)
    entropy1 = (-pb*torch.log(pb)).sum(1)
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    U = entropy1 - entropy2
    #print(U)
    return U.cpu()

