''' Configuration File.
'''

CUDA_VISIBLE_DEVICES = 0
NUM_TRAIN = 50000 # N \Fashion MNIST 60000, cifar 10/100 50000
# NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128 # B
SUBSET    = 10000 # M
ADDENDUM  = 1000 # K
START = 1000

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda

TRIALS = 3
CYCLES = 7

#refering to https://github.com/chengyangfu/pytorch-vgg-cifar10
EPOCH = 200
EPOCH_GCN = 200
LR = 5e-2
LR_GCN = 1e-3
MILESTONES = [30, 60, 90, 120, 150, 180, 210, 240, 270]
EPOCHL = 120#20 #120 # After 120 epochs, stop 
EPOCHV = 100 # VAAL number of epochs

MOMENTUM = 0.9
WDECAY = 5e-4#2e-3# 5e-4

NUM_CLASSES = 10

#for Noise Stability
NOISE_SCALE = 0.001
