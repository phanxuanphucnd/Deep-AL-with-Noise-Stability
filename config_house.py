''' Configuration File.
'''

CUDA_VISIBLE_DEVICES = 0
NUM_TRAIN = 730 # N \Fashion MNIST 60000, cifar 10/100 50000
# NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 64 # B
SUBSET    = 200 # M
ADDENDUM  = 20 # K
START = 20

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda

TRIALS = 100
CYCLES = 10

EPOCH = 500
EPOCH_GCN = 200
LR = 1e-3
LR_GCN = 1e-3
MILESTONES = [160]
EPOCHL = 120#20 #120 # After 120 epochs, stop 
EPOCHV = 100 # VAAL number of epochs

MOMENTUM = 0.9
WDECAY = 5e-4#2e-3# 5e-4

NUM_CLASSES = 1

#for Noise Stability
NOISE_SCALE = 0.001
