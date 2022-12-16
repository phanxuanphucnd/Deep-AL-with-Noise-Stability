''' Configuration File.
'''

CUDA_VISIBLE_DEVICES = 0
NUM_TRAIN = 60000 # N \Fashion MNIST 60000, cifar 10/100 50000
# NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 96 # B
SUBSET    = 1000 # M
ADDENDUM  = 20 # K
START = 20

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda

TRIALS = 10
CYCLES = 10

EPOCH = 50
EPOCH_GCN = 200
LR = 1e-3
LR_GCN = 1e-3
MILESTONES = [20]
EPOCHL = 120#20 #120 # After 120 epochs, stop 
EPOCHV = 100 # VAAL number of epochs

MOMENTUM = 0.9
WDECAY = 5e-4#2e-3# 5e-4

NUM_CLASSES = 10

#for Noise Stability
NOISE_SCALE = 0.001
