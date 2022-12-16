''' Configuration File.
'''

CUDA_VISIBLE_DEVICES = 0
NUM_TRAIN = 73257 # N
NUM_VAL   = 73257 - NUM_TRAIN
BATCH     = 128  #128 # B
SUBSET    = 29303 # M
ADDENDUM  = 3663 # K
START = 2 * ADDENDUM

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda

TRIALS = 3
CYCLES = 7

EPOCH = 1#200
EPOCH_GCN = 200
LR = 1e-1
LR_GCN = 1e-3
MILESTONES = [160, 240]
EPOCHL = 120#20 #120 # After 120 epochs, stop 
EPOCHV = 100 # VAAL number of epochs

MOMENTUM = 0.9
WDECAY = 5e-4#2e-3# 5e-4

NUM_CLASSES = 10

#for Noise Stability
NOISE_SCALE = 0.001