# This is the config file which can be used to set the parameters for the entire paper.
import torch 
import os
BATCH_SIZE =  64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = 0
'''
Enter all the paramters in lowercase only.
'''


#Dataset details
DATASET = 'cifar100'
NUMBER_CLASSES = 10
if DATASET == 'cifar10':
    NUMBER_CLASSES = 10
elif DATASET == 'cifar100':
    NUMBER_CLASSES = 100
elif DATASET == 'imagenet':
    NUMBER_CLASSES = 1000


MODEL = 'resnet'

#Directories
DATAROOT = './data'
ADV_OUTPUT_FOLDER = './adv_output/'
OUTF = ADV_OUTPUT_FOLDER+MODEL+'_'+DATASET+'/'
if os.path.isdir(OUTF) == False:
    os.makedirs(OUTF)

MODEL_PATH = './pre_trained/'+MODEL+'_'+DATASET+'.pth'

#Adversary details
ADVERSARY = 'deepfool' # other options are bim, deepfool, cw

#noise parameters for sample generation

if ADVERSARY == 'fgsm':
    EPSILON = 0.05
elif ADVERSARY == 'bim':
    EPSILON = 0.01
elif ADVERSARY == 'deepfool':
    if MODEL == 'resnet':
        if DATASET == 'cifar10':
            EPSILON = 0.18
        elif DATASET == 'cifar100':
            EPSILON = 0.03
        else:
            EPSILON = 0.1
    else:
        if DATASET == 'cifar10':
            EPSILON = 0.6
        elif DATASET == 'cifar100':
            EPSILON = 0.1
        else:
            EPSILON = 0.5

## Detector param and LR params
EDGE_THRESHOLD = 6
TRAIN_TEST_SPLIT_RATIO = 0.3


## Parameters for VIT pipeline

RECTIFIER = True # If True, then the rectifier is used, else the original emeddings are used.
