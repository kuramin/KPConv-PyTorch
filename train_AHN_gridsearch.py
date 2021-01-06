#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a grid search training on AHN dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Dmitry Kuramin - 06/01/2021
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from train_AHN_module import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    number_of_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    print('Number of GPUs is', number_of_gpus)

    if number_of_gpus == 1:
        GPU_ID = '0'
    else:
        GPU_ID = '3'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    train_AHN_on_hyperparameters(1.7)
    
    
#     ###############
#     # Previous chkp
#     ###############

#     # Choose here if you want to start training from a previous snapshot (None for new training)
#     # previous_training_path = 'Log_2020-03-19_19-53-27'
#     previous_training_path = ''

#     # Choose index of checkpoint to start from. If None, uses the latest chkp
#     chkp_idx = None
#     if previous_training_path:

#         # Find all snapshot in the chosen training folder
#         chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
#         chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

#         # Find which snapshot to restore
#         if chkp_idx is None:
#             chosen_chkp = 'current_chkp.tar'
#         else:
#             chosen_chkp = np.sort(chkps)[chkp_idx]
#         chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

#     else:
#         chosen_chkp = None

