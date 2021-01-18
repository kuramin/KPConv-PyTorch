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
    
    input_threads = 0 # 10
    max_epoch = 7
    steps_per_epoch = 50
    
    gridsearch_filename = time.strftime('results/train_AHN_gridsearch_%Y-%m-%d_%H-%M-%S.txt', time.gmtime())
    
    range_fsd = [0.5] # [0.5, 1.0]  # [0.2, 0.5, 1.0, 1.5]
    range_in_radius = [15] # [15, 25]  # [15, 25, 35]
    range_conv_radius = [1.5] # [1.5, 2.5]  # [1.5, 2.5, 3.5]
    range_deform_radius = [5.0] # [5.0, 7.0]  # [5.0, 6.0, 7.0]
    range_repulse_extent = [1.2]
    range_KP_extent = [1.2]
    range_num_kernel_points = [15] # [15, 25]
    range_deform_fitting_power = [0.1] # [0.1, 0.5]  # [0.1, 0.5, 1.0]
    
    # Lets loop
    for fsd in range_fsd:
        for in_radius in range_in_radius:
            for conv_radius in range_conv_radius:
                for deform_radius in range_deform_radius:
                    for repulse_extent in range_repulse_extent:
                        for KP_extent in range_KP_extent:
                            for num_kernel_points in range_num_kernel_points:
                                for deform_fitting_power in range_deform_fitting_power:
                                    print('Start new training')
                                    train_AHN_on_hyperparameters(fsd, 
                                                                 in_radius, 
                                                                 conv_radius, 
                                                                 deform_radius, 
                                                                 repulse_extent, 
                                                                 KP_extent, 
                                                                 num_kernel_points, 
                                                                 deform_fitting_power, 
                                                                 max_epoch, 
                                                                 steps_per_epoch, 
                                                                 input_threads, 
                                                                 gridsearch_filename)
                                    print('End this training')
    print('End of all ranges')