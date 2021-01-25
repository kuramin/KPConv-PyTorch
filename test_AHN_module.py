#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
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
#from datasets.ModelNet40 import *
#from datasets.S3DIS import *       # kuramin changed
from datasets.AHN import *
from datasets.SemanticKitti import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPFCNN
import subprocess


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    #if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
    if chosen_log in ['last_AHN', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_AHN', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def test_AHN(gridsearch_filename):
#if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

#     chosen_log = 'results/Log_2021-01-19_17-06-41'  # kuramin changed
    chosen_log = 'last_AHN'

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True # kuramin changed from True 

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
#     number_of_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
#     print('Number of GPUs is', number_of_gpus)

#     if number_of_gpus == 1:
#         GPU_ID = '0'
#     else:
#         GPU_ID = '3'

#     # Set GPU visible device
#     os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
#     chkp_path = os.path.join(chosen_log, 'checkpoints')
#     chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
#     if chkp_idx is None:
#         chosen_chkp = 'current_chkp.tar'
#     else:
#         chosen_chkp = np.sort(chkps)[chkp_idx]
#     chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', 'current_chkp.tar')

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200 #200 kuramin changed
    #config.input_threads = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    #test_dataset = AHNDataset(config, set='validation', use_potentials=True)  kuramin changed validation to test
    test_dataset = AHNDataset(config, set=set, use_potentials=True)
    test_sampler = AHNSampler(test_dataset)
    #collate_fn = AHNCollate

    # Initiate dataset
    # if config.dataset == 'ModelNet40':
    #     test_dataset = ModelNet40Dataset(config, train=False)
    #     test_sampler = ModelNet40Sampler(test_dataset)
    #     collate_fn = ModelNet40Collate
    # elif config.dataset == 'S3DIS':
    #     test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
    #     test_sampler = S3DISSampler(test_dataset)
    #     collate_fn = S3DISCollate
    # elif config.dataset == 'SemanticKitti':
    #     test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False)
    #     test_sampler = SemanticKittiSampler(test_dataset)
    #     collate_fn = SemanticKittiCollate
    # else:
    #     raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=AHNCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

    # if config.dataset_task == 'classification':
    #     net = KPCNN(config)
    # elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
    #     net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    # else:
    #     raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)
    # elif config.dataset_task == 'classification':
    #     tester.classification_test(net, test_loader, config)
    # elif config.dataset_task == 'slam_segmentation':
    #     tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    print('mious', config.mIoU_aver, config.IoUs_aver[0], config.IoUs_aver[1], config.IoUs_aver[2], config.mIoU_var, config.IoUs_var[0], config.IoUs_var[1], config.IoUs_var[2])
    
    message = ' {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(config.mIoU_aver * 100,
                                                                                  config.IoUs_aver[0] * 100,
                                                                                  config.IoUs_aver[1] * 100,
                                                                                  config.IoUs_aver[2] * 100,
                                                                                  config.mIoU_var * 100,
                                                                                  config.IoUs_var[0] * 100,
                                                                                  config.IoUs_var[1] * 100,
                                                                                  config.IoUs_var[2] * 100)
    print(message)
    with open(gridsearch_filename, "a") as file:
        file.write(message)