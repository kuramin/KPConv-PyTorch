# Common libs
import os

# Dataset
from datasets.S3DIS import *       # kuramin changed
from datasets.AHN import *
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

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model:
    #   '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = 'results_copied/Log_2021-01-25_14-12-25_88perc'  # kuramin changed
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True  # kuramin changed from True

    # Deal with 'last_XXXXXX' choices
    #chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    number_of_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    print('Number of GPUs is', number_of_gpus)

    if number_of_gpus == 1:
        GPU_ID = '0'
    else:
        GPU_ID = '2'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config.validation_size = 200 #200 kuramin changed
    config.input_threads = 10

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

    print('config.first_subsampling_dl', config.first_subsampling_dl)
    
    #test_dataset = AHNDataset(config, set='validation', use_potentials=True)  kuramin changed validation to test
    test_dataset = AHNDataset(config, set=set, use_potentials=True)
    test_sampler = AHNSampler(test_dataset)
    collate_fn = AHNCollate

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))
    print('\nStart test')
    print('**********\n')

    # Training
    tester.cloud_segmentation_test(net, test_loader, config)
