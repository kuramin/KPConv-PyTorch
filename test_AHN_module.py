# Common libs
import os

# Dataset
from datasets.AHN import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPFCNN


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

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = 'last_AHN'

    # Choose to test on validation or test split
    on_val = True  # kuramin changed from True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ###############
    # Previous chkp
    ###############

    chosen_chkp = os.path.join(chosen_log, 'checkpoints', 'current_chkp.tar')

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    #config.validation_size = 200 #200 kuramin changed

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

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    print('test_loader.dataset.set', test_loader.dataset.set)
    tester.cloud_segmentation_test(net, test_loader, config)

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
