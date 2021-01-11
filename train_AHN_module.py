#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on AHN dataset
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
import time

# Dataset
from datasets.AHN import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN
import subprocess


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class AHNConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    Inherit methods __init__, load and save from class Config
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'AHN'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 0  # 10 kuramin changed

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 15  # was 1.5 for s3dis

    # Number of kernel points
    num_kernel_points = 15  # kuramin changed back from 9

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.5  # was 2.0 before   # was 0.03 for s3dis

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128 # kuramin changed back from 8
    in_features_dim = 5 # kuramin changed back from 4

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points (repulsive regularisation)

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 10  # 500  kuramin changed

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    # Dictionary of all decay values with their epoch {epoch: decay}.
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    # Gradient clipping value (negative means no clipping)
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 6  # target_aver_batch_size will be set equal to it

    # Number of steps per epoch (how many batches will be created from dataloader by enumerate(dataloader))
    steps_per_epoch = 500  # kuramin changed back from 100

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 7  # 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we need to save convergence
    saving = True
    saving_path = None

    # kuramin copied from config.py
    # Regularization loss importance
    weight_decay = 1e-3

    # Fixed points in the kernel : 'none', 'center' or 'verticals'
    fixed_kernel_points = 'center'

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    class_w = []
    
    acc_aver = None
    acc_var = None
    save_potentials = False


def train_AHN_on_hyperparameters(fsd, 
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
                                 gridsearch_filename):
    
    print()
    print('Start new train_AHN_on_hyperparameters')
    print('****************')
    time.sleep(5)

    # Initialize configuration class
    config = AHNConfig()

    config.first_subsampling_dl = fsd
    config.in_radius = in_radius
    config.conv_radius = conv_radius
    config.deform_radius = deform_radius
    config.repulse_extent = repulse_extent
    config.KP_extent = KP_extent
    config.num_kernel_points = num_kernel_points
    config.deform_fitting_power = deform_fitting_power
    config.max_epoch = max_epoch
    config.steps_per_epoch = steps_per_epoch
    config.input_threads = input_threads

    message_param_string = 'Config set to {:2.3f} {:2.3f} {:2.3f} {:2.3f} {:2.3f} {:2.3f} {:2d} {:2.3f} {:3d} {:3d} {:2d} '
    message_param_string = message_param_string.format(config.first_subsampling_dl,
                                                       config.in_radius,
                                                       config.conv_radius,
                                                       config.deform_radius,
                                                       config.repulse_extent,
                                                       config.KP_extent,
                                                       config.num_kernel_points,
                                                       config.deform_fitting_power,
                                                       config.max_epoch,
                                                       config.steps_per_epoch,
                                                       config.input_threads)
    print(message_param_string)
    message_path_string = ''
    message = ''

    try:
        chosen_chkp = None
        
        # Initialize datasets
        training_dataset = AHNDataset(config, set='training', use_potentials=True)  # kuramin commented
        test_dataset = AHNDataset(config, set='validation', use_potentials=True)

        # Initialize samplers
        training_sampler = AHNSampler(training_dataset)  # defines the strategy to draw samples from the dataset
        test_sampler = AHNSampler(test_dataset)

        # Initialize the dataloader
        r"""
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
            sampler (Sampler, optional): defines the strategy to draw samples from
                the dataset. If specified, :attr:`shuffle` must be ``False``.
            batch_sampler (Sampler, optional): like :attr:`sampler`, but returns a batch of
                indices at a time. Mutually exclusive with :attr:`batch_size`,
                :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
                into CUDA pinned memory before returning them.  If your data elements
                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
                see the example below.
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            timeout (numeric, optional): if positive, the timeout value for collecting a batch
                from workers. Should always be non-negative. (default: ``0``)
            worker_init_fn (callable, optional): If not ``None``, this will be called on each
                worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
                input, after seeding and before data loading. (default: ``None``)
        """
        training_loader = DataLoader(training_dataset,
                                     batch_size=1,
                                     sampler=training_sampler,
                                     collate_fn=AHNCollate,
                                     num_workers=config.input_threads,
                                     pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 sampler=test_sampler,
                                 collate_fn=AHNCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)

        # Calibrate samplers
        training_sampler.calibration(training_loader, verbose=True)
        test_sampler.calibration(test_loader, verbose=True)

        # Optional debug functions
        # debug_timing(training_dataset, training_loader)
        # debug_timing(test_dataset, test_loader)
        # debug_upsampling(training_dataset, training_loader)

        print('\nModel Preparation')
        print('*****************')

        # Define network model
        t1 = time.time()
        net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

        # debug = False
        # if debug:
        #     print('\n*************************************\n')
        #     print(net)
        #     print('\n*************************************\n')
        #     for param in net.parameters():
        #         if param.requires_grad:
        #             print(param.shape)
        #     print('\n*************************************\n')
        #     print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        #     print('\n*************************************\n')

        # Define a trainer class
        trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
        print('Done in {:.1f}s\n'.format(time.time() - t1))
        message_path_string = str(config.saving_path)

        print('\nStart training')
        print('**************')

        # Training
        trainer.train(net, training_loader, test_loader, config)

        print('End attempt without forcing')
        #print('Forcing exit now')
        #os.kill(os.getpid(), signal.SIGINT)

    except Exception as e:
        message = message_param_string + message_path_string + ' Got exception ' + str(e) + '\n'
    else:
        acc_string = 'acc_aver and acc_var are {:1.4f} {:1.4f}\n'
        acc_string = acc_string.format(config.acc_aver, config.acc_var)
        message = message_param_string + message_path_string + acc_string
    finally:
        print(message)
    
        with open(gridsearch_filename, "a") as file:
            file.write(message)

        print('End of finally part of exception')
    print('End of program')
