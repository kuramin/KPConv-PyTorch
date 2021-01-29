# Common libs
import time
import numpy as np
import pickle
import torch
import math
#from multiprocessing import Lock
import multiprocessing

# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class S3DISDataset(PointCloudDataset):
    """Class to handle S3DIS dataset."""

    def __init__(self, config, set='training', use_potentials=True, load_data=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'S3DIS')

        ############
        # Parameters
        ############

        # Dict from labels to names
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'chair',
                               8: 'table',
                               9: 'bookcase',
                               10: 'sofa',
                               11: 'board',
                               12: 'clutter'}

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Dataset folder
        self.path = '../datasets/Stanford3dDataset_v1.2'
        #self.path = '../datasets/Vaihingen'  # kuramin changed

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Path of the training files
        self.train_path = 'original_ply'

        # List of files to process
        ply_path = join(self.path, self.train_path)

        # Proportion of validation scenes
        # self.cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
        # self.all_splits = [0, 1, 2, 3, 4, 5]
        # self.validation_split = 5    # kuramin changed

        #self.cloud_names = ['Vaihingen3D_Training_rgb', 'Vaihingen3D_Evaluation_rgb']
        #self.cloud_names = ['cloud7 - Cloud2', 'cloud7 - Cloud2']
        #self.cloud_names = ['Area_1_fake_rgb_remarked', 'Area_2']
        #self.cloud_names = ['Area_1_fake_rgb_remarked', 'Area_2_remarked']

        self.cloud_names = ['Area_1_fake_rgb_scalar_Classification', 'Area_3_fake_rgb_scalar_Classification']
        self.all_splits = [0, 1]
        self.validation_split = 1

        # Number of models used per epoch (used only during visualization)
        if self.set == 'training':
            self.epoch_n = config.steps_per_epoch * config.batch_num
        elif self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for S3DIS data: ', self.set)

        # Stop, data is not needed
        if not load_data:
            return

        ###################
        # Prepare ply files
        ###################

        #ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            print("No folder", ply_path, "with ply-files was found, let's create ply-files from txt-files")
            makedirs(ply_path)
            self.prepare_S3DIS_ply(ply_path)
        else:
            print("Ply-files are already created based on txt-files")

        ################
        # Load ply files
        ################

        # Fill in the list of training (or validation) files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.set == 'training':
                # if training, self.files collects all full paths based on cloud_names which are for training
                if self.all_splits[i] != self.validation_split:
                    self.files += [join(ply_path, f + '.ply')]
            elif self.set in ['validation', 'test', 'ERF']:
                # if validation or test, self.files collects all full paths based on cloud_names which are for validation (or test)
                if self.all_splits[i] == self.validation_split:
                    self.files += [join(ply_path, f + '.ply')]
            else:
                raise ValueError('Unknown set for S3DIS data: ', self.set)

        # leave only those cloud_names which are relevant for current procedure
        if self.set == 'training':
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] != self.validation_split]
        elif self.set in ['validation', 'test', 'ERF']:
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] == self.validation_split]

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Load data from self.files to self.pot_trees
        self.load_subsampled_clouds()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []

            # for every training/validation point cloud create a list of potentials - list of random values
            # in range (0, 1e-3) of the same size as pot_tree of this cloud
            # When list of potentials for current cloud is created, append it to self.potentials,
            # then find an index of minimal potential for last appended cloud,
            # and append this index to self.argmin_potentials.
            # The value of minimal potential of current cloud is appended to self.min_potentials
            # create lists of minimal values of potentials and their indexes
            # So, self.potentials[i] contains list of potentials for i-th cloud,
            # self.argmin_potentials[i] - index of point with minimal potential within this cloud
            # self.min_potentials[i] - value of minimal potential within this cloud
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            self.epoch_inds = torch.from_numpy(np.zeros((2, config.steps_per_epoch * config.batch_num), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = multiprocessing.Lock()  # Create a lock which will be used later inside

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        if self.use_potentials:
            # kuramins print
            # print('get_item')
            # print('Threading batch_i', batch_i)
            return self.potential_item(batch_i)
        else:
            return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):
        # print('Start potential item')  # kuramins print
        t = [time.time()]

        # Initiate concatеnation lists
        p_list = []
        f_list = []
        l_list = []
        pi_list = []
        i_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        # inp_counter = 0

        # info = get_worker_info()
        # if info is not None:
        #     wid = info.id
        # else:
        #     wid = None

        while True:
            # print('Begin new iteration of while')  # kuramins print
            t += [time.time()]

            # if debug_workers:
            #     message = ''
            #     for wi in range(info.num_workers):
            #         if wi == wid:
            #             message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
            #         elif self.worker_waiting[wi] == 0:
            #             message += '   '
            #         elif self.worker_waiting[wi] == 1:
            #             message += ' | '
            #         elif self.worker_waiting[wi] == 2:
            #             message += ' o '
            #     print(message)
            #     self.worker_waiting[wid] = 0  #kuramin commented

            with self.worker_lock:

                # if debug_workers:
                #     message = ''
                #     for wi in range(info.num_workers):
                #         if wi == wid:
                #             message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                #         elif self.worker_waiting[wi] == 0:
                #             message += '   '
                #         elif self.worker_waiting[wi] == 1:
                #             message += ' | '
                #         elif self.worker_waiting[wi] == 2:
                #             message += ' o '
                #     print(message)
                #     self.worker_waiting[wid] = 1

                # Get potential minimum  #kuramin commented
                cloud_ind = int(torch.argmin(self.min_potentials))  # cloud_ind is number of cloud, comes from whichever clouds minimal potential is randomized to be smaller
                point_ind = int(self.argmin_potentials[cloud_ind])  # index of the smallest potential in this cloud, which are assigned randomly

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Start creation of ball by assigning a Center point (taken as cloud point with minimal random potential)
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

                # Indices of points of potential trees which are inside ball around center point
                # Result allows to increase potentials of these points
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                #pot_inds = torch.tensor(pot_inds[0])  # query radius returns pot_inds wrapped in extra dimension
                pot_inds = pot_inds[0]  # kuramin returned it back to this

                # Update potentials (Tukey weights plot is -|x|+1 inside [-in_radius, in_radius] and 0 everywhere outside)
                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    tukeys = torch.tensor(tukeys)  # kuramin added when it didnt work on Hulk
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points of once-sampled cloud from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Indices of points of once-sampled original cloud which are inside ball around center point
            # Result is the ball of points which is used later
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]

            # Number collected
            number_of_inball_points = input_inds.shape[0]

            # Collect points, labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([float(self.label_to_idx[l]) for l in input_labels])

            t += [time.time()]

            # Data augmentation. Returns result of application of some random transformation and parameters of this transformation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            t += [time.time()]

            # kuramins print
            # print('input_points.shape', input_points.shape)
            # print('input_features.shape', input_features.shape)
            # print('input_labels.shape', input_labels.shape)
            # print('input_inds.shape', input_inds.shape)
            # print('point_ind is', point_ind)
            # print('cloud_ind is ', cloud_ind)
            # print('scale.shape', scale.shape)
            # print('R.shape', R.shape)

            # inp_filename = '/home/kuramin/Downloads/input'
            # inp_counter += 1
            # print(str(inp_filename)+str(inp_counter)+'.ply')
            # write_ply('/home/kuramin/Downloads/input.ply',
            #               [input_points, input_colors, input_labels],
            #               ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            # as a result, input_points is a ball of radius 1.3 m
            # and p_list is a set of such balls. Number of balls is limited by current value of self.batch_limit

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            i_list += [input_inds]
            pi_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # kuramins print
            # print('len(p_list)', len(p_list))
            # for i in range(len(p_list)):
            #     print('p_list[', i, '].shape', p_list[i].shape)

            # Update batch size
            batch_n += number_of_inball_points
            # kuramins print
            # print('batch_n is', batch_n)
            # print('batch_limit is', self.batch_limit)

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                # print('break out from while, cause batch_limit is', self.batch_limit)  # kuramins print
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if number_of_inball_points > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    number_of_inball_points = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        # print("p_list is full, its size is", len(p_list))  # kuramins print
        # inp_filename = '/home/kuramin/Downloads/stacked_points/stacked_points'  # kuramin added
        # for i, inp in enumerate(p_list):
        #     # print('p_list member', i, 'has size', p_list[i].shape[0])  # kuramins print
        #     inp_filename += '_' + str(p_list[i].shape[0])
        #     p_list[i] += pot_points[pi_list[i], :]

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(pi_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(i_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        #write_ply(str(inp_filename)+'.ply', [stacked_points, features, labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'height', 'class'])  # kuramin added

        # inp_filename = '/home/kuramin/Downloads/stacked_points_xyz_'
        # for i, inp in enumerate(p_list):
        #     write_ply(str(inp_filename)+str(i)+'.ply',
        #                   [p_list[i], f_list[i][:,0:3], l_list[i]],
        #                   ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        # Input features (feature with constant value 1 means that this point is real member of original cloud)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:  # without original height of point
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:  # with original height of point
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 5 (without and with XYZ)')  # Kuramin fixed typo: was 7, now 5

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Perform several batch_grid_samplings to get 5 levels of sampling of cloud stacked_points
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)
        
        #print('len(input_list)',len(input_list))
        #print('len(input_list[0])',len(input_list[0]))
        #print('len(input_list[1])',len(input_list[1]))
        #print('len(input_list[2])',len(input_list[2]))
        #print('len(input_list[3])',len(input_list[3]))
        #print('len(input_list[4])',len(input_list[4]))

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        # if debug_workers:
        #     message = ''
        #     for wi in range(info.num_workers):
        #         if wi == wid:
        #             message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
        #         elif self.worker_waiting[wi] == 0:
        #             message += '   '
        #         elif self.worker_waiting[wi] == 1:
        #             message += ' | '
        #         elif self.worker_waiting[wi] == 2:
        #             message += ' o '
        #     print(message)
        #     self.worker_waiting[wid] = 2  #kuramin commented

        t += [time.time()]

        # print('End of potential item')  # kuramins print
        return input_list

#    def random_item(self, batch_i):
#
#         # Initiate concatenation lists
#         p_list = []
#         f_list = []
#         l_list = []
#         pi_list = []
#         i_list = []
#         ci_list = []
#         s_list = []
#         R_list = []
#         batch_n = 0
#
#         while True:
#
#             with self.worker_lock:  # with the lock acquired
#
#                 print('self.epoch_i', self.epoch_i)
#                 print('self.epoch_inds', self.epoch_inds)
#                 # Get potential minimum
#                 cloud_ind = int(self.epoch_inds[0, self.epoch_i])
#                 point_ind = int(self.epoch_inds[1, self.epoch_i])
#
#                 # Update epoch indice
#                 self.epoch_i += 1
#
#             # Get points from tree structure
#             points = np.array(self.input_trees[cloud_ind].data, copy=False)
#
#             # Center point of input region
#             center_point = points[point_ind, :].reshape(1, -1)
#
#             # Add a small noise to center point
#             if self.set != 'ERF':
#                 center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)
#
#             #################################
#             #  Fill in input_points, input_features, input_labels, input_inds,
#             #  point_ind, cloud_ind, scale, R
#             #################################
#
#             # Indices of points in input region
#             input_inds = self.input_trees[cloud_ind].query_radius(center_point,
#                                                                   r=self.config.in_radius)[0]
#
#             # Number collected
#             number_of_inball_points = input_inds.shape[0]
#
#             # Collect labels and colors
#             input_points = (points[input_inds] - center_point).astype(np.float32)
#             input_colors = self.input_colors[cloud_ind][input_inds]
#             if self.set in ['test', 'ERF']:
#                 input_labels = np.zeros(input_points.shape[0])
#             else:
#                 input_labels = self.input_labels[cloud_ind][input_inds]
#                 input_labels = np.array([self.label_to_idx[l] for l in input_labels])
#
#             # Data augmentation
#             input_points, scale, R = self.augmentation_transform(input_points)
#
#             # Color augmentation
#             if np.random.rand() > self.config.augment_color:
#                 input_colors *= 0
#
#             # Get original height as additional feature
#             input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)
#
#             # Put those collected inputs into lists
#             # Stack batch
#             p_list += [input_points]
#             f_list += [input_features]
#             l_list += [input_labels]
#             i_list += [input_inds]
#             pi_list += [point_ind]
#             ci_list += [cloud_ind]
#             s_list += [scale]
#             R_list += [R]
#
#             # Update batch size
#             batch_n += n
#
#             # In case batch is full, stop
#             if batch_n > int(self.batch_limit):
#                 break
#
#             # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
#             # if number_of_inball_points > int(self.batch_limit):
#             #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
#             #    number_of_inball_points = input_inds.shape[0]
#
#         ###################
#         # Concatenate batch
#         ###################
#
#         stacked_points = np.concatenate(p_list, axis=0)
#         features = np.concatenate(f_list, axis=0)
#         labels = np.concatenate(l_list, axis=0)
#         point_inds = np.array(pi_list, dtype=np.int32)
#         cloud_inds = np.array(ci_list, dtype=np.int32)
#         input_inds = np.concatenate(i_list, axis=0)
#         stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
#         scales = np.array(s_list, dtype=np.float32)
#         rots = np.stack(R_list, axis=0)
#
#         # Input features
#         stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
#         if self.config.in_features_dim == 1:
#             pass
#         elif self.config.in_features_dim == 4:
#             stacked_features = np.hstack((stacked_features, features[:, :3]))
#         elif self.config.in_features_dim == 5:
#             stacked_features = np.hstack((stacked_features, features))
#         else:
#             raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')
#
#         #######################
#         # Create network inputs
#         #######################
#         #
#         #   Points, neighbors, pooling indices for each layers
#         #
#
#         # Get the whole input list
#         input_list = self.segmentation_inputs(stacked_points,
#                                               stacked_features,
#                                               labels,
#                                               stack_lengths)
#
#         # Add scale and rotation for testing
#         input_list += [scales, rots, cloud_inds, point_inds, input_inds]
#
#         return input_list

#     def prepare_S3DIS_ply(self, ply_path):

#         print('\nPreparing ply files')
#         t0 = time.time()

#         # Folder for the ply files
#         #ply_path = join(self.path, self.train_path)

#         for cloud_name in self.cloud_names:

#             # Pass if the cloud has already been computed
#             cloud_file = join(ply_path, cloud_name + '.ply')
#             if exists(cloud_file):
#                 print("Cloud_file", cloud_file, "already exists, dont use txt files")
#                 continue

#             # Get rooms of the current cloud
#             cloud_folder = join(self.path, cloud_name)
#             room_folders = [join(cloud_folder, room) for room in listdir(cloud_folder) if isdir(join(cloud_folder, room))]

#             # Initiate containers
#             cloud_points = np.empty((0, 3), dtype=np.float32)
#             cloud_colors = np.empty((0, 3), dtype=np.uint8)
#             cloud_classes = np.empty((0, 1), dtype=np.int32)

#             # Loop over rooms
#             for i, room_folder in enumerate(room_folders):

#                 print('Cloud %s - Room %d/%d : %s' % (cloud_name, i+1, len(room_folders), room_folder.split('/')[-1]))

#                 for object_name in listdir(join(room_folder, 'Annotations')):

#                     if object_name[-4:] == '.txt':

#                         # Text file containing point of the object
#                         object_file = join(room_folder, 'Annotations', object_name)

#                         # Object class and ID
#                         tmp = object_name[:-4].split('_')[0]
#                         if tmp in self.name_to_label:
#                             object_class = self.name_to_label[tmp]
#                         elif tmp in ['stairs']:
#                             object_class = self.name_to_label['clutter']
#                         else:
#                             raise ValueError('Unknown object name: ' + str(tmp))

#                         # Correct bug in S3DIS dataset
#                         if object_name == 'ceiling_1.txt':
#                             with open(object_file, 'r') as f:
#                                 lines = f.readlines()
#                             for l_i, line in enumerate(lines):
#                                 if '103.0\x100000' in line:
#                                     lines[l_i] = line.replace('103.0\x100000', '103.000000')
#                             with open(object_file, 'w') as f:
#                                 f.writelines(lines)

#                         # Read object points and colors
#                         object_data = np.loadtxt(object_file, dtype=np.float32)

#                         # Stack all data
#                         cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
#                         cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
#                         object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
#                         cloud_classes = np.vstack((cloud_classes, object_classes))

#             # Save as ply
#             write_ply(cloud_file,
#                       (cloud_points, cloud_colors, cloud_classes),
#                       ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

#         print('Done in {:.1f}s'.format(time.time() - t0))
#         return

    def load_subsampled_clouds(self):

        name_of_class_property = 'scalar_Classification'
        
        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree {:s} for cloud {:s} with path {:s}, subsampled at {:.3f}'.format(KDTree_file, cloud_name, sub_ply_file, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                #sub_labels = data['scalar_Classification']
                #sub_labels = data['class']
                sub_labels = data[name_of_class_property]

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree {:s} for cloud {:s} with path {:s}, subsampled at {:.3f}'.format(KDTree_file, cloud_name, sub_ply_file, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                #colors = np.vstack((data['scalar_NumberOfReturns'], data['scalar_ReturnNumber'], data['scalar_Intensity'])).T  # kuramin changed
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                #labels_float = data['scalar_Classification']  # kuramin class fake_rgb
                #labels_float = data['class']
                labels_float = data[name_of_class_property]
                labels = []
                for label_float in labels_float:
                    labels.append(int(label_float))

                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=dl)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                #search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                #search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', name_of_class_property])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10  # current value of pot_dl will be 0.15
            cloud_ind = 0

            # in this loop kuramin renamed search_tree to coarse_search_tree and sub_points - to search_tree
            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        coarse_search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    search_tree = np.array(self.input_trees[cloud_ind].data, copy=False)
                    print('Lets find coarse poiints with pot_dl =', pot_dl)
                    coarse_points = grid_subsampling(search_tree.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    coarse_search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(coarse_search_tree, f)

                # Fill data containers
                self.pot_trees += [coarse_search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    #print(data.shape)
                    #print(data['class'].shape)
                    #labels = data['class']

                    #print(data.shape)
                    #print(data['scalar_Classification'].shape)
                    #labels_float = data['scalar_Classification']
                    labels_float = data[name_of_class_property]
                    #print(data['class'].shape)  # kuramin class fake_rgb
                    #labels_float = data['class']  # kuramin class fake_rgb
                    labels = []
                    for label_float in labels_float:
                        label = int(label_float)
                        labels.append(label)
                        if len(labels) % 100000 == 0:
                            print('Transforming labels of whole cloud to int in order to fill self.test_proj. Number of processed members is', len(labels))

                    # Compute projection indices - which members of input_trees[i] is closest to every member of points
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    #dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class S3DISSampler(Sampler):
    """Sampler for S3DIS"""

    def __init__(self, dataset: S3DISDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.steps_per_epoch
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        # not used in our case, go to Generator loop
        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int32)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / (self.dataset.num_clouds * self.dataset.config.num_classes)))

            # Choose random points of each class for each cloud
            for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                epoch_indices = np.empty((0,), dtype=np.int32)
                for label_ind, label in enumerate(self.dataset.label_values):
                    if label not in self.dataset.ignored_labels:
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        if len(label_indices) <= random_pick_n:
                            epoch_indices = np.hstack((epoch_indices, label_indices))
                        elif len(label_indices) < 50 * random_pick_n:
                            new_randoms = np.random.choice(label_indices, size=random_pick_n, replace=False)
                            epoch_indices = np.hstack((epoch_indices, new_randoms.astype(np.int32)))
                        else:
                            rand_inds = []
                            while len(rand_inds) < random_pick_n:
                                rand_inds = np.unique(np.random.choice(label_indices, size=5 * random_pick_n, replace=True))
                            epoch_indices = np.hstack((epoch_indices, rand_inds[:random_pick_n].astype(np.int32)))

                # Stack those indices with the cloud index
                epoch_indices = np.vstack((np.full(epoch_indices.shape, cloud_ind, dtype=np.int32), epoch_indices))

                # Update the global indice container
                all_epoch_inds = np.hstack((all_epoch_inds, epoch_indices))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds[:, :num_centers])

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N


    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
            The neighborhood calibration function is here to control the number of neighbors.
            As we preferred radius neighborhood (for geometrical consistency),
            some of the neighborhoods (in the dense areas like vegetation) can contain a lot of points.
            For an input batch, the neighborhood matrix size is [N, n_max]
            where n_max is the maximum number of neighbors.
            As we have big batches, there is often a dense area among the points, forcing n_max to be large.
            Furthermore, in the case the density is not controlled with a grid, we could end up with very large n_max,
            causing an OOM crash. The neighborhood calibration function sets a limit for n_max,
            by checking some input batches.
            The limit (set for each layer of the network in the self.neighborhood_limits variable of the dataset class)
            is set as the 90th percentile of the distribution of neighbor numbers.
            It means that 90% of the neighborhoods won't be affected by this limit,
            and that the 10% most dense neighborhoods will lose some of their points.
            Because of the way we compute neighborhoods, they lose the furthest points,
            which means they become KNN neighborhoods.
            Thanks to this trick, our network is lighter, faster, and it does not affect the performances.

            Batch calibration: done to set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: done to set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # all the contents of this IF are just calculation histogram and
        # values of self.batch_limit and self.neighborhood_limits from it.
        # These values will define break when enough number of balls are collected during method "potential item"
        # which will be used inside Calibration (somehow it is used inside it too)
        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            upper_bound_of_neigh_number = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, upper_bound_of_neigh_number), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value (0 and 6)
            estim_aver_bat_size = 0
            target_aver_bat_size = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0  # defines which number of points will be added to current value of batch_limit (is multiplied by ERROR)
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################
            print('Before range10')
            for epoch in range(10):  # kuramin changed from 10 to 100 instead of increasing steps_per_epoch
                print('Begin iter o range10. Before enumerate(dataloader)')
                for batch_i, batch in enumerate(dataloader):

                    # kuramins print
                    # print('epoch', epoch, 'batch_i', batch_i, 'batch.neighbors len', len(batch.neighbors))
                    # print('epoch', epoch, 'batch_i', batch_i, 'batch.neighbors[0].shape', batch.neighbors[0].shape)
                    # print('epoch', epoch, 'batch_i', batch_i, 'batch.neighbors[1].shape', batch.neighbors[1].shape)
                    # print('epoch', epoch, 'batch_i', batch_i, 'batch.neighbors[2].shape', batch.neighbors[2].shape)
                    # print('epoch', epoch, 'batch_i', batch_i, 'batch.neighbors[3].shape', batch.neighbors[3].shape)
                    # print('epoch', epoch, 'batch_i', batch_i, 'batch.neighbors[4].shape', batch.neighbors[4].shape)
                    # print('epoch', epoch, 'batch_i', batch_i, 'batch.neighbors\n', batch.neighbors)
                    # print('epoch', epoch, 'batch_i', batch_i, 'end')

                    # Update neighborhood histogram
                    #for neighb_mat in batch.neighbors:
                    #    print('neighb_mat.numpy())', neighb_mat.numpy())
                    #    print('neighb_mat.shape[0])', neighb_mat.shape[0])

                    # number of neighbors of each point on every layer (5 layers)
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]

                    # on every layer we calculate how many points have 0 neighs, 1 neigh, 2 .... upper_bound_of_neigh_number neighs (which is much more than we need)
                    hists = [np.bincount(c, minlength=upper_bound_of_neigh_number)[:upper_bound_of_neigh_number] for c in counts]

                    # transform list hists to ndarray neighb_hists
                    neighb_hists += np.vstack(hists)
                    # kuramins print
                    # print('counts', counts)
                    # print('neighb_hists', neighb_hists)

                    # batch length is number of balls collected now within current value of batch_limit
                    b = len(batch.cloud_inds)

                    # Update estim_aver_bat_size (low pass filter)
                    estim_aver_bat_size += (b - estim_aver_bat_size) / low_pass_T

                    # Estimate error (noisy)
                    error = target_aver_bat_size - b

                    # Save smooth errors for convergence check
                    smooth_errors.append(target_aver_bat_size - estim_aver_bat_size)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error

                    # turn on finer low pass filter when estim_aver_bat_size is close to target_aver_bat_size
                    if not finer and np.abs(estim_aver_bat_size - target_aver_bat_size) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose:
                    #if verbose and (t - last_display) > 1.0: # kuramin commented
                        last_display = t
                        #message = 'For-loop through batches during calibration: Step {:5d} - estim_aver_bat_size ={:5.2f}, b ={:3d}, batch_limit ={:7d}, max_a_sm_err = {:3.7f}'
                        message = 'Step {:5d} - estim_aver_bat_size ={:5.2f}, b ={:3d}, bat_lim ={:7d}, error = {:2d}, sm_append = {:5.5f}, lets_finer ={:5.5f}, max_a_sm_err = {:3.7f}, low_pass = {:3d}, finer = {:1d}'
                        print(message.format(i,
                                             estim_aver_bat_size,
                                             b,
                                             int(self.dataset.batch_limit),
                                             error,
                                             target_aver_bat_size - estim_aver_bat_size,
                                             np.abs(estim_aver_bat_size - target_aver_bat_size),
                                             np.max(np.abs(smooth_errors)),
                                             low_pass_T,
                                             finer))
                        #print('smooth errors is', smooth_errors)
                    #print('Last step of enumerate(dataloader), breaking is ', breaking)
                print('After enumerate(dataloader), breaking is ', breaking)
                if breaking:
                    break
                print('End of iter of range10')
            # Use collected neighbor histogram to get neighbors limit
            print('After range10')
            # cumsum[i] contains a list: number of points which have i or less neighbors on sampling level 0, i or less neighbors on level 1, ...., level 4
            cumsum = np.cumsum(neighb_hists.T, axis=0)

            # number of neighbors which is exceeded by only 10% of points of sampling level 0, same for level 1, ...., level 4
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[upper_bound_of_neigh_number - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            # Print histogram
            if verbose:
                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                upper_bound_of_neigh_number = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(upper_bound_of_neigh_number):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary to batch_limits.pkl
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighborhood_limits to neighbors_limits.pkl
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)

        print('self.dataset.batch_limit', self.dataset.batch_limit, 'self.dataset.neighborhood_limits', self.dataset.neighborhood_limits)
        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class S3DISCustomBatch:
    """Custom batch definition with memory pinning for S3DIS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 7) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def S3DISCollate(batch_data):
    return S3DISCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/
#
#
# def debug_upsampling(dataset, loader):
#     """Shows which labels are sampled according to strategy chosen"""
#
#
#     for epoch in range(10):
#
#         for batch_i, batch in enumerate(loader):
#
#             pc1 = batch.points[1].numpy()
#             pc2 = batch.points[2].numpy()
#             up1 = batch.upsamples[1].numpy()
#
#             print(pc1.shape, '=>', pc2.shape)
#             print(up1.shape, np.max(up1))
#
#             pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))
#
#             # Get neighbors distance
#             p0 = pc1[10, :]
#             neighbs0 = up1[10, :]
#             neighbs0 = pc2[neighbs0, :] - p0
#             d2 = np.sum(neighbs0 ** 2, axis=1)
#
#             print(neighbs0.shape)
#             print(neighbs0[:5])
#             print(d2[:5])
#
#             print('******************')
#         print('*******************************************')
#
#     _, counts = np.unique(dataset.input_labels, return_counts=True)
#     print(counts)
#
#
# def debug_timing(dataset, loader):
#     """Timing of generator function"""
#
#     t = [time.time()]
#     last_display = time.time()
#     mean_dt = np.zeros(2)
#     estim_aver_bat_size = dataset.config.batch_num
#     estim_N = 0
#
#     for epoch in range(10):
#
#         for batch_i, batch in enumerate(loader):
#             # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)
#
#             # New time
#             t = t[-1:]
#             t += [time.time()]
#
#             # Update estim_aver_bat_size (low pass filter)
#             estim_aver_bat_size += (len(batch.cloud_inds) - estim_aver_bat_size) / 100
#             estim_N += (batch.features.shape[0] - estim_N) / 10
#
#             # Pause simulating computations
#             time.sleep(0.05)
#             t += [time.time()]
#
#             # Average timing
#             mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))
#
#             # Console display (only one per second)
#             if (t[-1] - last_display) > -1.0:
#                 last_display = t[-1]
#                 message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
#                 print(message.format(batch_i,
#                                      1000 * mean_dt[0],
#                                      1000 * mean_dt[1],
#                                      estim_aver_bat_size,
#                                      estim_N))
#
#         print('************* Epoch ended *************')
#
#     _, counts = np.unique(dataset.input_labels, return_counts=True)
#     print(counts)
#
#
# def debug_show_clouds(dataset, loader):
#
#
#     for epoch in range(10):
#
#         clouds = []
#         cloud_normals = []
#         cloud_labels = []
#
#         L = dataset.config.num_layers
#
#         for batch_i, batch in enumerate(loader):
#
#             # Print characteristics of input tensors
#             print('\nPoints tensors')
#             for i in range(L):
#                 print(batch.points[i].dtype, batch.points[i].shape)
#             print('\nNeigbors tensors')
#             for i in range(L):
#                 print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
#             print('\nPools tensors')
#             for i in range(L):
#                 print(batch.pools[i].dtype, batch.pools[i].shape)
#             print('\nStack lengths')
#             for i in range(L):
#                 print(batch.lengths[i].dtype, batch.lengths[i].shape)
#             print('\nFeatures')
#             print(batch.features.dtype, batch.features.shape)
#             print('\nLabels')
#             print(batch.labels.dtype, batch.labels.shape)
#             print('\nAugment Scales')
#             print(batch.scales.dtype, batch.scales.shape)
#             print('\nAugment Rotations')
#             print(batch.rots.dtype, batch.rots.shape)
#             print('\nModel indices')
#             print(batch.model_inds.dtype, batch.model_inds.shape)
#
#             print('\nAre input tensors pinned')
#             print(batch.neighbors[0].is_pinned())
#             print(batch.neighbors[-1].is_pinned())
#             print(batch.points[0].is_pinned())
#             print(batch.points[-1].is_pinned())
#             print(batch.labels.is_pinned())
#             print(batch.scales.is_pinned())
#             print(batch.rots.is_pinned())
#             print(batch.model_inds.is_pinned())
#
#             show_input_batch(batch)
#
#         print('*******************************************')
#
#     _, counts = np.unique(dataset.input_labels, return_counts=True)
#     print(counts)
#
#
# def debug_batch_and_neighbors_calib(dataset, loader):
#     """Timing of generator function"""
#
#     t = [time.time()]
#     last_display = time.time()
#     mean_dt = np.zeros(2)
#
#     for epoch in range(10):
#
#         for batch_i, input_list in enumerate(loader):
#             # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)
#
#             # New time
#             t = t[-1:]
#             t += [time.time()]
#
#             # Pause simulating computations
#             time.sleep(0.01)
#             t += [time.time()]
#
#             # Average timing
#             mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))
#
#             # Console display (only one per second)
#             if (t[-1] - last_display) > 1.0:
#                 last_display = t[-1]
#                 message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
#                 print(message.format(batch_i,
#                                      1000 * mean_dt[0],
#                                      1000 * mean_dt[1]))
#
#         print('************* Epoch ended *************')
#
#     _, counts = np.unique(dataset.input_labels, return_counts=True)
#     print(counts)
