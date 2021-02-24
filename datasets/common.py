# Common libs
import time
import os
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from utils.config import Config
from utils.mayavi_visu import *
from kernels.kernel_points import create_3D_rotations

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=1):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    Creates a grid with a specified cellsize, distributes points of cloud to these cells
    and finds barycenter of points (and features) in each cell.
    Returns a cloud where each non-empty cell is represented by one point

    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a batch grid subsampling (method = barycenter for points and features)
    If random_grid_orient=True, then every batch is rotated for a random angle around random axis,
    grid sampling of every batch is done and then every batch is rotated back.
    Returns batches which were subsampled by grids in different rotational positions

    :param points: (N, 3) matrix of input points
    :param batches_len (b,) list of lengths of batches which points were divided to
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)

    # batch_len contains amounts of points
    # for every member of batches_len define random direction of rotation axis and random rotation angle,
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Subsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points (which are centers of neighborhoods)
    :param supports: (N2, 3) the support points (which are initial point cloud)
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B) the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices - a 2D matrix: a list of neighbors for every point of cloud QUERIES found among points of cloud SUPPORT.

    Cpp-function returns just a 1-D list which is a flattened matrix of neighbors for each query point
    indexes are calculated as (i0 * max_count + j),
    where i0 is index of some query point
    and j is index of some neighbor of this query point
    If some query point does not have max_count neighbors, all redundant indexes are set to -1
    Later in the file wrapper.cppp it is specified how does 1D list become a 2D matrix
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/


class PointCloudDataset(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, name):
        """
        Initialize parameters of the dataset here.
        """

        self.name = name
        self.path = ''
        self.label_to_names = {}
        self.num_classes = 0
        self.label_values = np.zeros((0,), dtype=np.int32)
        self.label_names = []
        self.label_to_idx = {}
        self.name_to_label = {}
        self.config = Config()
        self.neighborhood_limits = []

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return 0

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """

        return 0

    def init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise

        return augmented_points, scale, R  # kuramin added instead of following lines which are used only for ModelNet
        # if normals is None:
        #     return augmented_points, scale, R
        # else:
        #     # Anisotropic scale of the normals thanks to cross product formula
        #     normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
        #     augmented_normals = np.dot(normals, R) * normal_scale
        #     # Renormalise
        #     augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)
        #
        #     if verbose:
        #         test_p = [np.vstack([points, augmented_points])]
        #         test_n = [np.vstack([normals, augmented_normals])]
        #         test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
        #         show_ModelNet_examples(test_p, test_n, test_l)
        #
        #     return augmented_points, augmented_normals, scale, R

    def big_neighborhood_filter(self, neigh_indices, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neigh_indices[:, :self.neighborhood_limits[layer]]
        else:
            return neigh_indices

    def segmentation_inputs(self,
                            stacked_points,
                            stacked_features,
                            labels,
                            stack_lengths):
        # print('Begin calculating segmentation input')  # kuramins print

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors_indices = []
        input_indices_of_neighs_of_pooled = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture
        # print(arch)  # kuramins print

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('strided' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # In our case all the following code of for-loop is performed only in case of strided or upsampling blocks
            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            # Convolutions are done in this layer, compute the neighbors with the good radius

            #if layer_blocks: #kuramin commented (indentation begins)
            if np.any(['deformable' in blck for blck in layer_blocks]):
                r = r_normal * self.config.deform_radius / self.config.conv_radius
                deform_layer = True
            else:
                r = r_normal
            # now lets build neighborhoods based on radius r.
            # neigh_indices are indices of neighbors for every point in stacked_points (not only barycenters)
            neigh_indices = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            # Pooling neighbors indices
            # *************************

            # If end of current layer is a block of resnetb_strided or resnetb_deformable_strided
            if 'strided' in block:

                # Set new subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                #print('r before pooled_points', r)  kuramin_print
                #print('dl before pooled_points', dl)  kuramin_print
                # And perform grid subsampling with this new value of dl
                pooled_points, pooled_batches = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                #print('r before indices_of_neighs_of_pooled', r) kuramin_print
                # Subsample indices
                indices_of_neighs_of_pooled = batch_neighbors(pooled_points, stacked_points, pooled_batches, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                upsampled_indices = batch_neighbors(stacked_points, pooled_points, stack_lengths, pooled_batches, 2 * r)

            else:
                # Upsampling layer is met, which means that last layer did not have strided (pooling) block
                # This layer will have input, but no points will be pooled. Thus, no pooling indices required
                indices_of_neighs_of_pooled = np.zeros((0, 1), dtype=np.int32)
                pooled_points = np.zeros((0, 3), dtype=np.float32)
                pooled_batches = np.zeros((0,), dtype=np.int32)
                upsampled_indices = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            # Length of input_points provides number of layer. Based on number of layer and list "self.neighborhood_limits"
            # which we read from a pickle-file, we can leave only specified amount of closest neighbors and cut rest
            # This is done on neigh_indices by function big_neighborhood_filter based on value self.neighborhood_limits
            neigh_indices = self.big_neighborhood_filter(neigh_indices, len(input_points))
            indices_of_neighs_of_pooled = self.big_neighborhood_filter(indices_of_neighs_of_pooled, len(input_points))
            if upsampled_indices.shape[0] > 0:
                upsampled_indices = self.big_neighborhood_filter(upsampled_indices, len(input_points)+1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors_indices += [neigh_indices.astype(np.int64)]
            input_indices_of_neighs_of_pooled += [indices_of_neighs_of_pooled.astype(np.int64)]
            input_upsamples += [upsampled_indices.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pooled_points
            stack_lengths = pooled_batches

            # Update radius and reset blocks
            r_normal *= 2
            # print('block_i', block_i, 'block', block, 'layer_blocks', layer_blocks)  # kuramins print
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                # print('break!')  # kuramins print
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs (concatenated in this way because of different dimensionality of components)
        li = input_points + input_neighbors_indices + input_indices_of_neighs_of_pooled + input_upsamples + input_stack_lengths
        li += [stacked_features, labels]
        # print('End of segmentation_input')  # kuramins print
        return li

    def draw_neighbors(self,
                       stacked_points,
                       stacked_features,
                       labels,
                       stack_lengths):
        # print('Begin calculating segmentation input')  # kuramins print

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors_indices = []
        input_indices_of_neighs_of_pooled = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture
        # print(arch)  # kuramins print

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('strided' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # In our case all the following code of for-loop is performed only in case of strided or upsampling blocks
            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            # Convolutions are done in this layer, compute the neighbors with the good radius

            #if layer_blocks: #kuramin commented (indentation begins)
            if np.any(['deformable' in blck for blck in layer_blocks]):
                r = r_normal * self.config.deform_radius / self.config.conv_radius
                deform_layer = True
            else:
                r = r_normal
            # now lets build neighborhoods based on radius r.
            # neigh_indices are indices of neighbors for every point in stacked_points (not only barycenters)
            #print('r before neigh_indices', r)  kuramin_print
            neigh_indices = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            if 'strided' in block:
                sub_colors = np.zeros_like(stacked_points, dtype=np.uint8)
                sub_labels = np.zeros(stacked_points.shape[0])
                color_code = [block_i * 23, 0, 0]
                sub_colors[1] = color_code
                array_of_edges = np.transpose(np.array([[], []]))
                print('neigh_indices[1]', neigh_indices[1])
                for neigh_ind in neigh_indices[1]:
                    if neigh_ind < neigh_indices.shape[0]:
                        sub_colors[neigh_ind] = color_code
                        print('neigh_ind', neigh_ind)
                        print('array_of_edges', array_of_edges)
                        array_of_edges = np.vstack((array_of_edges, np.array([[1, neigh_ind]])))

                write_ply('../datasets/AHN/input_0.500/sub_ply_file'+ str(block_i) + '.ply',
                          [stacked_points, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'scalar_Classification'], edges=array_of_edges)

        #
        #     # else: #kuramin commented (indentation ends)
        #     #     # This layer only perform pooling, no neighbors required
        #     #     neigh_indices = np.zeros((0, 1), dtype=np.int32)
        #
        #     # Pooling neighbors indices
        #     # *************************
        #
        #     # If end of current layer is a block of resnetb_strided or resnetb_deformable_strided
        #     if 'strided' in block:
        #
        #         # Set new subsampling length
        #         dl = 2 * r_normal / self.config.conv_radius
        #
        #         #print('r before pooled_points', r)  kuramin_print
        #         #print('dl before pooled_points', dl)  kuramin_print
        #         # And perform grid subsampling with this new value of dl
        #         pooled_points, pooled_batches = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)
        #
        #         # Radius of pooled neighbors
        #         if 'deformable' in block:
        #             r = r_normal * self.config.deform_radius / self.config.conv_radius
        #             deform_layer = True
        #         else:
        #             r = r_normal
        #
        #         #print('r before indices_of_neighs_of_pooled', r) kuramin_print
        #         # Subsample indices
        #         indices_of_neighs_of_pooled = batch_neighbors(pooled_points, stacked_points, pooled_batches, stack_lengths, r)
        #
        #         # Upsample indices (with the radius of the next layer to keep wanted density)
        #         upsampled_indices = batch_neighbors(stacked_points, pooled_points, stack_lengths, pooled_batches, 2 * r)
        #
        #     else:
        #         # Upsampling layer is met, which means that last layer did not have strided (pooling) block
        #         # This layer will have input, but no points will be pooled. Thus, no pooling indices required
        #         indices_of_neighs_of_pooled = np.zeros((0, 1), dtype=np.int32)
        #         pooled_points = np.zeros((0, 3), dtype=np.float32)
        #         pooled_batches = np.zeros((0,), dtype=np.int32)
        #         upsampled_indices = np.zeros((0, 1), dtype=np.int32)
        #
        #     # Reduce size of neighbors matrices by eliminating furthest point
        #     # Length of input_points provides number of layer. Based on number of layer and list "self.neighborhood_limits"
        #     # which we read from a pickle-file, we can leave only specified amount of closest neighbors and cut rest
        #     # This is done on neigh_indices by function big_neighborhood_filter based on value self.neighborhood_limits
        #     neigh_indices = self.big_neighborhood_filter(neigh_indices, len(input_points))
        #     indices_of_neighs_of_pooled = self.big_neighborhood_filter(indices_of_neighs_of_pooled, len(input_points))
        #     if upsampled_indices.shape[0] > 0:
        #         upsampled_indices = self.big_neighborhood_filter(upsampled_indices, len(input_points)+1)
        #
        #     # Updating input lists
        #     input_points += [stacked_points]
        #     input_neighbors_indices += [neigh_indices.astype(np.int64)]
        #     input_indices_of_neighs_of_pooled += [indices_of_neighs_of_pooled.astype(np.int64)]
        #     input_upsamples += [upsampled_indices.astype(np.int64)]
        #     input_stack_lengths += [stack_lengths]
        #     deform_layers += [deform_layer]
        #
        #     # New points for next layer
        #     stacked_points = pooled_points
        #     stack_lengths = pooled_batches

            # Update radius and reset blocks
            r_normal *= 2
            # print('block_i', block_i, 'block', block, 'layer_blocks', layer_blocks)  # kuramins print
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'upsample' in block:
                # print('break!')  # kuramins print
                break
        #
        # ###############
        # # Return inputs
        # ###############
        #
        # # list of network inputs (concatenated in this way because of different dimensionality of components)
        # li = input_points + input_neighbors_indices + input_indices_of_neighs_of_pooled + input_upsamples + input_stack_lengths
        # li += [stacked_features, labels]
        # # print('End of segmentation_input')  # kuramins print
        # return li












