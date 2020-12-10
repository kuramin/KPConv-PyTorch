#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import *
import numpy as np


def p2p_fitting_regularizer(net):

    # Fitting and repulsive loss at some point keep values of distances for every location of kernel.
    # It doesnt mean that it tries to learn it. It just calculates loss based on them.
    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independent from layers
            # m.min_d2 and KP_min_d2 are [n_points, n_kpoints]
            # Every kernel point in every kernel location has one of neighbors as the closest
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Fitting loss will be a sum along net.modules:
            # sum of [n_points, n_kpoints] of squared distances
            # [ every point of kernel in every point location - the closest to it cloud point ]
            # For some reason, its formulation more difficult
            # as a sum of absolute values of differences
            # between [n_points, n_kpoints] of squared distances
            # and zero-tensor of the same shape.
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations [n_points, n_kpoints, dim]
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):  # net.K is config.num_kernel_points

                # other_KP is [n_points, n_kpoints - 1, dim]
                # other_KP is all kernel points of this kernel except point i
                # other_KP is concatenation of all points before i-th will all points after i-th
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()

                # distances is [n_points, n_kpoints - 1]
                # in every kernel location distances from i-th kernel point to each other n_kpoints - 1
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))

                # Function Clamp_max leaves only those values, which are less than argument max, rest turns to max
                # As a result, (distances - repulse_extent) increases rep_loss only when
                # distance is less than repulse_extent (1.2).
                # If kernel points are further from each other, rep_loss is 0
                # Rep_loss for i-th point of kernel has size n_points and represents
                # how smaller than repulse_extent (1.2) are distances from i-th point to other points of this kernel
                # (much smaller -> big loss)
                rep_loss_i = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)

                # And repulsive_loss after for-loop will become vector of n_points.
                # repulsive_loss grows when distances from each to each kernel point
                # in every kernel location go smaller and smaller than repulse_extent (1.2)
                repulsive_loss += net.l1(rep_loss_i, torch.zeros_like(rep_loss_i)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


# class KPCNN(nn.Module):
#     """
#     Class defining KPCNN
#     """
#
#     def __init__(self, config):
#         super(KPCNN, self).__init__()
#
#         #####################
#         # Network opperations
#         #####################
#
#         # Current radius of convolution and feature dimension
#         layer = 0
#         r = config.first_subsampling_dl * config.conv_radius
#         in_dim = config.in_features_dim
#         out_dim = config.first_features_dim
#         self.K = config.num_kernel_points
#
#         # Save all block operations in a list of modules
#         self.block_ops = nn.ModuleList()
#
#         # Loop over consecutive blocks
#         block_in_layer = 0
#         for block_i, block in enumerate(config.architecture):
#
#             # Check equivariance
#             if ('equivariant' in block) and (not out_dim % 3 == 0):
#                 raise ValueError('Equivariant block but features dimension is not a factor of 3')
#
#             # Detect upsampling block to stop
#             if 'upsample' in block:
#                 break
#
#             # Apply the good block function defining tf ops
#             self.block_ops.append(block_decider(block,
#                                                 r,
#                                                 in_dim,
#                                                 out_dim,
#                                                 layer,
#                                                 config))
#
#
#             # Index of block in this layer
#             block_in_layer += 1
#
#             # Update dimension of input from output
#             if 'simple' in block:
#                 in_dim = out_dim // 2
#             else:
#                 in_dim = out_dim
#
#
#             # Detect change to a subsampled layer
#             if 'pool' in block or 'strided' in block:
#                 # Update radius and feature dimension for next layer
#                 layer += 1
#                 r *= 2
#                 out_dim *= 2
#                 block_in_layer = 0
#
#         self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
#         self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)
#
#         ################
#         # Network Losses
#         ################
#
#         self.criterion = torch.nn.CrossEntropyLoss()
#         self.deform_fitting_mode = config.deform_fitting_mode
#         self.deform_fitting_power = config.deform_fitting_power
#         self.deform_lr_factor = config.deform_lr_factor
#         self.repulse_extent = config.repulse_extent
#         self.output_loss = 0
#         self.reg_loss = 0
#         self.l1 = nn.L1Loss()
#
#         return
#
#     def forward(self, batch, config):
#
#         # Save all block operations in a list of modules
#         x = batch.features.clone().detach()
#
#         # Loop over consecutive blocks
#         for block_op in self.block_ops:
#             x = block_op(x, batch)
#
#         # Head of network
#         x = self.head_mlp(x, batch)
#         x = self.head_softmax(x, batch)
#
#         return x
#
#     def loss(self, outputs, labels):
#         """
#         Runs the loss on outputs of the model
#         :param outputs: logits
#         :param labels: labels
#         :return: loss
#         """
#
#         # Cross entropy loss
#         self.output_loss = self.criterion(outputs, labels)
#
#         # Regularization of deformable offsets
#         if self.deform_fitting_mode == 'point2point':
#             self.reg_loss = p2p_fitting_regularizer(self)
#         elif self.deform_fitting_mode == 'point2plane':
#             raise ValueError('point2plane fitting mode not implemented yet.')
#         else:
#             raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)
#
#         # Combined loss
#         return self.output_loss + self.reg_loss
#
#     @staticmethod
#     def accuracy(outputs, labels):
#         """
#         Computes accuracy of the current batch
#         :param outputs: logits predicted by the network
#         :param labels: labels
#         :return: accuracy value
#         """
#
#         predicted = torch.argmax(outputs.data, dim=1)
#         total = labels.size(0)
#         correct = (predicted == labels).sum().item()
#
#         return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Detect change to next layer for skip connection
            # If block is strided or upsample, append it to list of block_numbers and list of in_dims
            if np.any([tmp in block for tmp in ['strided', 'upsample']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Append corresponding encoder blocks to the network
            # Apply the good block function defining tf ops -- ??
            self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            if 'simple' in block:
                # sets in_dim for next block (resnetb block of layer 0) to 64
                in_dim = out_dim // 2
            else:
                # sets in_dim for next block equal to out_dim of current block
                in_dim = out_dim

            # In the end of every encoder layer update number of layer, radius and out_dim for next layer
            if 'strided' in block:
                layer += 1
                r *= 2
                out_dim *= 2

        print('encoder_blocks is calculated as', self.encoder_blocks)
        print('layer after encoder is', layer)
        print('r after encoder is', r)
        print('out_dim after encoder is', out_dim)

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks starting from block start_i
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                # in_dim of this decoder layer is taken from corresponding value from encoder
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)  # keep block numbers of upsample blocks in decoder

            # Append corresponding decoder blocks to the network
            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input for next block from output of current block
            in_dim = out_dim

            # In the beginning of every decoder layer decrease number of layer, radius and out_dim for next layer
            if 'upsample' in block:
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        print('decoder.blocks is', self.decoder_blocks)
        print('layer after decoder is', layer)
        print('r after decoder is', r)
        print('out_dim after decoder is', out_dim)

        # head of network is the end
        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
            print('class_w is', class_w)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        print('Initialized the following KPFCNN architecture', self)

        return

    def forward(self, batch, config):
        """
        Describes how forward pass comes from input features to softmax values.
        As described in S3DIS.S3DISCustomBatch, batch is list of points (with their features, labels etc) 
        which were taken within one of balls, then concatenated and subsampled to 5 different levels
        Batch - consists of batch.points, batch.features, batch.neighbors, batch.labels etc
        Batch.points[0] - consists of points of level 0, each is [X, Y, Z] (~80 000 for S3DIS)
        Batch.points[1] - consists of points of level 1, each is [X, Y, Z] (~22 000 for S3DIS)
        Batch.points[2] - consists of points of level 2, each is [X, Y, Z] (~ 6 000 for S3DIS)
        Batch.points[3] - consists of points of level 3, each is [X, Y, Z] (~ 1 500 for S3DIS)
        Batch.points[4] - consists of points of level 4, each is [X, Y, Z] (~   400 for S3DIS)
        
        Batch.features[0] - consists of feature-lists of those points of level 0 (~80 000 for S3DIS), each list contains values of 5 features from initial cloud 
        Batch.features[1] - consists of feature-lists of those points of level 1 (~22 000 for S3DIS), each list contains values of 5 features from initial cloud
        Batch.features[2] - consists of feature-lists of those points of level 2 (~ 6 000 for S3DIS), each list contains values of 5 features from initial cloud
        Batch.features[3] - consists of feature-lists of those points of level 3 (~ 1 500 for S3DIS), each list contains values of 5 features from initial cloud
        """

        # Get input features
        x = batch.features.clone().detach()

        print('len(batch.points)', len(batch.points))
        print('len(batch.points[0])', len(batch.points[0]))
        print('len(batch.points[1])', len(batch.points[1]))
        print('len(batch.points[2])', len(batch.points[2]))
        print('len(batch.points[3])', len(batch.points[3]))   
        print('len(batch.points[4])', len(batch.points[4]))       

        print('x.size() before for', x.size())
        print('len(batch.features)', len(batch.features))
        print('len(batch.features[0])', len(batch.features[0]))
        print('len(batch.features[1])', len(batch.features[1]))
        print('len(batch.features[2])', len(batch.features[2]))
        print('len(batch.features[3])', len(batch.features[3]))   
        print('len(batch.features[4])', len(batch.features[4]))   
        print('len(batch.features[5])', len(batch.features[5]))   

        # Loop over consecutive blocks
        skip_x = []
        # self.encoder_skips is [2, 5, 8, 11, 14] (all strided blocks and the first upsample block)
        for block_i, block_op in enumerate(self.encoder_blocks):
            print()
            print('In for before if: block_i', block_i, 'will apply block_op', block_op, 'to x, where x.size() is', x.size())
            if block_i in self.encoder_skips:
                print('In for before append: block_i', block_i, 'but we append x to skip_x, x.size() is', x.size())
                skip_x.append(x)
            x = block_op(x, batch)  # apply the block to x
            print('In for after if: now block is applied to x, x.size() is', x.size())
            
        print('self.encoder_skips is', self.encoder_skips)
        print('skip_x is', skip_x)
        print('skip_x size is', len(skip_x))
        print('skip_x[0] len is', len(skip_x[0]))
        print('skip_x[1] len is', len(skip_x[1]))
        print('skip_x[2] len is', len(skip_x[2]))
        print('skip_x[3] len is', len(skip_x[3]))
        
#         print('skip_x[0][0] len is', len(skip_x[0][0]))
#         print('skip_x[0][1] len is', len(skip_x[0][1]))
        
#         print('skip_x[1][0] len is', len(skip_x[1][0]))
#         print('skip_x[1][1] len is', len(skip_x[1][1]))
        
#         print('skip_x[2][0] len is', len(skip_x[2][0]))
#         print('skip_x[2][1] len is', len(skip_x[2][1]))
        
#         print('skip_x[3][0] len is', len(skip_x[3][0]))
#         print('skip_x[3][1] len is', len(skip_x[3][1]))
        
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        #print('self.decoder_concats is', self.decoder_concats)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i
        #print('target before unsqueeze(0) is', target)
        #print('outputs before transpose and unsqueeze(0) is', outputs)

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0).long()

        #print('target after unsqueeze(0) is', target)
        #print('outputs after transpose and unsqueeze(0) is', outputs)

        # Use chosen CrossEntropyLoss which was assigned during init
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total





















