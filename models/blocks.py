import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from kernels.kernel_points import load_kernels

from utils.ply import write_ply

# ----------------------------------------------------------------------------------------------------------------------
#
#           Simple functions
#       \**********************/
#


def gather(x, idx, method=0): # 2 kuramin changed
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    
    Performs gathering of elements from an array with respect to the indexes. 
    For example, to extract [7009, 62] from [7009, 259]: neighb_limit changes to new_max_neighb.
    Every of 7009 points has neighbors: some have full 259, some have shadow neighbors.
    We want to keep only those, which can influence the kernel in current position 
    (which are within KP_extent to one of kernel points).
    Some of 7009 have 62 which are in game. These 62 are extracted by GATHER.
    Neighb_row_inds contains row indices of members of neighb_inds which are in game.
    Thus, new_neighb_inds are total indices of points in game.
    """

    if method == 0:
        #print(x.shape, 'gather x', x)
        #print(idx.shape, 'gather idx', idx)
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unknown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0] - each radius defines how far a point is from center
    :param sig: extents of gaussians [d1, d0] or [d0] or float - defines shape of the bell
    :return: gaussian of sq_r [dn, ..., d1, d0] - each member is a
    value of gaussian function at corresponding point (provided without coefficient before exp)
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[:, 0])


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConv class
#       \******************/
#


class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        # initialize an object of class which is parent for KPConv (class nn.Module)
        super(KPConv, self).__init__()
        # What about the second parameter "self"?
        # Remember, this is an object that is an instance of the class used as the first parameter.
        # For an example, isinstance(Cube, Square) must return True.
        # By including an instantiated object, super() returns a bound method:
        # a method that is bound to the object, which gives the method the object’s context
        # such as any instance attributes. If this parameter is not included,
        # the method returned is just a function, unassociated with an object’s context.

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights, which will be trained
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)
        #print(self.weights.shape, 'self.weights in init before reset is', self.weights)

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:  # S3DIS or AHN is not modulated
                self.offset_dim = self.p_dim * self.K
            # all parameters already have some defined value
            # KPConv can call itself because deformable is False by default,
            # so it will not be an endless recursive process
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None


        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)  # Fills the Tensor self.offset_bias with the scalar value `0`
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition inside a sphere of self.radius (as numpy array)
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        """Passes values forward through KPConv module.
        :param q_pts: points which will get feature vectors (n_points[lev] for non-strided and n_points[lev+1] for strided blocks)
        :param s_pts: points which will provide their features
        :param neighb_inds: indices of neighbors for every point of s_pts
        :param x: signal which will be passed through (features)
        """

#         print('len(q_pts) is', len(q_pts))
#         print('len(s_pts) is', len(s_pts))
#         print('len(q_pts[0]) is', len(q_pts[0]))
#         print('len(q_pts[1]) is', len(q_pts[1]))
#         print('len(s_pts[0]) is', len(s_pts[0]))
#         print('len(s_pts[1]) is', len(s_pts[1]))
        ###################
        # Offset generation
        ###################

        if self.deformable:

            # Get offsets with a KPConv that only takes part of the features
#             print("Dive into first offset_conv")
            ofconv = self.offset_conv(q_pts, s_pts, neighb_inds, x)
            self.offset_features = ofconv + self.offset_bias
#             print("Came out from first offset_conv")
#             print('len(ofconv) calculated inside deformable is', len(ofconv), len(ofconv[0]))
#             print('ofconv[0]', ofconv[0])
#             print('ofconv[10]', ofconv[10])
#             print('len(self.offset_bias) calculated inside deformable is', len(self.offset_bias), self.offset_bias)            
            
#             #print('len(self.offset_bias) calculated inside deformable is', len(self.offset_bias), len(self.offset_bias[0]))
#             print('len(self.offset_features) (sum of ofconv and bias) calculated inside deformable is', len(self.offset_features), len(self.offset_features[0]))
            
            # Get offset (in normalized scale) from features
            unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)

            # Rescale offset for this layer: now its a scaled matrix [self.K, self.p_dim]
            offsets = unscaled_offsets * self.KP_extent
            #print(offsets.shape, 'offsets is', offsets)
            #print('len(offsets) calculated inside deformable is', len(offsets), len(offsets[0]), offsets[0])

        else:
            offsets = None

        ######################
        # Deformed convolution
        ######################
        
        # Add a fake point with 1e6 in X,Y,Z into the last row for shadow neighbors
        # Now s_pts is [n_s_points + 1, p_dim] = 
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        #print(neighb_inds.shape, 'neighb_inds is', neighb_inds)
        #print(s_pts.shape, 's_pts')
        #print(q_pts.shape, 'q_pts')
        # Get neighbor points [n_q_points, n_neighbors, dim] - indices of s-neighbors for every q-point
        neighbors = s_pts[neighb_inds, :]
        #print(neighbors.shape, 'neighbors')

        # s_pts has size [n_points[lev], p_dim]
        # q_pts has size [n_points[lev], p_dim] for non-strided blocks
        # and size [n_points[lev+1], p_dim] for strided blocks
        
        # Center every neighborhood around its q-pts point
        # (unsqueeze for subtraction: q_pts from [n_q_points, p_dim] to [n_q_points, 1, p_dim])
        neighbors = neighbors - q_pts.unsqueeze(1)
        #print('len(neighbors) is', len(neighbors), len(neighbors[0]))
        #print('neighbors[0] after is', neighbors[0])
        
        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            #print('len(offsets) remembered inside deformable to calculate deformed kpoints locations is', len(offsets))
            #print('self.kernel_points is', self.kernel_points)
            deformed_K_points = self.deformed_KP.unsqueeze(1)
            #print('len(deformed_K_points) calculated inside deformable is', len(deformed_K_points))
        else:
            deformed_K_points = self.kernel_points
            #print('deformed_K_points calculated inside rigid is just self.kernel_points, its len is', len(deformed_K_points), deformed_K_points[0])

        #print('deformed_K_points[0] is', deformed_K_points[0])
        # Turn neighbors from [n_q_points, n_neighbors, dim] to [n_q_points, n_neighbors, 1, dim]
        neighbors.unsqueeze_(2)

        # Get all difference matrices [n_q_points, n_neighbors, n_kpoints, dim]
        # Each of 95000 points has 28 neighbors,
        # each neighbor has coordinate difference to each of 15 kernel points
        # each difference has X, Y, Z coordinates
        differences = neighbors - deformed_K_points

        # Get the squared distances between each neighbor and each kernel point
        # for each location of kernel (95000 locations)
        # by summing squared differences along X, Y, Z
        # sq_distances is [n_q_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=3)
        #print('len(sq_distances) is', len(sq_distances))
        #print(sq_distances.shape, 'sq_distances after first calculation on kpoints and neighbors is') #, sq_distances)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss [n_q_points, n_kpoints]
            # Every kernel point in every kernel location has one of neighbors as the closest
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Boolean of the neighbors within distance KP_extent from any kernel point with kernel located in every point  
            # [n_q_points, n_neighbors] of boolean if this neighbor is within KP_extent from this point
            in_range = torch.any(sq_distances < self.KP_extent ** 2, dim=2).type(torch.int32)
            #print('inrange0 is', in_range[0])

            # New int value of max neighbors (maximal among all layer-points number of neighs who are within KP_extent from some kernel point)
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))
            #print('new_max_neighb is', new_max_neighb)
            #print(in_range.shape, 'in_range is', in_range)

            # Top new_max_neighb values from each row of in_range
            # which are ones (points within KP_extent) and some zeros (shadow neighbors)
            # indices of those points are returned too
            # [n_q_points, new_max_neighb] - for every point: indices of neighbors who are reachable by kernel in this location
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)
            #print(neighb_row_bool.shape, 'neighb_row_bool is', neighb_row_bool)
            #print(neighb_row_inds.shape, 'neighb_row_inds is', neighb_row_inds)
            #print('neighb_row_inds[0]', neighb_row_inds[0])
            #print('neighb_row_inds > new_max_neighb', torch.any(neighb_row_inds > new_max_neighb))

            # Gather from general matrix "neighb_inds" those members who got 1 in in_range  [n_q_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)
            #print('global neighb_inds[0] is', neighb_inds[0])
            #print('new_neighb_inds[0] is', new_neighb_inds[0])
            #print(new_neighb_inds.shape, 'new_neighb_inds is', new_neighb_inds)

            # Prepare neighb_row_inds to gather new distances to KP
            neighb_row_inds.unsqueeze_(2) # makes it [n_q_points, new_max_neighb, 1]
            #print(neighb_row_inds.shape, 'neighb_row_inds after unsqueeze is', neighb_row_inds) # copies it 14 more times so that its [n_q_points, new_max_neighb, 15]
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K) # copies it 14 more times so that its [n_q_points, new_max_neighb, n_kpoints]
            #print(neighb_row_inds.shape, 'neighb_row_inds after expand is', neighb_row_inds)
            #print(sq_distances.shape, 'sq_distances before gather is', sq_distances)
            # for every point A of n_point: get sq_distances only to those neighbors 
            # which are within KP_extent from some kernel-point of kernel located in A
            # sq_distances turns from [n_q_points, n_neighbors, n_kpoints] to [n_q_points, new_max_neighb, n_kpoints]
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)
            #print(sq_distances.shape, 'sq_distances after gather inside defcalc is') #, sq_distances)

            # New shadow neighbors have to point to the last shadow point
            #print(new_neighb_inds.shape, 'new_neighb_inds before * is', new_neighb_inds)
            # turn indices of neighbors which are not within KP_extent (represented with boolean 0 in in_range) to integer 0 
            new_neighb_inds *= neighb_row_bool

            # turn indices of neighbors which are not within KP_extent from integer 0 (result of previous line) to integer -1 and then to integer n_s_points (make them shadow neighbors)
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
            #print(new_neighb_inds.shape, 'new_neighb_inds after - is') #, new_neighb_inds)
        else:
            # we dont need to cut off useless points because rigid conv_radius is much smaller than deformable conv_radius
            # so there are not points to cut off
            new_neighb_inds = neighb_inds 
            #print(new_neighb_inds.shape, 'new_neighb_inds from else is', new_neighb_inds)

        # Get Kernel point influences h(y, Xk) [n_q_points, n_kpoints, n_neighbors] (dims 1 and 2 are swapped by transpose)
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            h_Yi_Xk = torch.ones_like(sq_distances)
            h_Yi_Xk = torch.transpose(h_Yi_Xk, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when distance = KP_extent.
            h_Yi_Xk = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            h_Yi_Xk = torch.transpose(h_Yi_Xk, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            h_Yi_Xk = radius_gaussian(sq_distances, sigma)
            h_Yi_Xk = torch.transpose(h_Yi_Xk, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')
        # Now we want to apply h_Yi_Xk so that they influence 
        # how big is part of certain weight matrices 
        # in feature dimentionality extention in certain input points

        # Our aggregation mode is 'sum', we sum influences of kernel points in this kernel location
        # In case of mode 'closest', only the closest KP can influence each point
        # if self.aggregation_mode == 'closest':
        #     neighbors_1nn = torch.argmin(sq_distances, dim=2)
        #     h_Yi_Xk *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)
        #
        # elif self.aggregation_mode != 'sum':
        #     raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
        #print(x.shape, 'x is', x)

        # Get the features of each neighborhood [n_q_points, n_neighbors, f_dim_in]
        neighb_x = gather(x, new_neighb_inds)
        #print('neighb_x', neighb_x.shape) #, 'neighb_x[0][0]', neighb_x[0][0])
#         for ind in range(len(neighb_x[0])):
#             print(neighb_x[0][ind][0], neighb_x[0][ind][1])
#         print('h_Yi_Xk is', h_Yi_Xk.shape)
        
        # Apply distance weights [n_q_points, n_kpoints, f_dim_in]
        features_projected_to_kernel_point = torch.matmul(h_Yi_Xk, neighb_x)       
        features_projected_to_kernel_point = features_projected_to_kernel_point.permute((1, 0, 2))  # permute is a 3d version of Transpose
#         print('features_projected_to_kernel_point', features_projected_to_kernel_point.shape)
#         print('self.weights', len(self.weights), len(self.weights[0]), len(self.weights[0][0]), self.weights[0][0])
        
        # Apply network weights. Kernel_outputs is [n_kpoints, n_q_points, f_dim_out]
        kernel_outputs = torch.matmul(features_projected_to_kernel_point, self.weights)  # self.weights is trained Parameter
        #print('kernel_outputs', kernel_outputs.shape)

        #ou_sum = torch.sum(kernel_outputs, dim=0)
        #print('trained torch.sum(kernel_outputs, dim=0) len is', len(ou_sum), len(ou_sum[0]), ou_sum[0])
        # Convolution sum (output features from picture 2; sum of kernel responses) [n_q_points, f_dim_out]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#


def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config):
    """
    Chooses a block based on provided name and returns it to architectures.KPFCNN.__init__
    """
    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['resnetb',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)


    elif block_name == 'simple':
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class BatchNormBlock(nn.Module):
    """
    Describes initialization and forward passing for BatchNormModule
    """
    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization,
        replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            #self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:
            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze()
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension output features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()  # initialize an object of parent-class (Module)
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim

        # set mlp as a 1-layer network with 1 matrix of weights and without bias
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)

        # add batch_normalisation of the output
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)

        # add activation function
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        """Passes x forward through mlp, batch_norm and leakyReLU
        """
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()  # Initialize an object of parenting class (Module)
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        # arguments are [n_points, d_feat] of features of points of this layer
        # and [n_points, neigh_lim[self.layer_ind - 1]] of upsampling neighbors of points of lower layer
        # for every of n_points : derive its features based on indices
        # of its upsampling neighbors and already known features of those neighbors
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
                                                                  self.layer_ind - 1)


class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a simple convolution block (KPConv, BatchNorm and leakyReLU.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_KP_extent = config.KP_extent * radius / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define KPConv of the block. Out_dim is set to first_feat_dim = 128, so hear its out_dim // 2
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             in_dim,
                             out_dim // 2,
                             current_KP_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)

        # Other opperations
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x, batch):
        """Passes x forward through KPConv, batch_norm and leakyReLU
        Takes points and neighbors from batch
        """
        #print(x.shape, 'x.shape in Simple')
        #for i in range(0, 1000):
        #    print('In Simple x[', i, '] is', x[i])

        # Choose point for q_pts and s_pts from batch
        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]  # pools is indices of s_pts-neighbors for every q_pts-point
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]  # neighbors is indices of s_pts-neighbors for every s_pts-point

        # Apply KPConv described in SimpleBlock.__init__ to chosen points
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return self.leaky_relu(self.batch_norm(x))


class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_KP_extent = config.KP_extent * radius / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        # KPConv block
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             out_dim // 4,
                             out_dim // 4,
                             current_KP_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mlp
        if in_dim != out_dim:  # when block in beginning of layer requires increase in number of features
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:  # direct shortcut
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, features, batch):
        """
        Passes features forward through ResnetBottleneckBlock.
        Takes points and neighbors from batch
        """

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)
        #print('len(x) before kpconv is', len(x))
        #print('len(x[0]) before kpconv is', len(x[0]))

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        #print('len(x) before bn-relu is', len(x))
        #print('len(x[0]) before bn-relu is', len(x[0]))

        x = self.leaky_relu(self.batch_norm_conv(x))

        #print('len(x) before unary2 is', len(x))
        #print('len(x[0]) before unary2 is', len(x[0]))

        # Second upscaling mlp
        x = self.unary2(x)
        
        #print('x[0] after unary2 is', x[0])

        # Since KPConv of strided block will perform pooling of points from s_pts to q_pts,
        # we need to perform pooling of features in the same manner as points were pooled
        if 'strided' in self.block_name:
            #print('strided len(features) before is', len(features), 'len(features[0]) before is ', len(features[0]))
            shortcut = max_pool(features, neighb_inds)
            #print('strided len(features) after is', len(features), 'len(features[0]) after is ', len(features[0]))
            #print('strided len(shortcut) after is', len(shortcut), 'len(shortcut[0]) after is ', len(shortcut[0]))
        else:
            shortcut = features
            #print('nonstrided len(features) after is', len(features), 'len(features[0]) after is ', len(features[0]))
            #print('nonstrided len(shortcut) after is', len(shortcut), 'len(shortcut[0]) after is ', len(shortcut[0]))

        shortcut = self.unary_shortcut(shortcut)
        
        #print('shortcut[0] is', shortcut[0])
        #print('x[0]+shortcut[0] is', x[0] + shortcut[0])

        return self.leaky_relu(x + shortcut)
