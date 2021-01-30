# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def cloud_segmentation_test(self, net, test_loader, config, req_last_min_potential_stairstep=10, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        # self.test_probs consists of zeros and has size (n_points, 3)
        # test_loader.dataset.input_labels consists of labels from [0,1,2] for n_points
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]
        print('len(self.test_probs) is', len(self.test_probs))

        # Create folders for saving
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
            if not exists(join(test_path, 'potentials')):
                makedirs(join(test_path, 'potentials'))
            iou_log_path = join(test_path, "IoU_log.txt")
        else:
            test_path = None

        # If validation, compute list of length nc_model: [N_points_with_label_0, N_points_with_label_1, N_points_with_label_2]
#         if test_loader.dataset.set == 'validation':
#             val_proportions = np.zeros(nc_model, dtype=np.float32)
#             i = 0
#             for label_value in test_loader.dataset.label_values:
#                 if label_value not in test_loader.dataset.ignored_labels:
#                     val_proportions[i] = np.sum([np.sum(labels == label_value)
#                                                  for labels in test_loader.dataset.validation_labels])
#                     i += 1
#         else:
#             val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min_potential_stairstep = 0  #-0.5  # -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        
        mIoU_smooth = []
        IoUs_smooth = []

        # Start test loop
        #for _ in range(2):
        #if True:
        #while True:  # kuramin changed
        zeros_only = 0
        while last_min_potential_stairstep < req_last_min_potential_stairstep:
            print('Initialize workers')

            print('len(test_loader.dataset.potentials[0]) bef', len(test_loader.dataset.potentials[0]))
            print('test_loader.dataset.potentials[0] bef', test_loader.dataset.potentials[0])
            # work with every batch provided by test_loader
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                # if i == 0:
                #     print('Done in {:.1f}s'.format(t[1] - t[0]))

                #print('self.device.type', self.device.type)
                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)               

                t += [time.time()]

                # Get probabilities, s_points, lengths, input_inds and cloud inds
                # of the whole batch which consists of several balls
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                print('len(outputs)', len(outputs))
                print(np.sum(np.argmax(stacked_probs, axis = 1) == 0, axis = 0))
                if len(outputs) == np.sum(np.argmax(stacked_probs, axis = 1) == 0, axis = 0):
                    zeros_only += 1
                
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

#                 print('len(batch.points)', len(batch.points))
#                 print('len(batch.points[0])', len(batch.points[0]))
#                 print('len(batch.lengths)', len(batch.lengths))
#                 print('len(batch.lengths[0])', len(batch.lengths[0]))
#                 print('batch.lengths[0]', batch.lengths[0])
                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                # for every ball of the batch
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]
                    
                    #print('len(points)', len(points))

                    # Leave inds and probs of only those points which are within test_radius_ratio
                    if 0 < test_radius_ratio < 1:
                        # mask will be 1 for every point within radius test_radius_ratio * config.in_radius from ball center
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2  # will be 1 for every point within
                        inds = inds[mask]
                        probs = probs[mask]

                    # Use probabilities of MASKED points to update probs in whole cloud
                    # New value of test_probs will depend a lot on old value
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length
                    
                    print('len(self.test_probs[c_i])', len(self.test_probs[c_i]))

                    print('b_i, len(probs)', b_i, len(probs))
                    print('probs', probs)

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Finished the batch e{:03d}-i{:04d} => testing is ready for {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # By now test_loader finished providing batches, probabilities from every MASKED ball of every batch were
            # used to update test_probs
            #print('self.test_probs.shape', self.test_probs.shape)

            # Now we want to distribute probabilities and predictions to the very original cloud.
            # Batching process used potentials to take batches uniformly, so now potentials of the cloud are changed
            # Update minimum of potentials
            print('len(test_loader.dataset.potentials)', len(test_loader.dataset.potentials))
            print('len(test_loader.dataset.potentials[0])', len(test_loader.dataset.potentials[0]))
            print('test_loader.dataset.potentials', test_loader.dataset.potentials)
            
            print('len(test_loader.dataset.min_potentials)', len(test_loader.dataset.min_potentials))
            print('test_loader.dataset.min_potentials', test_loader.dataset.min_potentials)
                        
            print('len(test_loader.dataset.argmin_potentials)', len(test_loader.dataset.argmin_potentials))
            print('test_loader.dataset.argmin_potentials', test_loader.dataset.argmin_potentials)
            
            new_min_potential = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.4f}'.format(test_epoch, new_min_potential))
            print('len(self.test_probs)', len(self.test_probs))
            print('len(self.test_probs[0])', len(self.test_probs[0]))
            print('self.test_probs', self.test_probs)
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])
            print('bef if last_min_potential_stairstep', last_min_potential_stairstep, 'new_min_potential', new_min_potential)

            # Every time the minimal potential of the cloud is more than the next integer value last_min_potential_stairstep, the next if-code will be launched.
            # This if-code will reproject votes to the original cloud (every non-segmented point gets probs from its closest input_tree neighbor), 
            # calculate IoUs (if we are in validation mode) and save clouds of potentials, probs for each class and preds
            # Such an approach with last_min_potential_stairstep is used because min_potential increment can be very small (for big clouds) 
            # or very big (for tiny clouds). However if increment of min_potential is big, we dont want it to be the first and only decider. We want more voters
            
            # Save predicted cloud
            if new_min_potential > last_min_potential_stairstep + 1:

                print('Entered if because new_min_potential > last_min_potential_stairstep + 1', new_min_potential, last_min_potential_stairstep)

                # Save real IoU once in a while
                #if int(np.ceil(new_min_potential)) % 1000 == 0:
                #if last_min_potential_stairstep > req_last_min_potential_stairstep:
                #if True:
                    
                # Project predictions
                print('\nReproject Vote #{:d}'.format(int(np.floor(new_min_potential))))
                t1 = time.time()
                proj_probs = []
                for i, file_path in enumerate(test_loader.dataset.files):

                    # test_loader.dataset.test_proj consists of proj_inds, assigned during sampling of the cloud
                    print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)
                    print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                    print(test_loader.dataset.test_proj[i][:5])

                    # Reproject probs on the evaluations points
                    # Probs will be test probabilities for those members of sampled cloud
                    # which are closest neighbors of the original cloud
                    probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]

                    # proj_probs will collect such probs from several training clouds
                    proj_probs += [probs]

                print('len(proj_probs)', len(proj_probs))
                print('len(proj_probs[0])', len(proj_probs[0]))
                # Now we want to distribute values of proj_probs to the whole original cloud
                # calculate IoUs if we are in validation mode

                t2 = time.time()
                print('Done in {:.1f} s\n'.format(t2 - t1))

                # Show vote results
                if test_loader.dataset.set == 'validation':  # if real labels are known, we compute IoUs
                    print('Confusion on full clouds')
                    t1 = time.time()
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels (we dont have probabilities for them)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                        # Get the predicted labels from calculated before
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                        # Get the true values of labels and calculate Confusions
                        targets = test_loader.dataset.validation_labels[i]
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    print('Confusions on full cloud before regroup', Confs)    
                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0)
                    print('Confusions on full cloud after regroup', C)    

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = 'e{:03d} ---- mIoU and IoUs on full clouds {:5.2f} | '.format(test_epoch, 100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print('-' * len(s))
                    print(s)
                    print('-' * len(s) + '\n')

                    # Write acc to gridsearch file (kuramin added)
                    mIoU_smooth.append(mIoU)
                    print('mIoU_smooth before crop', mIoU_smooth)
                    if len(mIoU_smooth) > 30:
                        mIoU_smooth = mIoU_smooth[1:]
                    print('mIoU_smooth after crop', mIoU_smooth)

                    IoUs_smooth.append(IoUs)                            
                    if len(IoUs_smooth) > 30:
                        IoUs_smooth = IoUs_smooth[1:]

                    # kuramin added
                    s += '\n' 
                    with open(iou_log_path, "a") as myfile:
                        myfile.write(s)

                # Save predictions
                print('Saving clouds')
                t1 = time.time()
                for i, file_path in enumerate(test_loader.dataset.files):

                    # Get coordinates only from the file of point cloud
                    points = test_loader.dataset.load_evaluation_points(file_path)

                    # Get the predicted labels
                    preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                    # Save predictions
                    cloud_name = file_path.split('/')[-1]
                    test_name = join(test_path, 'predictions', cloud_name)
                    write_ply(test_name,
                              [points, preds],
                              ['x', 'y', 'z', 'preds'])  # kuramin commented saving clouds

                    # Save probabilities
                    test_name2 = join(test_path, 'probs', cloud_name)
                    prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                  for label in test_loader.dataset.label_values]
                    write_ply(test_name2,
                              [points, proj_probs[i]],
                              ['x', 'y', 'z'] + prob_names)  # kuramin commented saving clouds

                    # Save potentials
                    pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                    pot_name = join(test_path, 'potentials', cloud_name)
                    pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                    write_ply(pot_name,
                              [pot_points.astype(np.float32), pots],
                              ['x', 'y', 'z', 'pots'])  # kuramin commented saving clouds

#                         # Save ascii preds
#                         if test_loader.dataset.set == 'test':
#                             if test_loader.dataset.name.startswith('Semantic3D'):
#                                 ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
#                             else:
#                                 ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
#                             np.savetxt(ascii_name, preds, fmt='%d')

                t2 = time.time()
                print('Done in {:.1f} s\n'.format(t2 - t1))
            
                # If new_min_potential was higher than last_min_potential_stairstep, update it
                last_min_potential_stairstep += 1 
        
            test_epoch += 1

            print('before if-break last_min_potential_stairstep', last_min_potential_stairstep, 'new_min_potential', new_min_potential, 'req_last_min_potential_stairstep', req_last_min_potential_stairstep)
            
            # Break when reaching number of desired votes
#             if last_min_potential_stairstep > req_last_min_potential_stairstep:   # kuramin commented out
#                 break   

        # kuramin added
        mIoU_aver = np.sum(mIoU_smooth) / len(mIoU_smooth)
        mIoU_var = np.var(mIoU_smooth)
        print('mIoU_aver from inside is', mIoU_aver)
        print('mIoU_var from inside is', mIoU_var)
        config.mIoU_aver = mIoU_aver
        config.mIoU_var = mIoU_var 
        
        # kuramin added
        IoUs_aver = np.sum(IoUs_smooth, axis=0) / len(IoUs_smooth)
        IoUs_var = np.var(IoUs_smooth, axis=0)
        print('IoUs_aver from inside is', IoUs_aver)
        print('IoUs_var from inside is', IoUs_var)
        config.IoUs_aver = IoUs_aver
        config.IoUs_var = IoUs_var 
        
        print('zeros_only', zeros_only)
                
        return
