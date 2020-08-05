import os
import json
import time

import cv2
import numpy as np

from srl_zoo.utils import printYellow
from state_representation.client import SRLClient

#for sobel edges
import scipy.ndimage as ndimage

from PIL import Image


#for normal
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from rlbench.backend.utils import rgb_handles_to_mask


def lstq(A, Y, lamb=0.01):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        print (torch.matrix_rank(A))
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = lstq(A_dash, Y_dash)
        return x
    
def least_square_normal_regress(x_depth3d, size=9, gamma=0.15, depth_scaling_factor=1, eps=1e-5):    
    stride=1

    # xyz_perm = xyz.permute([0, 2, 3, 1])
    xyz_padded = F.pad(x_depth3d, (size//2, size//2, size//2, size//2), mode='replicate')
    xyz_patches = xyz_padded.unfold(2, size, stride).unfold(3, size, stride) # [batch_size, 3, width, height, size, size]
    xyz_patches = xyz_patches.reshape((*xyz_patches.shape[:-2], ) + (-1,))  # [batch_size, 3, width, height, size*size]
    xyz_perm = xyz_patches.permute([0, 2, 3, 4, 1])

    diffs = xyz_perm - xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs = diffs / xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs[..., 0] = diffs[..., 2]
    diffs[..., 1] = diffs[..., 2]
    xyz_perm[torch.abs(diffs) > gamma] = 0.0

    A_valid = xyz_perm * depth_scaling_factor                           # [batch_size, width, height, size, 3]

    # Manual pseudoinverse
    A_trans = xyz_perm.permute([0, 1, 2, 4, 3]) * depth_scaling_factor  # [batch_size, width, height, 3, size]

    A = torch.matmul(A_trans, A_valid)

    A_det = torch.det(A)
    A[A_det < eps, :, :] = torch.eye(3, device="cuda")
    
    A_inv = torch.inverse(A)
    b = torch.ones(list(A_valid.shape[:4]) + [1], device=x_depth3d.device)
    lstsq = A_inv.matmul(A_trans).matmul(b)
#     except Exception as e:
#         print(e)
# #         # More stable, but slow (and must be done on cpu)
# #         return torch.zeros((x_depth3d.shape[0], 3, 256, 256)).to(x_depth3d.device) # give up
#         lstsq = None
#         for i in range(100):
#             try:
#                 b = torch.ones(list(A_valid.shape[:4]) + [1], device='cpu')
#                 lstsq = torch.pinverse(A_valid.cpu()).matmul(b).to(x_depth3d.device)
#                 break
#             except:
#                 pass

    lstsq = lstsq / torch.norm(lstsq, dim=3).unsqueeze(3)
    lstsq[lstsq != lstsq] = 0.0
    return -lstsq.squeeze(-1).permute([0, 3, 1, 2])
    

class LeastSquareModule(nn.Module):

    def __init__(self, gamma=0.15, beta=9):
        self.cached_cr = None
        self.shape = None
        self.patch_size = beta
        self.z_depth_thresh = gamma
        super().__init__()

    
    def forward(self, x_depth, field_of_view_rads):
        x_depth3d, cached_cr = reproject_depth(x_depth, field_of_view_rads, cached_cr=self.cached_cr, max_depth=1.)
#         plt.imshow(x_depth3d[0].transpose((1,2,0)))
        if self.cached_cr is None:
            self.cached_cr = cached_cr
        return least_square_normal_regress(x_depth3d, size=self.patch_size, gamma=self.z_depth_thresh)


def reproject_depth(depth, field_of_view, cached_cr=None, max_depth=1.):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """


    dx, dy = torch.tensor(depth.shape[2:4]) - 1
    cx, cy = torch.tensor([dx, dy]) / 2

    fx, fy = torch.tensor([[depth.shape[2]], [depth.shape[3]]], device=field_of_view.device, dtype=torch.float32) \
                / (2. * torch.tan(field_of_view.float() / 2.).unsqueeze(0))

    if cached_cr is None:
        cols, rows = depth.shape[2], depth.shape[3]
        c, r = torch.tensor(np.meshgrid(np.arange(cols), np.arange(rows), sparse=False), device=field_of_view.device, dtype=torch.float32)
    else:
        c, r = cached_cr

    z = depth.squeeze(1) * max_depth

    c = c.float()
    cx = cx.float()
    cy = cy.float()
    x = z * ((c - cx).unsqueeze(0) / fx.unsqueeze(1).unsqueeze(1))
    y = z * ((r - cy).unsqueeze(0) / fy.unsqueeze(1).unsqueeze(1))
    return torch.stack((x, y, z), dim=1), cached_cr


class EpisodeSaver(object):
    """
    Save the experience data from a gym env to a file
    and notify the srl server so it learns from the gathered data
    :param name: (str)
    :param max_dist: (float)
    :param state_dim: (int)
    :param globals_: (dict) Environments globals
    :param learn_every: (int)
    :param learn_states: (bool)
    :param path: (str)
    :param relative_pos: (bool)
    """

    def __init__(self, name, env_name=None,
                 path='data/'):
        super(EpisodeSaver, self).__init__()
        self.name = name
        self.data_folder = path + name
        self.path = path
        try:
            os.makedirs(self.data_folder)
        except OSError:
            printYellow("Folder already exist")

        self.actions = []
        self.rewards = []
        self.images = []
        self.episode_starts = []
        self.ground_truth_states = []
        self.images_path = []
        self.episode_step = 0
        self.episode_idx = -1
        self.episode_folder = None
        self.episode_success = False
        self.n_steps = 0

        self.env_name = env_name

    def saveImage(self, observation):
        """
        Write an image to disk
        :param observation
        """
        image, depth, mask = observation
        image_path = "{}/{}/frame{:06d}".format(self.data_folder, self.episode_folder, self.episode_step)
        self.images_path.append(image_path)

        def save(img, type):
            img = Image.fromarray(img)
            img.save(fp="{}_{}.png".format(image_path, type))
        
        #Base Image
        rgb = (image.transpose(1,2,0)*255).astype(np.uint8)
        save(rgb, "rgb")

        #Depth
        img_depth = (depth[0] * 255).astype(np.uint8)
        save(img_depth, "depth")


        #Normals estimation
        depth = depth.reshape(288,288)
        depth = depth[ np.newaxis, np.newaxis,...]
        depth = (depth - np.min(depth))/np.max(depth)
        depth = torch.from_numpy(depth.copy()).float().cuda()
        fov = torch.tensor(60.0 * 3.1415926/180).float().cuda()
        normals = LeastSquareModule(2,3)(depth*1000, fov) 
        normals = np.array((normals[0].cpu() + 1 )/2)
        normals = (normals.transpose(1,2,0)*255).astype(np.uint8)
        save(normals, "normal")

        #Sobel Edges
        def sobel_transform(x, blur=0):
            image = x.mean(axis=0)
            blur = ndimage.filters.gaussian_filter(image, sigma=blur, )
            sx = ndimage.sobel(blur, axis=0, mode='constant')
            sy = ndimage.sobel(blur, axis=1, mode='constant')
            sob = np.hypot(sx, sy)
            # edge = torch.FloatTensor(sob).unsqueeze(0)
            return sob
        sobel = sobel_transform(image, 0)
        sobel = np.array(sobel/np.max(sobel) * 255).astype(np.uint8)
        save(sobel, "sobel")

        #Image Segmentation
        def map_new_classes(x):
            #x is the handle
            env_classes = ['ResizableFloor_5_25_visibleElement', 'Panda_leftfinger_visible', 'Panda_rightfinger_visual', 'Panda_gripper_visual', 'Panda_link7_visual', 'Panda_link6_visual', 'Panda_link5_visual', 'Panda_link4_visual', 'Panda_link3_visual', 'Panda_link2_visual', 'Panda_link1_visual', 'Panda_link0_visual', 'workspace', 'diningTable_visible', 'Wall1', 'Wall2', 'Wall3']
            new_inds = [0,1,1,2,2,2,2,2,2,2,2,2,3,3,0,0,0]
            mapping_task_to_scene_objects = {
                'reach_target_easy': {
                    'target': 4
                },
                'slide_block_to_target': {
                    'block': 4,
                    'target':4
                },
                'pick_and_lift': {
                    'pick_and_lift_target': 4,
                    'success_visual':4
                }
            }
            name = Object.get_object_name(int(x))
            if name == '':
                return 4
            if name == 'Floor':
                return 0
            try:
                ind = new_inds[env_classes.index(name)]
            except:
                ind = 4
            return ind
            
        ordered = np.vectorize(map_new_classes)(mask)
        save(ordered.astype(np.uint8), 'segment_img')

        #Sobel Edges 3d
        def sobel_transform_3d(x, blur=0):
            # image = x.mean(axis=0)
            blur = ndimage.filters.gaussian_filter(x, sigma=blur, )
            sx = ndimage.sobel(blur, axis=0, mode='constant')
            sy = ndimage.sobel(blur, axis=1, mode='constant')
            sob = np.hypot(sx, sy)
            # edge = torch.FloatTensor(sob).unsqueeze(0)
            return sob
        sobel = sobel_transform_3d(np.array(depth.cpu())[0][0], 0)

        sobel = np.array(sobel/np.max(sobel) * 255).astype(np.uint8)
        save(sobel, "sobel_3d")
        
        def denoise(img, std):
            # takes in rgb, which is 256, 256, 3 shaped and 0 - 255 ints
            return np.clip(((img / 255 + np.random.normal(0,std,img.shape)) * 255), 0, 255)
        denoise = denoise(rgb,0.1).astype(np.uint8)
        save(denoise, "denoise")

    def reset(self, observation, ground_truth):
        """
        Called when starting a new episode
        :param observation: 
        :param ground_truth: (numpy array)
        """
        if len(self.episode_starts) == 0 or self.episode_starts[-1] is False:
            self.episode_idx += 1

            self.episode_step = 0
            self.episode_success = False
            self.episode_folder = "record_{:03d}".format(self.episode_idx)
            os.makedirs("{}/{}".format(self.data_folder, self.episode_folder), exist_ok=True)

            self.episode_starts.append(True)
            self.ground_truth_states.append(ground_truth)
            self.saveImage(observation)

    def step(self, observation, action, reward, done, ground_truth_state):
        """
        :param observation
        :param action: (int)
        :param reward: (float)
        :param done: (bool) whether the episode is done or not
        :param ground_truth_state: (numpy array)
        """
        
        self.episode_step += 1
        self.n_steps += 1
        self.rewards.append(reward)
        self.actions.append(action)
        if reward > -0.01:
            self.episode_success = True

        if not done:
            self.episode_starts.append(False)
            self.ground_truth_states.append(ground_truth_state)
            
            self.saveImage(observation)
        else:   
            # Save the gathered data at the end of each episode
            self.save()
    
    def save(self):
        """
        Write data and ground truth to disk
        """
        # # Sanity checks
        # assert len(self.actions) == len(self.rewards)
        # assert len(self.actions) == len(self.episode_starts)
        # assert len(self.actions) == len(self.images_path)
        # assert len(self.actions) == len(self.ground_truth_states)

        data = {
            'rewards': np.array(self.rewards),
            'actions': np.array(self.actions),
            'episode_starts': np.array(self.episode_starts)
        }

        ground_truth = {
            'ground_truth_states': np.array(self.ground_truth_states),
            'images_path': np.array(self.images_path)
        }
        print("Saving preprocessed data...")
        np.savez('{}/preprocessed_data.npz'.format(self.data_folder), **data)
        np.savez('{}/ground_truth.npz'.format(self.data_folder), **ground_truth)


class LogRLStates(object):
    """
    Save the experience data (states, normalized states, actions, rewards) from a gym env to a file
    during RL training. It is useful to debug SRL models.
    :param log_folder: (str)
    """

    def __init__(self, log_folder):
        super(LogRLStates, self).__init__()

        self.log_folder = log_folder + 'log_srl/'
        try:
            os.makedirs(self.log_folder)
        except OSError:
            printYellow("Folder already exist")

        self.actions = []
        self.rewards = []
        self.states = []
        self.normalized_states = []

    def reset(self, normalized_state, state):
        """
        Called when starting a new episode
        :param normalized_state: (numpy array)
        :param state: (numpy array)
        """
        # self.episode_starts.append(True)
        self.normalized_states.append(normalized_state)
        self.states.append(np.squeeze(state))

    def step(self, normalized_state, state, action, reward, done):
        """
        :param normalized_state: (numpy array)
        :param state: (numpy array)
        :param action: (int)
        :param reward: (float)
        :param done: (bool) whether the episode is done or not
        """
        self.rewards.append(reward)
        self.actions.append(action)

        if not done:
            self.normalized_states.append(normalized_state)
            self.states.append(np.squeeze(state))
        else:
            # Save the gathered data at the end of each episode
            self.save()

    def save(self):
        """
        Write data to disk
        """
        # Sanity checks
        assert len(self.actions) == len(self.rewards)
        assert len(self.actions) == len(self.normalized_states)
        assert len(self.actions) == len(self.states)

        data = {
            'rewards': np.array(self.rewards),
            'actions': np.array(self.actions),
            'states': np.array(self.states),
            'normalized_states': np.array(self.normalized_states),
        }

        np.savez('{}/full_log.npz'.format(self.log_folder), **data)
        np.savez('{}/states_rewards.npz'.format(self.log_folder),
                 **{'states': data['states'], 'rewards': data['rewards']})
        np.savez('{}/normalized_states_rewards.npz'.format(self.log_folder),
                 **{'states': data['normalized_states'], 'rewards': data['rewards']})
