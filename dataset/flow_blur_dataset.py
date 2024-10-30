from torch.utils import data as data

import random
import numpy as np
import torch
import cv2, os
from ptlflow.utils import flow_utils
from ptlflow.utils.flow_utils import flow_to_rgb

class Flow_Blur_dataset_C(data.Dataset):

    def __init__(self, blur_txt_path, seed=None):
        super(Flow_Blur_dataset_C, self).__init__()
   
        self.blur_txt_path = blur_txt_path
        self.paths = []
        self.max_magnitude = 100  # M
        
        # Seed 설정
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        else:
            self.seed = None

        with open(self.blur_txt_path, 'r') as file:
            for line in file:
                self.paths.append(line.strip())

    def __getitem__(self, index):

        index = index % len(self.paths)

        blur_path = self.paths[index]        
        blur_img = cv2.imread(blur_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        blur_img = (blur_img - 127.5) / 127.5
        dataset_name = blur_path.split('/')[3]  # vimeo . Monkaa, Gopro

        # Optical flow array
        flow_path = blur_path.replace('blur', 'flow/flows')
        # flow_path = flow_path.replace('png', 'pfm')
        if flow_path.split('/')[3] == 'Monkaa_my':
            flow_path = flow_path.replace('png', 'pfm')
        else:
            flow_path = flow_path.replace('png', 'flo')
        
        max_flow = 10000
        flow = flow_utils.flow_read(flow_path)

        nan_mask = np.isnan(flow)
        flow[nan_mask] = max_flow + 1
        flow[nan_mask] = 0
        flow = np.clip(flow, -max_flow, max_flow)

        # crop vimeo dataset
        if dataset_name == 'vimeo':
            flow = flow[50:-50,:,:]
            blur_img = blur_img[50:-50,:,:]


        blur_img_h, blur_img_w = blur_img.shape[0],blur_img.shape[1]
        target_h, target_w = 540,960

        if blur_img_h > target_h and blur_img_w  > target_w:
            blur_img, flow = self.paired_random_crop(blur_img, flow, patch_size=[target_h,target_w])
        else:
            blur_img = cv2.resize(blur_img,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)
            flow = cv2.resize(flow,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)

        ### augmentation ###
        # Randomly select an augmentation command
        commands = ['hflip', 'vflip' , 'noop']
        command = np.random.choice(commands)

        # augment flow 
        if command != 'noop':
            flow = self.augment_flow(flow, command)

        # augment blur 
        if command == 'hflip':
            blur_img = np.fliplr(blur_img).copy()

        elif command == 'vflip':
            blur_img = np.flipud(blur_img).copy()


        ### normalize_flow_to_tensor ###
        n_flow = self.normalize_flow_to_tensor(flow)

        flow_tensor = torch.from_numpy(n_flow.transpose(2, 0, 1)).float()  # [3,H,W]
        blur_tensor = torch.from_numpy(blur_img.transpose(2, 0, 1)).float()


        return {
            'blur': blur_tensor,
            'lq_path': blur_path,
            'flow_path': flow_path,
            'flow': flow_tensor
        }

    def __len__(self):
        return len(self.paths)


    def paired_random_crop(self,img_lqs, flow, patch_size):


        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(flow, list):
            flow = [flow]


        h_lq, w_lq, _ = img_lqs[0].shape

        new_h, new_w = patch_size[0], patch_size[1]

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - new_h)
        left = random.randint(0, w_lq - new_w)

        # crop lq patch
        img_lqs = [
            v[top:top + new_h, left:left + new_w, ...]
            for v in img_lqs
        ]

        # crop corresponding flow patch
        flow = [
            v[top:top + new_h, left:left + new_w, ...]
            for v in flow
        ]

        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        if len(flow) == 1:
            flow = flow[0]

        return img_lqs, flow
    
    def augment_flow(self, flow, command):

        if command == 'hflip':  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1

        elif command ==  'vflip':  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1

        else:
            assert 'Wrong command!'

        return flow

    def normalize_flow_to_tensor(self, flow):
        """
        Normalize the optical flow and compute the 3D tensor C with x, y, and z components.

        Parameters:
        - flow: 2D array representing optical flow.

        Returns:
        - C: 3D tensor with shape (H, W, 3), where C[..., 0] is x, C[..., 1] is y, and C[..., 2] is z.
        """
        # Calculate the magnitude of the flow vectors
        u, v = flow[:,:,0], flow[:,:,1]
        magnitude = np.sqrt(u**2 + v**2)
        
        # Avoid division by zero by setting small magnitudes to a minimal positive value
        magnitude[magnitude == 0] = 1e-8
        
        # Normalize u and v components to get unit vectors for x and y
        x = u / magnitude
        y = v / magnitude

        # Normalize the magnitude to [0, 1] range for the z component
        z = magnitude / self.max_magnitude  # 100
        z = np.clip(z, 0, 1)


        # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
        C = np.stack((x, y, z), axis=-1)

        return C
    

class Flow_Blur_dataset(data.Dataset):

    def __init__(self, blur_txt_path, seed=None):
        super(Flow_Blur_dataset, self).__init__()
   
        self.blur_txt_path = blur_txt_path
        self.paths = []

        # Seed 설정
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        else:
            self.seed = None

        with open(self.blur_txt_path, 'r') as file:
            for line in file:
                self.paths.append(line.strip())

    def __getitem__(self, index):

        index = index % len(self.paths)

        blur_path = self.paths[index]        
        blur_img = cv2.imread(blur_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        blur_img = (blur_img - 127.5) / 127.5

        # Optical flow array
        flow_path = blur_path.replace('blur', 'flow/flows')
        # flow_path = flow_path.replace('png', 'pfm')
        if flow_path.split('/')[3] == 'Monkaa_my':
            flow_path = flow_path.replace('png', 'pfm')
        else:
            flow_path = flow_path.replace('png', 'flo')
        
        max_flow = 10000
        flow = flow_utils.flow_read(flow_path)

        nan_mask = np.isnan(flow)
        flow[nan_mask] = max_flow + 1
        flow[nan_mask] = 0
        flow = np.clip(flow, -max_flow, max_flow)

        ### 2d version ###
        # flow_padded = np.pad(flow, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        # flow_rgb = flow_padded # 2d -> 3d with zero padding, actucally not rgb       


        ### rgb version ###
        flow_rgb = flow_to_rgb(flow,flow_max_radius=150)  # 150 
        flow_rgb = (flow_rgb - 127.5) / 127.5

        # Random horizontal flipping
        if np.random.rand() > 0.5:
            blur_img = np.fliplr(blur_img).copy()
            flow_rgb = np.fliplr(flow_rgb).copy()


        blur_img_h, blur_img_w = blur_img.shape[0],blur_img.shape[1]
        target_h, target_w = 480,640
        # target_h, target_w = 540,960

        if blur_img_h > target_h and blur_img_w  > target_w:
            blur_img, flow_rgb = self.paired_random_crop(blur_img, flow_rgb, patch_size=[target_h,target_w])
            # blur_img, flow_rgb = self.paired_random_crop(blur_img, flow_rgb, patch_size=[480,640])
        else:
            blur_img = cv2.resize(blur_img,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)
            flow_rgb = cv2.resize(flow_rgb,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)


        blur_tensor = torch.from_numpy(blur_img.transpose(2, 0, 1)).float()
        flow_tensor = torch.from_numpy(flow_rgb.transpose(2, 0, 1)).float()

        return {
            'blur': blur_tensor,
            'lq_path': blur_path,
            'flow_path': flow_path,
            'flow': flow_tensor
        }

    def __len__(self):
        return len(self.paths)


    def paired_random_crop(self,img_lqs, flow, patch_size):


        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(flow, list):
            flow = [flow]


        h_lq, w_lq, _ = img_lqs[0].shape

        new_h, new_w = patch_size[0], patch_size[1]

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - new_h)
        left = random.randint(0, w_lq - new_w)

        # crop lq patch
        img_lqs = [
            v[top:top + new_h, left:left + new_w, ...]
            for v in img_lqs
        ]

        # crop corresponding flow patch

        flow = [
            v[top:top + new_h, left:left + new_w, ...]
            for v in flow
        ]



        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        if len(flow) == 1:
            flow = flow[0]

        return img_lqs, flow


