from torch.utils import data as data
import random
import numpy as np
import torch
import cv2, os
from ptlflow.utils import flow_utils


class Flow_dataset(data.Dataset):

    def __init__(self, blur_txt_path, seed=None):
        super(Flow_dataset, self).__init__()
   
        # blur path will change to flow path
        self.blur_txt_path = blur_txt_path
        self.paths = []
        self.max_magnitude = 147  # M

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


        ### resize or random crop ###
        flow_h, flow_w = flow.shape[0],flow.shape[1]
        target_h, target_w = 768, 768

        if flow_h > target_h and flow_w  > target_w:
            flow = self.random_crop(flow, patch_size=[target_h,target_w])

        else:
            flow = cv2.resize(flow, (target_w,target_h), interpolation=cv2.INTER_LANCZOS4)

        ### augment flow ###
        # Randomly select an augmentation command
        commands = ['hflip', 'vflip' , 'noop']
        command = np.random.choice(commands)
        if command != 'noop':
            flow = self.augment_flow(flow,command)


        ### normalize_flow_to_tensor ###
        n_flow = self.normalize_flow_to_tensor(flow)


        flow_tensor = torch.from_numpy(n_flow.transpose(2, 0, 1)).float()  # [3,H,W]

        return {
            'flow_path': flow_path,
            'flow': flow_tensor
        }

    def __len__(self):
        return len(self.paths)


    def random_crop(self, flow, patch_size):


        if not isinstance(flow, list):
            flow = [flow]


        h_lq, w_lq, _ = flow[0].shape
        new_h, new_w = patch_size[0], patch_size[1]

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - new_h)
        left = random.randint(0, w_lq - new_w)

        # crop corresponding flow patch

        flow = [
            v[top:top + new_h, left:left + new_w, ...]
            for v in flow
        ]

        if len(flow) == 1:
            flow = flow[0]

        return flow

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

        # Set M to the largest blur magnitude in the set (maximum of the magnitude array)
        M = magnitude.max()
        
        # Normalize the magnitude to [0, 1] range for the z component
        z = magnitude / self.max_magnitude
        z = np.clip(z, 0, 1) 


        # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
        C = np.stack((x, y, z), axis=-1)

        return C
    

if __name__ == "__main__":
    # Define the path to the text file with blur paths
    blur_txt_path = "/workspace/Marigold/dataset/train/train_vae.txt"
    dataset = Flow_dataset(blur_txt_path, seed=42)

    # Test dataset length
    print(f"Dataset length: {len(dataset)}")

    # Test first sample
    sample = dataset[0]
    print(f"Sample flow path: {sample['flow_path']}")
    print(f"Sample flow tensor shape: {sample['flow'].shape}")