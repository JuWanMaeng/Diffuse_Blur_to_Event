import os,glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import numpy as np
from ptlflow.utils import flow_utils

def normalize_flow_to_tensor(flow):
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
    z = magnitude / 100  # 100
    z = np.clip(z, 0, 1)
    z = z * 2 - 1


    # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
    C = np.stack((x, y, z), axis=-1)  # data range [-1,1]

    return C

# imgs = glob.glob('/workspace/ptlflow/Gopro/train/flow/*.flo')
# imgs.sort()

# total_mse = 0

# for idx,flow_path in enumerate(tqdm(imgs, ncols=80)):
#     scene = flow_path.split('/')[-1][:-4]

#     flow = flow_utils.flow_read(flow_path)
#     flow = normalize_flow_to_tensor(flow)
#     out = (flow + 1) / 2


#     l_flow_path = f'/workspace/data/Gopro_my/train/{scene}/flow/flows/{scene}.flo'
#     max_flow = 10000
#     flow = flow_utils.flow_read(l_flow_path)
#     nan_mask = np.isnan(flow)
#     flow[nan_mask] = max_flow + 1
#     flow[nan_mask] = 0
#     flow = np.clip(flow, -max_flow, max_flow)
#     l_flow = normalize_flow_to_tensor(flow)
#     l_flow = (l_flow + 1) /2

#     new_scene = int(scene) + len(imgs)
#     new_scene = str(new_scene).zfill(6)
#     r_flow_path = f'/workspace/data/Gopro_my/train/{new_scene}/flow/flows/{new_scene}.flo'
#     flow = flow_utils.flow_read(r_flow_path)
#     nan_mask = np.isnan(flow)
#     flow[nan_mask] = max_flow + 1
#     flow[nan_mask] = 0
#     flow = np.clip(flow, -max_flow, max_flow)
#     r_flow = normalize_flow_to_tensor(flow)
#     r_flow = (r_flow + 1) /2


#     l_mse = np.mean((out - l_flow) ** 2)
#     r_mse = np.mean((out - r_flow) ** 2)
    
#     mse = min(r_mse,l_mse)   
#     total_mse += mse



# print(total_mse/ len(imgs))





import os,glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ptlflow.utils.flow_utils import flow_to_rgb
from ptlflow.utils import flow_utils

mother_path = '/workspace/data/Gopro_my/flows/train'
folder = os.listdir(mother_path)
folder.sort()
total_mse = 0
total_psnr = 0

for scene in tqdm(folder, ncols=80):
    img_folder = os.path.join(mother_path,scene)
    imgs = os.listdir(img_folder)
    imgs.sort()
    imgs = imgs[2:]

    mse = 100

    for img in imgs:
        img_path = os.path.join(img_folder,img)
        out = Image.open(img_path).convert('RGB')

        l_flow_path = f'/workspace/data/Gopro_my/train/{scene}/flow/flows/{scene}.flo'
        l_flow = flow_utils.flow_read(l_flow_path)
        l_flow = flow_to_rgb(l_flow,flow_max_radius=150)
   
        new_scene = int(scene) + len(folder)
        new_scene = str(new_scene).zfill(6)
        r_flow_path = f'/workspace/data/Gopro_my/train/{new_scene}/flow/flows/{new_scene}.flo'
        r_flow = flow_utils.flow_read(r_flow_path)
        r_flow = flow_to_rgb(r_flow,flow_max_radius=150)

        out = np.array(out) / 255.0
        l_flow = np.array(l_flow) / 255.0
        r_flow = np.array(r_flow) / 255.0

        l_mse = np.mean((out - l_flow) ** 2)
        l_psnr = 20*np.log10(1.0/np.sqrt(l_mse))

        r_mse = np.mean((out - r_flow) ** 2)
        r_psnr = 20*np.log10(1.0/np.sqrt(r_mse))


        mse = min(r_mse,l_mse,mse)
        
    total_mse += mse



print(total_mse/ len(folder))
