import h5py
import os

root_path = '/workspace/data/GOPRO/train'
h5_folder = os.listdir(root_path)
h5_folder.sort()

for h5 in h5_folder:
    h5_path = os.path.join(root_path, h5)
    with h5py.File(h5_path, 'a') as f:
        print(f"\n### Inspecting file: {h5} ###")
        print("Existing keys before deletion:", list(f.keys()))
        
        
        if 'gen_event' in f:
            del f['gen_event']  # 그룹 또는 데이터셋 삭제
            print("Deleted 'gen_event' group.")
        else:
            print("Group 'gen_event' does not exist.")
        
        print("Existing keys after deletion:", list(f.keys()))
