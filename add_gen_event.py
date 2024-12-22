import h5py
import numpy as np
import os

my_data_path = '/workspace/data/Gopro_Event_Train'


root_path = '/workspace/data/GOPRO/train'
h5_folder = os.listdir(root_path)
h5_folder.sort()

for h5 in h5_folder:
    h5_path = os.path.join(root_path, h5)
    h5_name = h5[:-3]   # GOPR0384_11_00
    with h5py.File(h5_path, 'a') as f:
        imgs = f['images']

        if 'gen_event' not in f:
            gen_event_group = f.create_group('gen_event')
            print("Created new group: 'gen_event'")
        else:
            gen_event_group = f['gen_event']
            print("Group 'gen_event' already exists.")

        # 각 이미지에 대응하는 gen_event 데이터 생성 및 저장
        for img in imgs:
            blur_img = imgs[f'{img}'][:]  # 이미지 데이터 불러오기

            number = img[8:]
            gen_event_key = f'image{number}'
            gen_event_path = os.path.join(my_data_path, h5_name, gen_event_key, 'out.npy' )
            gen_event_data = np.load(gen_event_path)
            gen_event_data = gen_event_data.transpose(2,0,1)


            gen_event_group.create_dataset(f'{img}', data=gen_event_data, dtype=np.float32)
 

        #     break
        # break
        f.close()