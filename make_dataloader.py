import os

import cv2
import numpy as np
import torch


class Dataset2Class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

    def __getitem__(self, idx):
        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        # img = (img[:,:,0:1]+img[:,:,1:2]+img[:,:,2:3])/3 - if we want to make the image black and white

        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.transpose(
            2, 0, 1
        )  # put the length, width, channels in the right place

        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)

        return {"img": t_img, "label": t_class_id}
