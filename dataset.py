import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from sampling_alter import prior_guided_patch_sampling
import warnings


warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated.*")

class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None, seed=30):
        self.seed = seed
        self.data = pd.read_csv(csv_path)
        ref_name = self.data.iloc[:, 0].tolist()
        dis_name = self.data.iloc[:, 1].tolist()
        self.ref_image_name = ref_name
        self.dis_image_name = dis_name
        self.transforms = transform
    
    def __len__(self):
        return len(self.ref_image_name)
    
    def __getitem__(self, idx):
        dis_image_basename = os.path.splitext(self.dis_image_name[idx])[0]
        ref_image_basename = os.path.splitext(self.ref_image_name[idx])[0]
        
        for index, row in self.data.iterrows():
            if row[1] == self.dis_image_name[idx]:
                label = row[2]

        dis_patch_path = f'/mnt/10T/lzy/2025_TCSVT/data896/dis/{dis_image_basename}_dis_patches.npy'
        ref_patch_path = f'/mnt/10T/lzy/2025_TCSVT/data896/ref/{ref_image_basename}_ref_patches.npy'
        position_path = f'/mnt/10T/lzy/2025_TCSVT/data896/position/{dis_image_basename}_positions.npy'
        
        ref_patchs = np.load(ref_patch_path)
        dis_patchs = np.load(dis_patch_path)
        position = np.load(position_path)

        if os.path.exists(dis_patch_path) and os.path.exists(ref_patch_path) and os.path.exists(position_path):
            
            dis_patchs = np.load(dis_patch_path)
            ref_patchs = np.load(ref_patch_path)
            position = np.load(position_path)
        else:
            
            ref_image_path = os.path.join('/home/iqateam/team/media1/ZJY/DOIQA/database/JUFE_10k/final_ref_430', self.ref_image_name[idx])
            dis_image_path = os.path.join('/home/iqateam/team/media1/ZJY/DOIQA/database/JUFE_10k/final_dis_10320', self.dis_image_name[idx])
            ref_image = Image.open(ref_image_path)
            dis_image = Image.open(dis_image_path)
            
            dis_image = self.transforms(dis_image)
            ref_image = self.transforms(ref_image)
            
            ref_patchs, position = prior_guided_patch_sampling(ref_image, patch_num=10, p_h=0.2, p_m=0.6, p_l=0.2, seed=self.seed)
            dis_patchs, _ = prior_guided_patch_sampling(dis_image, patch_num=10, p_h=0.2, p_m=0.6, p_l=0.2, seed=self.seed)
    
            np.save(dis_patch_path, dis_patchs)
            np.save(ref_patch_path, ref_patchs)
            np.save(position_path, position)
        
        sample = (dis_patchs, ref_patchs, label)

        return sample, position

