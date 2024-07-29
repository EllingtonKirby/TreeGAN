import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import glob

class NuscenesObjectsDataLoader(Dataset):
    def __init__(self, root, split, real_or_generated, num_points,):
        super().__init__()
        self.dirs = glob.glob(f'{root}/{split}/**')
        self.object_name = 'generated_0' if real_or_generated=='generated' else 'original_0'
        self.npoints = num_points

    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):
        curr_dir = self.dirs[index]
        object_data = np.loadtxt(f'{curr_dir}/{self.object_name}.txt', dtype=np.float32)
        # point_set = np.zeros((self.npoints, 3))
        # point_set[:object_data.shape[0], :] = object_data[:, :3]
        label = 1
        return object_data[:, :3], label