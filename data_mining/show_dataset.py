import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch

import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np

from scipy.ndimage import binary_fill_holes

class RadioProtectDataset(Dataset):
    def __init__(self, slices_dir, objs_dir):
        super().__init__()
        self.slices_dir = slices_dir
        self.objs_dir = objs_dir

        self.avail_slices = []
        path = Path(objs_dir)

        for item in path.iterdir():
            if item.is_file() and item.suffix == ".npy":
                self.avail_slices.append(item.stem)
        print(f"INFO: Loaded {len(self.avail_slices)} examples.")

    def __len__(self):
        return len(self.avail_slices)
    
    def __getitem__(self, index):
        # get slice
        id = self.avail_slices[index]
        slice_path = os.path.join(self.slices_dir,f"slice{id}.npy")
        slice_scan = np.load(slice_path)
        obj_path = os.path.join(self.objs_dir,f"{id}.npy")
        obj_patch = binary_fill_holes(np.load(obj_path))

        # get object
        sample = {'slice': slice_scan, 'object': obj_patch.astype(int)}
        return sample
    
spin_cord_path = '/Users/oliverklimt/projects/rakathon2025/kyrykyrybii/out/patient1/SpinalCord'
slices_path = '/Users/oliverklimt/projects/rakathon2025/kyrykyrybii/out/patient1/slices'
if len(sys.argv) == 3:
    slices_path = sys.argv[-2]
    spin_cord_path = sys.argv[-1]
else:
    print("Using default paths for slices and objects.")

dataset = RadioProtectDataset(slices_path,spin_cord_path)
for example in dataset:


    fig, ax = plt.subplots(1, 1)

    # Display the background image
    background_array = example['slice']
    foreground_array = example['object']
    alpha=0.5
    ax.imshow(background_array, extent=[0, background_array.shape[1], 0, background_array.shape[0]])

    # Display the foreground image with transparency
    ax.imshow(foreground_array, alpha=alpha, extent=[0, foreground_array.shape[1], 0, foreground_array.shape[0]])
    plt.show(block=True)