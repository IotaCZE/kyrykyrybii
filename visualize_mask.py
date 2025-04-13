import os

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
    

#spinal cord path
spin_cord_path = '/Users/oliverklimt/projects/rakathon2025/kyrykyrybii/out/patient1/series4/SpinalCord'
slices_path = '/Users/oliverklimt/projects/rakathon2025/kyrykyrybii/out/patient1/series4/slices'

dataset = RadioProtectDataset(slices_path,spin_cord_path)

for example in dataset:

    mask = example['object']
    colored_mask = np.zeros((*mask.shape, 4))  # Initialize a transparent RGBA image

    # Set color (e.g., red) and alpha for mask == 1
    colored_mask[mask == 1] = [1, 0, 0, 0.5]  # Red with 50% opacity

    # Plot
    plt.imshow(example['slice'], cmap='gray')
    plt.imshow(colored_mask)

    plt.axis('off')  # Optional
    plt.show()