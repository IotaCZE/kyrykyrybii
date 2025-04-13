"""
Data annotation tool
"""
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

example = next(iter(dataset))


mask = example['object']
colored_mask = np.zeros((*mask.shape, 4))  # Initialize a transparent RGBA image

# Set color (e.g., red) and alpha for mask == 1
colored_mask[mask == 1] = [1, 0, 0, 0.5]  # Red with 50% opacity

import matplotlib.pyplot as plt
import numpy as np

# Settings
width, height =  example['slice'].shape  # Canvas size
drawing = False
brush_size = 3

# Arrays
background = example['slice']  # Grayscale background

# Convert grayscale to RGB for drawing
background_rgb = (background)#.astype(np.float32)
background_rgb = np.stack([background_rgb]*3, axis=-1)

# Copy for drawing
canvas = background_rgb.copy()

def on_press(event):
    global drawing
    drawing = True
    update_canvas(event)

def on_release(event):
    global drawing
    drawing = False

def update_canvas(event):
    if event.xdata is not None and event.ydata is not None and drawing:
        x, y = int(event.xdata), int(event.ydata)
        y = np.clip(y, 0, height-1)
        x = np.clip(x, 0, width-1)
        y0, y1 = max(0, y - brush_size), min(height, y + brush_size)
        x0, x1 = max(0, x - brush_size), min(width, x + brush_size)

        # Draw green directly on canvas
        canvas[y0:y1, x0:x1] = [0, 255, 0]  # RGB green

        img.set_data(canvas)
        fig.canvas.draw()

# Plot setup
fig, ax = plt.subplots()
img = ax.imshow(canvas)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', update_canvas)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.title("Draw on Background (Green)")
plt.axis('off')
plt.show()