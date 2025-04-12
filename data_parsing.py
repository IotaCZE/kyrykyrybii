import pydicom
from pydicom import dcmread

from matplotlib import pyplot as plt
import numpy as np

import sys

import os

from pathlib import Path

from scipy.ndimage import binary_fill_holes

import pydicom
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from ct_series import *


def load_ct_series(folder):
    slices = [pydicom.dcmread(os.path.join(folder, f)) for f in os.listdir(folder) if 'CT' in f]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))  # sort by z-position
    #print(f"Loaded {len(slices)} slices.")
    #print(set([x.pixel_array.shape for x in slices]))
    for s in reversed(slices):
        if s.pixel_array.shape != (512, 512):
            slices.remove(s)

    image = np.stack([s.pixel_array for s in slices])
    spacing = [float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])]
    return image, slices, spacing


def get_rs_structures(folder):
    for f in os.listdir(folder):
        if "RS" in f:
            return pydicom.dcmread(os.path.join(folder, f))
    return None

def get_structure_contours(rs, structure_name):
    contours = []
    for roi in rs.StructureSetROISequence:
        if roi.ROIName == structure_name:
            roi_number = roi.ROINumber
            break
    else:
        return []

    for item in rs.ROIContourSequence:
        if item.ReferencedROINumber == roi_number:
            if hasattr(item, 'ContourSequence'):
                for c in item.ContourSequence:
                    coords = np.array(c.ContourData).reshape(-1, 3)
                    contours.append(coords)
            break
    return contours


def display_ct_with_contours(ct_image, ct_slices, contours):
    z_positions = [s.ImagePositionPatient[2] for s in ct_slices]
    
    for contour in contours:
        z = contour[0][2]
        try:
            idx = min(range(len(z_positions)), key=lambda i: abs(z_positions[i] - z))
            slice_img = ct_image[idx]
            x = contour[:, 0]
            y = contour[:, 1] - 256
            # Convert physical coords to pixel
            origin = ct_slices[0].ImagePositionPatient
            spacing = ct_slices[0].PixelSpacing
            px = (x - origin[0]) / spacing[0]
            py = (y - origin[1]) / spacing[1]

            plt.imshow(slice_img, cmap='gray')
            plt.plot(px, py, 'r')  # red contour
            plt.title(f"Slice {idx}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Skipping contour at z={z}: {e}")

import numpy as np

def bresenham_line(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1) using Bresenham's algorithm."""
    points = []
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x1, y1))
    return points

def project_discrete_curve_to_grid(curve_points, grid_shape):
    """
    Projects a closed discrete curve onto a grid using Bresenham's line algorithm.

    Args:
        curve_points (list of tuples or ndarray): (x, y) points in continuous space.
        grid_shape (tuple): (rows, cols) for grid size.

    Returns:
        numpy.ndarray: Boolean grid with True where the curve passes.
    """
    rows, cols = grid_shape
    grid = np.zeros((rows, cols), dtype=bool)
    num_points = len(curve_points)

    # Convert coordinates from real values to integer grid indices
    for i in range(num_points - 1):
        x0, y0 = curve_points[i]
        x1, y1 = curve_points[(i + 1) % num_points]  # Wrap around

        # Map real coordinates to grid indices
        col0 = int(np.clip(x0 / cols * cols, 0, cols - 1))
        row0 = int(np.clip(y0 / rows * rows, 0, rows - 1))
        col1 = int(np.clip(x1 / cols * cols, 0, cols - 1))
        row1 = int(np.clip(y1 / rows * rows, 0, rows - 1))

        for x, y in bresenham_line(col0, row0, col1, row1):
            if 0 <= y < rows and 0 <= x < cols:
                grid[y, x] = True

    return grid


def generate_patient_files(ct_image, ct_slices, contours, structure_name, patient_id):
    path_out = Path('./out')
    path_out.mkdir(exist_ok=True,parents=True)

    patient_out_path = path_out / f"patient{patient_id}"
    z_positions = [s.ImagePositionPatient[2] for s in ct_slices]

    slices_path_out = patient_out_path / "slices"
    slices_path_out.mkdir(exist_ok=True,parents=True)
    
    for contour in contours:
        z = contour[0][2]
        roi_path_out = patient_out_path / structure_name
        roi_path_out.mkdir(exist_ok=True,parents=True)
        try:
            idx = min(range(len(z_positions)), key=lambda i: abs(z_positions[i] - z))
            slice_img = ct_image[idx]

            slice_file = slices_path_out/f"slice{idx}.npy"
            if not slice_file.is_file():
                np.save(slice_file,slice_img)

            x = contour[:, 0]
            y = contour[:, 1] #- 256
            # Convert physical coords to pixel
            origin = ct_slices[0].ImagePositionPatient
            spacing = ct_slices[0].PixelSpacing
            px = (x - origin[0]) / spacing[0]
            py = (y - origin[1]) / spacing[1]
            px_adj = np.array(list(px)+[px[0]])
            py_adj = np.array(list(py)+[py[0]])
            p_adj = np.stack((px_adj,py_adj),axis=1)


            bool_grid = project_discrete_curve_to_grid(p_adj,slice_img.shape)
            bool_grid = binary_fill_holes(bool_grid)

            obj_file = roi_path_out / f"{idx}.npy"
            np.save(obj_file,bool_grid)
        except Exception as e:
            print(f"Skipping contour at z={z}: {e}")


def process_patient(patient_id):
    folder = f"SAMPLE_00{patient_id}"

    series_dict = group_ct_series(folder)

    pairs_ct_rs = []
    for f in os.listdir(folder):
        if "RS" in f:
            path = os.path.join(folder, f)
            series_uid, rs = find_ct_series_uid_for_rs(path)
            pairs_ct_rs.append((series_uid, rs))
            # print(f"{f} → SeriesInstanceUID: {series_uid}")

    #rs_file =  f'SAMPLE_001/{pairs_ct_rs[0][0]}' #"path/to/your/folder/RS123.dcm"
        
    # print(len(pairs_ct_rs))
    pairs_ct_rs = filter_series(pairs_ct_rs, 'target_contours.txt')

    rs = pairs_ct_rs[0][1]
    series_uid = pairs_ct_rs[0][0]

    # Load CT
    ct_image, ct_slices, spacing = load_ct_from_series(series_dict[series_uid])

    # Choose a structure (e.g., "Bladder" or the name you printed above)
    with open('target_contours.txt', 'r') as f:
        data = f.readline().strip()
        target_contours = data.split(sep=',')

    for structure_name in target_contours:
        contours = get_structure_contours(rs, structure_name)

        generate_patient_files(ct_image, ct_slices, contours, structure_name, patient_id)


if __name__ == "__main__":
    process_patient(1)


