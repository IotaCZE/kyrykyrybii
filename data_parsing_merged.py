import pydicom
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from scipy.ndimage import binary_fill_holes

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
    roi_numbers = []
    for roi in rs.StructureSetROISequence:
        if structure_name in roi.ROIName:
            roi_numbers.append(roi.ROINumber)

    for item in rs.ROIContourSequence:
        if item.ReferencedROINumber in roi_numbers:
            if hasattr(item, 'ContourSequence'):
                for c in item.ContourSequence:
                    coords = np.array(c.ContourData).reshape(-1, 3)
                    contours.append(coords)
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


def generate_patient_files(ct_image, ct_slices, contours, structure_name, patient_id, series_num, contour_masks):
    path_out = Path('./out')
    path_out.mkdir(exist_ok=True,parents=True)

    series_out_path = path_out / f"patient{patient_id}/series{series_num}"
    z_positions = [s.ImagePositionPatient[2] for s in ct_slices]

    slices_path_out = series_out_path / "slices"
    slices_path_out.mkdir(exist_ok=True,parents=True)
    
    for contour in contours:
        z = contour[0][2]
        # roi_path_out = series_out_path / structure_name
        # roi_path_out.mkdir(exist_ok=True,parents=True)
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
            bool_grid = binary_fill_holes(bool_grid).astype(np.bool)

            if structure_name in contour_masks.keys():
                if idx in contour_masks[structure_name].keys():
                    contour_masks[structure_name][idx] = np.logical_or(contour_masks[structure_name][idx], bool_grid)
                else:
                    contour_masks[structure_name][idx] = bool_grid
            else:
                contour_masks[structure_name] = {}
                contour_masks[structure_name][idx] = bool_grid 
            # if "CTV" in structure_name:
            #     if idx in ctv_combined_masks:
            #             ctv_combined_masks[idx] = np.logical_or(ctv_combined_masks[idx], bool_grid)
            #     else:
            #         ctv_combined_masks[idx] = bool_grid
            # else:
            #     obj_file = roi_path_out / f"{idx}.npy"
            #     np.save(obj_file,bool_grid)
        except Exception as e:
            print(f"Skipping contour at z={z}: {e}")
            
def check_contour_slices(target_contours, slice_shape, patient_id, series_num):
    positive_path = Path(f"./out/patient{patient_id}/series{series_num}/positive")
    positive_path.mkdir(exist_ok=True, parents=True)
    
    
    negative_path = Path(f"./out/patient{patient_id}/series{series_num}/negative")
    negative_path.mkdir(exist_ok=True, parents=True)
    
    out_dir = Path(f"./out/patient{patient_id}/series{series_num}")
    slice_files = os.listdir(out_dir/"slices")
    indxs = [f.replace('slice', '').replace('.npy', '') for f in slice_files]
    for idx in indxs:
        pos_file = positive_path/f"{idx}.npy"
        if not pos_file.is_file():
            np.save(pos_file,np.zeros(shape=slice_shape, dtype=np.bool))
        neg_file = negative_path/f"{idx}.npy"
        if not neg_file.is_file():
            np.save(neg_file,np.zeros(shape=slice_shape, dtype=np.bool))
        


# def check_contour_slices(target_contours, slice_shape, patient_id, series_num):
#     out_dir = Path(f"./out/patient{patient_id}/series{series_num}")
#     slice_files = os.listdir(out_dir/"slices")
#     indxs = [f.replace('slice', '').replace('.npy', '') for f in slice_files]
#     for contour in target_contours: 
#         contour_dir = out_dir/contour
#         contour_dir.mkdir(exist_ok=True,parents=True)
#         for idx in indxs:
#             slice_file = contour_dir/f"{idx}.npy"
#             if not slice_file.is_file():
#                 np.save(slice_file,np.zeros(shape=slice_shape, dtype=np.bool))


def process_series(rs, series_uid, series_dict, patient_id, series_num):
    # Load CT
    ct_image, ct_slices, spacing = load_ct_from_series(series_dict[series_uid])

    
    with open('target_contours.txt', 'r') as f:
        positive_conts = f.readline().strip()
        positive_conts = positive_conts.split(sep=',')
        negative_conts = f.readline().strip()
        negative_conts = negative_conts.split(sep=',')
        target_contours = {}
        target_contours["positive"] = positive_conts
        target_contours["negative"] = negative_conts

    contour_masks = {}
    for structure_name in target_contours['negative']:
        contours = get_structure_contours(rs, structure_name)

        generate_patient_files(ct_image, ct_slices, contours, structure_name, patient_id, series_num, contour_masks)
    
    for structure_name in target_contours['positive']:
        contours = get_structure_contours(rs, structure_name)

        generate_patient_files(ct_image, ct_slices, contours, structure_name, patient_id, series_num, contour_masks)
    
    

    positive_path = Path(f"./out/patient{patient_id}/series{series_num}/positive")
    positive_path.mkdir(exist_ok=True, parents=True)
    
    
    negative_path = Path(f"./out/patient{patient_id}/series{series_num}/negative")
    negative_path.mkdir(exist_ok=True, parents=True)
    pos_contours = {}
    neg_contours = {}
    for contour in contour_masks.keys():
        if contour in target_contours["positive"]:
            for idx in contour_masks[contour].keys():
                if idx in pos_contours.keys():
                    pos_contours[idx] = np.logical_or(pos_contours[idx], contour_masks[contour][idx])
                else:
                    pos_contours[idx] = contour_masks[contour][idx]
        else:
            for idx in contour_masks[contour].keys():
                if idx in neg_contours.keys():
                    neg_contours[idx] = np.logical_or(neg_contours[idx], contour_masks[contour][idx])
                else:
                    neg_contours[idx] = contour_masks[contour][idx]
        
    for idx in pos_contours.keys():
        mask = pos_contours.get(idx, np.zeros_like(ct_image[0], dtype=bool))
        np.save(positive_path / f"{idx}.npy", mask)

    
    for idx in neg_contours.keys():
        mask = neg_contours.get(idx, np.zeros_like(ct_image[0], dtype=bool))
        np.save(negative_path / f"{idx}.npy", mask)

    check_contour_slices(target_contours, ct_image[0].shape, patient_id, series_num)


def process_patient(patient_id):
    # folder = f"SAMPLE_00{patient_id}"
    folder = f"Rackaton_Data/SAMPLE_00{patient_id}"

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

    for i in range(0, 4):
        series_uid, rs = pairs_ct_rs[i]
        process_series(rs, series_uid, series_dict, patient_id, i)


if __name__ == "__main__":
    process_patient(1)


