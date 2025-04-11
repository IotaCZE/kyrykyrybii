import pydicom
import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

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

folder = "SAMPLE_001"
ct_image, ct_slices, spacing = load_ct_series(folder)
rs = get_rs_structures(folder)

for roi in rs.StructureSetROISequence:
    print(roi.ROINumber, roi.ROIName)

contours = get_structure_contours(rs, structure_name="SpinalCord")  # or any ROIName in RS
display_ct_with_contours(ct_image, ct_slices, contours)
