from collections import defaultdict
import pydicom
import os
import numpy as np
# import SimpleITK as sitk
from matplotlib import pyplot as plt

def group_ct_series(folder):
    series_dict = defaultdict(list)
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if ds.Modality == "CT":
                series_uid = ds.SeriesInstanceUID
                series_dict[series_uid].append(path)
        except:
            continue
    return series_dict  # key: SeriesInstanceUID, value: list of file paths


def load_ct_from_series(file_list):
    slices = [pydicom.dcmread(f) for f in file_list]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices])
    spacing = [float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])]
    return image, slices, spacing



def find_ct_series_uid_for_rs(rs_file):
    rs = pydicom.dcmread(rs_file)
    try:
        ref_series_uid = rs.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0] \
                             .RTReferencedSeriesSequence[0].SeriesInstanceUID
        return ref_series_uid, rs
    except:
        return None, None
    

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

    

def process_all(folder):
    ct_series_dict = group_ct_series(folder)
    print(f"Found {len(ct_series_dict)} CT series.")

    for f in os.listdir(folder):
        if not f.startswith("RS"):
            continue
        rs_path = os.path.join(folder, f)
        series_uid, rs = find_ct_series_uid_for_rs(rs_path)
        if series_uid is None or series_uid not in ct_series_dict:
            print(f"Skipping RS {f}: no matching CT series.")
            continue

        print(f"Processing RS {f} → CT Series UID: {series_uid}")
        ct_image, ct_slices, spacing = load_ct_from_series(ct_series_dict[series_uid])

        for roi in rs.StructureSetROISequence:
            roi_name = roi.ROIName
            contours = get_structure_contours(rs, roi_name)
            if contours:
                print(f"Displaying structure: {roi_name}")
                display_ct_with_contours(ct_image, ct_slices, contours)


def display_ct_with_contours(ct_image, ct_slices, contours):
    z_positions = [s.ImagePositionPatient[2] for s in ct_slices]
    
    print(len(contours))
    for contour in contours:
        z = contour[0][2]
        try:
            idx = min(range(len(z_positions)), key=lambda i: abs(z_positions[i] - z))
            slice_img = ct_image[idx]
            x = contour[:, 0]
            y = contour[:, 1] #- 256
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

def filter_series(pairs_ct_rs, contours):
    keep = []
    for pair in pairs_ct_rs:
        rs = pair[1]
        print(f"RS file: {rs.file_meta.MediaStorageSOPInstanceUID}")
        for roi in rs.StructureSetROISequence:
            if roi.ROIName in contours:
                print(f"Found wanted contour: ROINumber: {roi.ROINumber}, ROIName: {roi.ROIName}")
                keep.append(pair)
            else:
                print(f"No wanted contour found.")
                # pairs_ct_rs.remove(pair)
    return keep


folder = "Rackaton_Data/SAMPLE_004"

series_dict = group_ct_series(folder)
print(f"Found {len(series_dict)} CT series.")
for uid, files in series_dict.items():
    print(f"SeriesInstanceUID: {uid}, #Slices: {len(files)}")

pairs_ct_rs = []
for f in os.listdir(folder):
    if "RS" in f:
        path = os.path.join(folder, f)
        series_uid, rs = find_ct_series_uid_for_rs(path)
        pairs_ct_rs.append((series_uid, rs))
        print(f"{f} → SeriesInstanceUID: {series_uid}")

#rs_file =  f'SAMPLE_001/{pairs_ct_rs[0][0]}' #"path/to/your/folder/RS123.dcm"

print(len(pairs_ct_rs))
pairs_ct_rs = filter_series(pairs_ct_rs, ["SpinalCord", "GTV", "PTV_all", "CTV_Mid01", "Glnd_Submand_R"])

print(len(pairs_ct_rs))

rs = pairs_ct_rs[0][1]
series_uid = pairs_ct_rs[0][0]

print(f"RS file: {rs.file_meta.MediaStorageSOPInstanceUID}")
print("Available structures:")
for roi in rs.StructureSetROISequence:
    print(f"→ ROINumber: {roi.ROINumber}, ROIName: {roi.ROIName}")


# Load CT
ct_image, ct_slices, spacing = load_ct_from_series(series_dict[series_uid])

# Choose a structure (e.g., "Bladder" or the name you printed above)
structure_name = "SpinalCord"
contours = get_structure_contours(rs, structure_name)

# Display
display_ct_with_contours(ct_image, ct_slices, contours)


'''from collections import Counter
structure_counter = Counter()

folder = "SAMPLE_001"
for f in os.listdir(folder):
    if "RS" in f:
        rs_path = os.path.join(folder, f)
        _, rs = find_ct_series_uid_for_rs(rs_path)
        for roi in rs.StructureSetROISequence:
            structure_counter[roi.ROIName] += 1

print("Structure frequency across RS files:")
for k, v in structure_counter.items():
    print(f"{k}: {v} RS files")'''