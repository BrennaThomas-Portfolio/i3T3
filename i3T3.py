###########################################################################################################################
#################################### IMAGE TO MATRIX ######################################################################
###########################################################################################################################

from tkinter.ttk import Label
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage as ski
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.feature import peak_local_max


# Load image
image_matrx = iio.imread("run02_24hr_plate1_d_0001.jpg")   # CHANGE THIS for each image

# Convert to grayscale
gray = ski.color.rgb2gray(image_matrx)

# Convert to 8-bit (0-255)
gray_uint8 = img_as_ubyte(gray)

# Save as integer matrix (0-255)
np.savetxt("run02_24hr_plate1_d_0001_matrix_uint8.csv", gray_uint8, delimiter=",", fmt="%d")

print("run02_24hr_plate1_d_0001_matrix shape:", gray.shape)


###############################################################################################################################
############################################### SEGMENTATION ##################################################################
###############################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk
)
from skimage.segmentation import watershed
from skimage.segmentation import find_boundaries
from skimage.feature import local_binary_pattern
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
import imageio.v3 as iio

# Load original image and matrix
image = iio.imread("run02_24hr_plate1_d_0001.jpg")               # CHANGE THIS for each image
matrix = np.loadtxt("run02_24hr_plate1_d_0001_matrix_uint8.csv", delimiter=",")  # CHANGE THIS for each image


background_mask = matrix > 10

# Smooth matrix
blurred = gaussian(matrix, sigma=4.0)

############################################
# 1. SEGMENT CELL BODIES
############################################

body_threshold = threshold_otsu(blurred)
body_mask = (blurred < body_threshold) & background_mask

# Clean the body mask
body_mask = binary_closing(body_mask, disk(2))
body_mask = remove_small_objects(body_mask, max_size=500)
body_mask = remove_small_holes(body_mask, max_size=500)

############################################
# 2. SEGMENT NUCLEI
############################################

# Nuclei are among the darkest regions
nucleus_threshold = np.percentile(matrix, 41)
nucleus_mask = (matrix < nucleus_threshold) & background_mask

# Clean the nucleus mask
nucleus_mask = binary_opening(nucleus_mask, disk(1))
nucleus_mask = remove_small_objects(nucleus_mask, max_size=500)
nucleus_mask = remove_small_holes(nucleus_mask, max_size=500)

############################################
# 3. SPLIT TOUCHING CELL BODIES USING WATERSHED
############################################

# Label nuclei (used for nucleus-aware features)
nucleus_markers = label(nucleus_mask)

# Distance transform of body mask
distance = ndi.distance_transform_edt(body_mask)

# Find peaks inside cell blobs
peak_coords = peak_local_max(
    distance,
    labels=body_mask & background_mask,
    min_distance=10,   # increase to merge nearby peaks; decrease to split more aggressively
    threshold_abs=2
)

# Visualize peak markers
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image)
ax.imshow(body_mask, cmap="gray", alpha=0.25)
for r, c in peak_coords:
    ax.plot(c, r, "ro", markersize=2)
ax.set_title("Peak Markers Inside Of Each NIH 3T3 Fibroblast Body")
ax.axis("off")
plt.show()

# Build marker image from peaks
peak_markers = np.zeros_like(body_mask, dtype=int)
for i, (r, c) in enumerate(peak_coords, start=1):
    peak_markers[r, c] = i

# Watershed segmentation
labeled = watershed(-distance, peak_markers, mask=body_mask & background_mask)

# Count from peak markers
peak_count = peak_markers.max()

# Convert labels to color image
colored_label_image = ski.color.label2rgb(labeled, bg_label=0)

############################################
# 4. MEASURE FEATURES AND KEEP TRUE FIBROBLASTS
############################################

regions = regionprops(labeled, intensity_image=matrix)

fibroblast_objects = []

for region in regions:
    area = region.area
    perimeter = region.perimeter
    eccentricity = region.eccentricity
    major_axis = region.axis_major_length
    minor_axis = region.axis_minor_length
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
    solidity = region.solidity
    extent = region.extent

    region_mask = labeled == region.label
    nuclei_inside = np.unique(nucleus_markers[region_mask])
    nuclei_inside = nuclei_inside[nuclei_inside > 0]
    n_nuclei = len(nuclei_inside)

    fibroblast_rule = (
        area >= 1000 and area <= 800000 and
        eccentricity >= 0.0 and eccentricity <= 1.0 and
        aspect_ratio >= 0.10 and aspect_ratio <= 8.0 and
        minor_axis >= 8 and
        solidity >= 0.20 and
        extent >= 0.22 and
        n_nuclei >= 1 and n_nuclei <= 2
    )

    if fibroblast_rule:
        fibroblast_objects.append(region)

print("Peak Count:", peak_count)

fibroblast_label_set = {region.label for region in fibroblast_objects}
print("Total segmented objects found:", len(regions))

############################################
# 5. VISUALIZE
############################################

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(nucleus_mask, cmap="gray")
ax[1].set_title("Binary")
ax[1].axis("off")

ax[2].imshow(colored_label_image)
ax[2].set_title(f"NIH 3T3 Fibroblasts Counted: {len(regions)}")
ax[2].axis("off")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.imshow(colored_label_image, alpha=0.45)

for region in regions:
    y, x = region.centroid
    ax.text(
        x, y, str(region.label),
        color="yellow",
        fontsize=5,
        ha="center",
        va="center",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=0.4)
    )

ax.set_title(f"{len(regions)} Labeled NIH 3T3 Cells")
ax.axis("off")
plt.show()


####################################################################################################################################################################
##################################################### ML OBJECT ####################################################################################################
####################################################################################################################################################################


def safe_divide(a, b):
    return a / b if b not in [0, None] else 0


def get_extreme_perimeter_points(region):
    """Return top, bottom, left, and right perimeter points in full-image coordinates."""
    boundary = find_boundaries(region.image, mode="outer")
    boundary_coords = np.argwhere(boundary)

    if len(boundary_coords) == 0:
        boundary_coords = np.argwhere(region.image)

    min_row, min_col, max_row, max_col = region.bbox

    top_local = boundary_coords[np.argmin(boundary_coords[:, 0])]
    top_y = min_row + top_local[0]
    top_x = min_col + top_local[1]

    bottom_local = boundary_coords[np.argmax(boundary_coords[:, 0])]
    bottom_y = min_row + bottom_local[0]
    bottom_x = min_col + bottom_local[1]

    left_local = boundary_coords[np.argmin(boundary_coords[:, 1])]
    left_y = min_row + left_local[0]
    left_x = min_col + left_local[1]

    right_local = boundary_coords[np.argmax(boundary_coords[:, 1])]
    right_y = min_row + right_local[0]
    right_x = min_col + right_local[1]

    return {
        "top_x": top_x, "top_y": top_y,
        "bottom_x": bottom_x, "bottom_y": bottom_y,
        "left_x": left_x, "left_y": left_y,
        "right_x": right_x, "right_y": right_y
    }


def compute_skeleton_features(region_mask):
    skel = skeletonize(region_mask)
    skeleton_length = np.count_nonzero(skel)

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    nbrs = ndi.convolve(skel.astype(int), kernel, mode="constant", cval=0)
    endpoints = np.logical_and(skel, nbrs == 11)
    n_endpoints = np.count_nonzero(endpoints)

    return skeleton_length, n_endpoints


def compute_lbp_histogram(region_intensity_image, region_mask, P=8, R=1):
    """Compute normalized LBP histogram on object pixels only (P+2 bins for uniform LBP)."""
    lbp = local_binary_pattern(region_intensity_image, P=P, R=R, method="uniform")
    object_lbp = lbp[region_mask]

    n_bins = P + 2
    hist, _ = np.histogram(object_lbp, bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(float)

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


########################################
# EXPORT ML FEATURE CSV
########################################

import pandas as pd
import os

image_filename = "run02_24hr_plate1_d_0001.jpg"   # CHANGE THIS for each image
image_id = os.path.splitext(image_filename)[0]

ml_rows = []

for region in regions:
    i = region.label
    object_id = f"{image_id}_obj_{i:04d}"

    area = region.area
    perimeter = region.perimeter
    eccentricity = region.eccentricity
    major_axis = region.axis_major_length
    minor_axis = region.axis_minor_length
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0

    centroid_y, centroid_x = region.centroid

    solidity = region.solidity
    extent = region.extent
    orientation = region.orientation
    convex_area = region.convex_area
    equivalent_diameter = region.equivalent_diameter_area
    feret_diameter_max = region.feret_diameter_max

    coords = region.coords
    pixel_values = matrix[coords[:, 0], coords[:, 1]]

    mean_intensity = np.mean(pixel_values)
    min_intensity = np.min(pixel_values)
    max_intensity = np.max(pixel_values)
    std_intensity = np.std(pixel_values)

    region_mask_global = labeled == region.label
    nuclei_inside = np.unique(nucleus_markers[region_mask_global])
    nuclei_inside = nuclei_inside[nuclei_inside > 0]
    n_nuclei = len(nuclei_inside)

    nucleus_area_total = 0
    for nuc_label in nuclei_inside:
        nucleus_area_total += np.sum(nucleus_markers == nuc_label)

    nucleus_area_ratio = safe_divide(nucleus_area_total, area)
    body_to_nucleus_ratio = safe_divide(area, nucleus_area_total)

    circularity = safe_divide(4 * np.pi * area, perimeter**2)
    compactness = safe_divide(perimeter**2, 4 * np.pi * area)
    roundness = safe_divide(4 * area, np.pi * major_axis**2)
    convexity = safe_divide(convex_area, area)
    perimeter_area_ratio = safe_divide(perimeter, area)
    elongation_index = safe_divide(major_axis, minor_axis)

    skeleton_length, n_endpoints = compute_skeleton_features(region.image)
    lbp_hist = compute_lbp_histogram(region.intensity_image, region.image, P=8, R=1)

    extreme_points = get_extreme_perimeter_points(region)
    bounding_width = extreme_points["right_x"] - extreme_points["left_x"]
    bounding_height = extreme_points["bottom_y"] - extreme_points["top_y"]

    row = {
        "image_id": image_id,
        "image_filename": image_filename,
        "object_number": i,
        "object_id": object_id,

        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "eccentricity": eccentricity,
        "major_axis_length_pixels": major_axis,
        "minor_axis_length_pixels": minor_axis,
        "aspect_ratio": aspect_ratio,

        "centroid_x": centroid_x,
        "centroid_y": centroid_y,

        "solidity": solidity,
        "extent": extent,
        "orientation": orientation,

        "mean_intensity": mean_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "std_intensity": std_intensity,

        "circularity": circularity,
        "compactness": compactness,
        "roundness": roundness,
        "convex_area": convex_area,
        "convexity": convexity,
        "equivalent_diameter": equivalent_diameter,
        "feret_diameter_max": feret_diameter_max,

        "n_nuclei": n_nuclei,
        "nucleus_area_total": nucleus_area_total,
        "nucleus_area_ratio": nucleus_area_ratio,

        "skeleton_length": skeleton_length,
        "n_endpoints": n_endpoints,

        "perimeter_area_ratio": perimeter_area_ratio,
        "elongation_index": elongation_index,
        "body_to_nucleus_ratio": body_to_nucleus_ratio,

        "top_x": extreme_points["top_x"],
        "top_y": extreme_points["top_y"],
        "bottom_x": extreme_points["bottom_x"],
        "bottom_y": extreme_points["bottom_y"],
        "left_x": extreme_points["left_x"],
        "left_y": extreme_points["left_y"],
        "right_x": extreme_points["right_x"],
        "right_y": extreme_points["right_y"],

        "bounding_width": bounding_width,
        "bounding_height": bounding_height,

        "manual_image_count": 10,        # CHANGE for each image
        "algorithm_count_image": peak_count,

        "passed_fibroblast_rule": 1 if region.label in fibroblast_label_set else 0,
        "class_label": ""
    }

    for j, val in enumerate(lbp_hist):
        row[f"lbp_hist_{j}"] = val

    ml_rows.append(row)


# Save CSV
ml_df = pd.DataFrame(ml_rows)

csv_output = f"{image_id}_fibroblast_ml_features.csv"
ml_df.to_csv(csv_output, index=False)

print(f"Saved ML feature CSV: {csv_output}")
print(f"Objects exported: {len(ml_df)}")
print(ml_df.head())
