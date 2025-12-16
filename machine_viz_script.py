import cv2 
import numpy as np
from matplotlib import pyplot as plt

# STEP 1: Load image
img_bgr = cv2.imread("test_image.png")
img_bgr_copy = img_bgr.copy()

# Show original
cv2.imshow("Original", img_bgr_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale (make copy)
img_gray = cv2.cvtColor(img_bgr_copy, cv2.COLOR_BGR2GRAY)
img_gray_copy = img_gray.copy()

# Show grayscale
cv2.imshow("Grayscale", img_gray_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# STEP 2: Histogram Equalization

img_eq = cv2.equalizeHist(img_gray_copy)
img_eq_copy = img_eq.copy()

# Show original grayscale
hist_compare = np.hstack((img_gray, img_eq))

cv2.imshow(
    "Grayscale (LEFT) | Hist Equalization (RIGHT)",
    hist_compare
)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Histogram comparison 
hist_orig = cv2.calcHist([img_gray], [0], None, [256], [0,256])
hist_eq   = cv2.calcHist([img_eq],   [0], None, [256], [0,256])

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(hist_orig, color='black')
plt.title("Histogram — Original Grayscale")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")

plt.subplot(1,2,2)
plt.plot(hist_eq, color='black')
plt.title("Histogram — AfterEqualization")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")

plt.tight_layout()
plt.show()

#histograms look a bit strange, most pixels are background and concentrated around
#zero however the equalized hist has more variety

# STEP 3: CLAHE (local histogram equalization)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_gray_copy)
img_clahe_copy = img_clahe.copy()

# Show all iterations
side_by_side = np.hstack((img_gray, img_eq, img_clahe))

cv2.imshow("Grayscal (LEFT)| Hist Equalization (MID) | CLAHE (RIGHT)", side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()

#CLAHE has more detail within the bones seemingly 

# STEP 4: Filtering (applied to CLAHE image)
# Apply filters
gauss_5 = cv2.GaussianBlur(img_clahe_copy, (5, 5), 0)
median_3 = cv2.medianBlur(img_clahe_copy, 3)
bilateral = cv2.bilateralFilter(
    img_clahe_copy,
    d=7,
    sigmaColor=50,
    sigmaSpace=50
)

#hstack +v stack grid
top_row = np.hstack((img_clahe_copy, gauss_5))
bottom_row = np.hstack((median_3, bilateral))
grid = np.vstack((top_row, bottom_row))

cv2.imshow(
    "Post-Filtering - Top: CLAHE | Gaussian   Bottom: Median | Bilateral",
    grid
)
cv2.waitKey(0)
cv2.destroyAllWindows()

#median filtering looks to be the best option going forward based on visual examination

#STEP 5: Canny Edge Detector and Trackbar
def nothing(x):
    pass

# Use CLAHE image as input (same goal as notebook)
img_track = median_3.copy()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#starting image used the median filter

# Trackbars: starting pos, max value, callback
cv2.createTrackbar('T1:low', 'image', 50, 255, nothing)
cv2.createTrackbar('T2:high', 'image', 150, 255, nothing)

while True:
    current1 = cv2.getTrackbarPos('T1', 'image')
    current2 = cv2.getTrackbarPos('T2', 'image')

    # Ensure valid Canny thresholds
    if current2 < current1:
        current2 = current1

    edges = cv2.Canny(img_track, current1, current2, apertureSize=3)

    # Side-by-side display (same as pool example)
    result = np.hstack((img_track, edges))
    cv2.imshow('image', result)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC to exit
        break

cv2.destroyAllWindows()

# STEP 6: Otsu thresholding on CLAHE image
img_for_mask = img_clahe_copy.copy()

# Otsu thresholding
_, bone_mask = cv2.threshold(
    img_for_mask,
    0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

bone_mask_copy = bone_mask.copy()

otsu_comp = np.hstack((img_for_mask, bone_mask))

cv2.imshow(
    "CLAHE (LEFT) | Otsu Bone Mask (RIGHT)",
    otsu_comp
)
cv2.waitKey(0)
cv2.destroyAllWindows()

# STEP 6.5: Compare thresholding methods on CLAHE image
img_thresh_base = img_clahe_copy.copy()

# Fixed threshold
_, th_binary = cv2.threshold(img_thresh_base, 120, 255, cv2.THRESH_BINARY)
_, th_binary_inv = cv2.threshold(img_thresh_base, 120, 255, cv2.THRESH_BINARY_INV)

# Otsu threshold
_, th_otsu = cv2.threshold(
    img_thresh_base, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

_, th_otsu_inv = cv2.threshold(
    img_thresh_base, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

top_row = np.hstack((th_binary, th_binary_inv))
bottom_row = np.hstack((th_otsu, th_otsu_inv))
grid = np.vstack((top_row, bottom_row))

cv2.imshow(
    "Top: Binary | Binary Inv  Bottom: Otsu | Otsu Inv",
    grid
)
cv2.waitKey(0)
cv2.destroyAllWindows()

# STEP 7: Recompute a clean edge map using your chosen settings
# (Set these to the values you liked from the trackbars)
T1 = 50
T2 = 150

# Use median-filtered image for edges 
img_for_edges = median_3.copy()        # from Step 4
edges = cv2.Canny(img_for_edges, T1, T2, apertureSize=3)

# Make sure mask is 0/255 uint8 and same size
mask = bone_mask.copy()

# Apply mask: keep edges only where mask is white
edges_masked = cv2.bitwise_and(edges, mask)

# Show before/after in a 2x2 grid:
# Top: edges | masked edges
# Bottom: mask | CLAHE
top_row = np.hstack((edges, edges_masked))
bottom_row = np.hstack((mask, img_clahe_copy))
grid = np.vstack((top_row, bottom_row))

cv2.imshow("Top: Edges | Masked Edges  Bottom: Mask | CLAHE", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()


# STEP 8: Connected components on masked edges
edges_in = edges_masked.copy()

# Ensure binary (0/255) uint8
edges_bin = (edges_in > 0).astype(np.uint8) * 255

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_bin, connectivity=8)

print("Connected components found (including background):", num_labels)

# If there are not enough components, stop early
if num_labels < 3:
    print("Not enough components to pick a 'disconnected bone'.")
    cv2.imshow("Step 8 - Masked Edges", edges_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # Sort components (exclude background label 0) by area descending
    areas = stats[1:, cv2.CC_STAT_AREA]               # labels 1.. end
    order = np.argsort(areas)[::-1] + 1               # convert back to label ids

    main_label = order[0]      # biggest component
    broken_label = order[1]    # 2nd biggest component (often the separated bone)

    print("Main label:", main_label, "Area:", stats[main_label, cv2.CC_STAT_AREA])
    print("Broken label:", broken_label, "Area:", stats[broken_label, cv2.CC_STAT_AREA])

    # Base image to overlay on (choose CLAHE or median_3)
    base = img_clahe_copy.copy()
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Mask for the broken component
    broken_mask = (labels == broken_label)

    # Color it red
    overlay[broken_mask] = (0, 0, 255)

    # Draw bounding box around the broken component
    x = stats[broken_label, cv2.CC_STAT_LEFT]
    y = stats[broken_label, cv2.CC_STAT_TOP]
    w = stats[broken_label, cv2.CC_STAT_WIDTH]
    h = stats[broken_label, cv2.CC_STAT_HEIGHT]
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show a 2x2 grid: edges | labels(vis) | base | overlay
    labels_vis = (labels.astype(np.float32) / labels.max() * 255).astype(np.uint8)

    top_row = np.hstack((edges_bin, labels_vis))
    bottom_row = np.hstack((base, cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)))

    grid = np.vstack((top_row, bottom_row))

    cv2.imshow("Disconnected Bone Highlight (Red)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
