# Modules
from __future__ import annotations
import numpy as np

import cv2 as cv

from cfm.devices.tracker import TrackerDevice

# Methods
## Array Min-Max
def minmax(arr):
    return np.min(arr), np.max(arr)
## Find Mode Pixel
def find_mode_pixel(arr):
    values, counts = np.unique(arr, return_counts=True)
    value_mode = values[np.argmax(counts)]
    return value_mode
## Rectangle Area Overlap Ratio
def rects_overlap(rect1, rect2):
    x1l,y1l,w1,h1,s1 = rect1
    x1r,y1r = x1l+w1, y1l+h1
    x2l,y2l,w2,h2,s2 = rect2
    x2r,y2r = x2l+w2, y2l+h2
    # Cases
    dx, dy = 0, 0
    if x2l <= x1l <= x2r and y2l <= y1l <= y2r:  # 1: top-left
        dx = x1l-x2l
        dy = y1l-y2l
    elif x2l <= x1r <= x2r and y2l <= y1l <= y2r:  # 1: top-right
        dx = x1r-x2l
        dy = y1l-y2l
    elif x2l <= x1l <= x2r and y2l <= y1r <= y2r:  # 1: bottom-left
        dx = x1l-x2l
        dy = y1r-y2l
    elif x2l <= x1r <= x2r and y2l <= y1r <= y2r:  # 1: bottom-right
        dx = x1r-x2l
        dy = y1r-y2l
    area_intersection = dx*dy
    area_min = min(w1*h1, w2*h2)
    return area_intersection / area_min
## Rectangle Center Distance
def rects_distance_ratio(rect1, rect2, normalize=False):
    x1,y1,w1,h1,s1 = rect1
    x2,y2,w2,h2,s2 = rect2
    r1, r2 = np.sqrt(w1*h1), np.sqrt(w2*h2)
    r_max = max(r1, r2)
    distance = np.sqrt(
        (x1+w1/2-x2-w2/2)**2 + (y1+h1/2-y2-h2/2)**2
    )
    if not normalize:
        return distance
    distance_normalized = distance / r_max
    return  distance_normalized - 1.0
## Merge Rectangles
def merge_rectangles(labels, rectangles, threshold_ratio_overlap=0.10, threshold_distance = 30.0):
    rectangles_merged = [
        list(rectangles[0])
    ]
    rectangels_to_merge = [ (i+1,rect) for i, rect in enumerate(rectangles[1:,:]) ]
    while len(rectangels_to_merge) > 0:
        idx = 0
        label_current, rect_current = rectangels_to_merge[idx]
        x0,y0,w0,h0,s0 = rect_current
        x0r, y0r = x0+w0, y0+h0
        ids_merged = {idx}
        for idx, entry in enumerate(rectangels_to_merge):
            label, rect = entry
            if idx == 0:
                continue
            # Check Ratio of Overlap
            ratio_overlap = rects_overlap( rect_current, rect )
            distance = rects_distance_ratio( rect_current, rect, normalize=False )
            if ratio_overlap >= threshold_ratio_overlap or distance <= threshold_distance:
                labels[ labels == label ] = label_current
                x,y,w,h,s = rect
                xr, yr = x+w, y+h
                ids_merged.add(idx)
                x0, y0 = min(x, x0), min(y, y0)
                x0r, y0r = max(xr, x0r), max(yr, y0r)
                s0 += s
        # Remove Indices
        rectangels_to_merge = [
            entry for i, entry in enumerate(rectangels_to_merge) if i not in ids_merged
        ]
        # Store
        rectangles_merged.append([
            x0, y0, x0r-x0+1, y0r-y0+1, s0
        ])
    #
    rectangles_merged = np.array(rectangles_merged)
    centroids = rectangles_merged[:,:2] + rectangles_merged[:,2:4]/2 
    return labels, rectangles_merged, centroids


# Class
## Tracker Methods
class XYTrackerThreshold:
    """This is an instance of tracking method for XY coordinates."""
    def __init__(self, tracker: TrackerDevice) -> None:
        self.tracker = tracker
        return
    def img_to_objects_mask(self, img):
        ## Find Background
        img_blurred = cv.blur(
            img,
            (self.tracker.MASK_KERNEL_BLUR, self.tracker.MASK_KERNEL_BLUR)
        )
        img_mask_brights = img_blurred >= self.tracker.MASK_WORM_THRESHOLD
        _, labels, _, _ = cv.connectedComponentsWithStats(
            img_mask_brights.astype(np.uint8)
        )  # output: num_labels, labels, stats, centroids
        label_values, label_counts = np.unique(labels.flatten(), return_counts=True)
        img_mask_background = np.zeros_like(labels, dtype=np.bool_)  # everything is bright
        if len(label_values) > 1:  # at least one dark pixel
            for label_value, label_count in zip(label_values, label_counts):
                if label_value == 0:
                    continue
                if label_count > self.tracker.SMALLES_TRACKING_OBJECT:  # check if component is background
                    # Check if component is bright
                    _mask = labels == label_value
                    img_mask_background |= _mask
        else:
            img_mask_background[:] = True
        # Mask
        mask = ~img_mask_background
        return mask
class XYTrackerRatio:
    """This is an instance of tracking method for XY coordinates."""
    def __init__(self, tracker: TrackerDevice) -> None:
        self.tracker = tracker
        self.PIXELS_MODE = 140
        self.DEVIATION_RATIO_THRESHOLD = 0.3  # For tracking Adult
        # self.DEVIATION_RATIO_THRESHOLD = 0.2  # For tracking Egg to L1
        self.KERNEL_DILATE = np.ones((13,13))
        self.KERNEL_ERODE = np.ones((7,7))  # For Tracking Adult
        # self.KERNEL_ERODE = np.ones((3,3))  # For Tracking Egg to L1
        self.IMG_BLUR_SIZE = 5
        return
    def img_to_objects_mask(self, img):
        img_blurred = cv.blur(img, (self.IMG_BLUR_SIZE, self.IMG_BLUR_SIZE))
        # Objects
        pixels_mode = self.PIXELS_MODE
        pixels_mode = find_mode_pixel(img_blurred)
        img_blurred_deviation = np.abs(img_blurred.astype(np.float32) - pixels_mode)/pixels_mode
        img_objects = (img_blurred_deviation >= self.DEVIATION_RATIO_THRESHOLD).astype(np.float32)
        # Erode -> Remove Small Objects
        img_objects_eroded = cv.erode(
            img_objects,
            self.KERNEL_ERODE
        )
        # Dilate -> Expand
        img_objects_dilated = cv.dilate(
            img_objects_eroded,
            self.KERNEL_DILATE
        )
        # Connected Components
        _, labels, rectangles, _ = cv.connectedComponentsWithStats(
            img_objects_dilated.astype(np.uint8)
        )
        for i, rectangle in enumerate(rectangles):
            _size = rectangle[-1]
            if _size <= self.tracker.SMALLES_TRACKING_OBJECT:
                indices = labels == i
                labels[indices] = 0
        # Mask
        mask = labels > 0
        return mask
class ZTrackerDisabled:
    """This is a class to disable z tracking."""
    # TODO
    def __init__(self) -> None:
        pass
class ZTrackerFree:
    """This is a class to handle tracking in z direction in freely moving worm on agar plates."""
    # TODO
    def __init__(self) -> None:
        pass
class ZTrackerInterpolation:
    """This is a class to handle tracking in z direction when worms are put between coverslip/glass
    using marks on coverslip/worm-plane."""
    # TODO
    def __init__(self) -> None:
        pass