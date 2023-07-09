#! python
#
# Copyright 2021
# Authors: Mahdi Torkashvand

"""
This creates a device for the auto tracker

Usage:
    tracker.py                   [options]

Options:
    -h --help                           Show this help.
    --commands_in=HOST:PORT             Host and Port for the incomming commands.
                                            [default: localhost:5001]
    --commands_out=HOST:PORT            Host and Port for the outgoing commands.
                                            [default: localhost:5000]
    --data_in=HOST:PORT                 Host and Port for the incomming image.
                                            [default: localhost:5005]
    --data_out=HOST:PORT                Host and Port for the outgoing image.
                                            [default: localhost:5005]
    --format=UINT8_YX_512_512           Size and type of image being sent.
                                            [default: UINT8_YX_512_512]
    --name=NAME                         Device Name.
                                            [default: tracker]
    --interpolation_tracking=BOOL       Uses user-specified points to interpolate z.
                                            [default: False]
    --data_out_debug=HOST:PORT                Host and Port for the outgoing debug image.
                                            [default: localhost:5009]
"""

# Modules
from __future__ import annotations
import time
import json
from typing import Tuple
import onnxruntime


import zmq
import cv2 as cv
import numpy as np
from cfm.model.ort_loader import load_ort
from docopt import docopt
from cfm.devices.pid_controller import PIDController

from cfm.zmq.array import TimestampedSubscriber, TimestampedPublisher
from cfm.zmq.publisher import Publisher
from cfm.zmq.subscriber import ObjectSubscriber
from cfm.zmq.utils import parse_host_and_port
from cfm.devices.utils import array_props_from_string

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
## TrackerDevice
class TrackerDevice():
    """This creates a device that subscribes to images from a camera
    and sends commands to the motors"""

    def __init__(
            self,
            commands_in: Tuple[str, int, bool],
            commands_out: Tuple[str, int],
            data_in: Tuple[str, int, bool],
            data_out: Tuple[str, int],
            fmt: str,
            interpolation_tracking:bool,
            name: str,
            data_out_debug: Tuple[str, int] = None
        ):
        
        # Tracking Parameters
        ## Camera Related
        self.SMALLES_TRACKING_OBJECT = 200  # For Eggs to L1
        self.PIXEL_RATIO_WORM_MAX = 0.25
        self.TRACKEDWORM_SIZE_FLUCTUATIONS = 0.25
        self.TRACKEDWORM_CENTER_SPEED = 100
        self.missing_worm_idx = 0
        self.MISSING_WORM_TOLERANCE = 45
        ## Masking Worm
        self.MASK_WORM_THRESHOLD = 110
        self.MASK_WORM_THRESHOLD_BLURED = self.MASK_WORM_THRESHOLD*1.1
        self.MASK_MEDIAN_BLUR = 9
        self.MASK_KERNEL_BLUR = 5
        ## Sharpness Calculation
        self.SHRPNESS_SAMPLES = 30
        self.SHARPNESS_PADDING = 5
        self.SHARPNESS_MIN_LENGTH = 20
        # CAUTION: `interpolation_tracking` is passed as `str`
        self.interpolation_tracking = interpolation_tracking.lower() == 'true' if isinstance(interpolation_tracking, str) else interpolation_tracking
        self.are_points_set = False
        self.points = np.zeros((3, 3)) * np.nan
        self.curr_point = np.zeros(3)
        self.N = np.zeros(3) * np.nan
        self.isN = False
        ## Tracker Class
        self.xytracker = XYTrackerRatio(tracker = self)
        

        np.seterr(divide = 'ignore')
        self.status = {}
        self.data_out = data_out
        self.data_out_debug = data_out_debug
        self.data_in = data_in
        self.poller = zmq.Poller()
        self.name = name
        (self.dtype, _, self.shape) = array_props_from_string(fmt)  # UINT8_YX_512_512 -> dtype = uint8 , shape = (512,512)
        self.out = np.zeros(self.shape, dtype=self.dtype)
        self.data = np.zeros(self.shape)
        # y is the first index and x is the second index in the image
        self.y_worm = self.shape[0]//2
        self.x_worm = self.shape[1]//2
        self.pid_controller = PIDController(Kpy=15, Kpx=15, Kiy=0, Kix=0, Kdy=0, Kdx=0, SPy=self.shape[0]//2, SPx=self.shape[1]//2)

        ## Z Tracking
        self.shrp_idx = 0
        self.shrp_hist_size = 30
        self.shrp_hist = np.zeros(self.shrp_hist_size)  # TODO change value
        self.VZ_MAX = 8
        self.vz = self.VZ_MAX

        ## Tracked Worm Info
        self.trackedworm_size = None
        self.trackedworm_center = None

        self.tracking = False
        self.running = True

        self.ort_session = None

        self.command_publisher = Publisher(
            host=commands_out[0],
            port=commands_out[1],
            bound=commands_out[2])
        
        self.data_publisher = TimestampedPublisher(
            host=self.data_out[0],
            port=self.data_out[1],
            bound=self.data_out[2],
            shape=self.shape,
            datatype=self.dtype)
        
        # self.data_publisher_debug = TimestampedPublisher(
        #     host=self.data_out_debug[0],
        #     port=self.data_out_debug[1],
        #     bound=self.data_out_debug[2],
        #     shape=self.shape,
        #     datatype=self.dtype) if self.data_out_debug is not None and self.name != "tracker_gcamp" else None

        self.command_subscriber = ObjectSubscriber(
            obj=self,
            name=name,
            host=commands_in[0],
            port=commands_in[1],
            bound=commands_in[2])

        self.data_subscriber = TimestampedSubscriber(
            host=self.data_in[0],
            port=self.data_in[1],
            bound=self.data_in[2],
            shape=self.shape,
            datatype=self.dtype)

        self.poller.register(self.command_subscriber.socket, zmq.POLLIN)
        self.poller.register(self.data_subscriber.socket, zmq.POLLIN)

        time.sleep(1)
        self.publish_status()

        # DEBUG
        self.debug_T = 15*5*2
        self.debug_idx = 0
        self.debug_durations = np.zeros(self.debug_T)
        if self.interpolation_tracking:
            self.print(f"\n{type(self.interpolation_tracking)}\n")
            self.print(f"\n{self.interpolation_tracking}\n")
            self.print("\nTRACKING BY INTERPOLATION\n")
        self.idx_detector_send = 0
        self.idx_detector_receive = 0
    

    def set_point(self, i):
        self.command_publisher.send(f"teensy_commands get_pos {self.name} {i}")
        self.send_log(f"get_pos({self.name},{i}) request sent")

    def set_pos(self, i, x, y, z):
        idx = i % 3 - 1
        self.points[idx] = [x, y, z]
        self.send_log(f"set_pos called with i->idx: {i}->{idx}, pos ({x}, {y}, {z})")
        if not np.any(np.isnan(self.points)):
            # Surface Normal
            A = self.points[1] - self.points[0]
            B = self.points[1] - self.points[2]
            self.N = np.cross(A, B)
            self.N /= np.linalg.norm(self.N)
            if self.N[2] < 0 :
                self.N *= -1
            self.d0 = np.dot(self.N, self.points[0])
            self.isN = True
            # Surface Parameters
            self.send_log({
                "desc": "Surface Normal Parameters",
                "normal": str(self.N),
                "intercept": str(self.d0),
            })

    def estimate_vz_by_interpolation(self):
        if not self.isN:
            self.vz = 0
            return
        d = np.dot(self.curr_point, self.N) - self.d0
        sign = -np.sign(d)
        magnitude = (self.VZ_MAX * 2) * ( np.abs(d) / (1+np.abs(d)) )
        self.vz = int( sign * magnitude )

        

    def get_curr_pos(self):
        self.command_publisher.send(f"teensy_commands get_curr_pos {self.name}")
        self.send_log(f"get_curr_pos() request sent")

    def set_curr_pos(self, x, y, z):
        self.curr_point[:] = [x, y, z]
        self.send_log(f"received position ({x},{y},{z})")

    def detect_using_XY_tracker(self, img):
        img_size = img.size
        ### y is the first index and x is the second index in the image
        ny, nx = img.shape[:2]
        
        ## Find Objects
        img_mask_objects = self.xytracker.img_to_objects_mask(img)
        ## Worm(s)/Egg(s) Mask
        _ys, _xs = np.where(~img_mask_objects)
        img_mask_worms = img_mask_objects
        _, labels, rectangles, centroids = cv.connectedComponentsWithStats(
            img_mask_worms.astype(np.uint8)
        )  # output: num_labels, labels, stats, centroids
        centroids = centroids[:,::-1]  # convert to ij -> xy
        # Merge Overlaping or Close Rectangles
        rectangles = rectangles[
            rectangles[:,-1] >= self.SMALLES_TRACKING_OBJECT
        ]  # DEBUG disabling small rectangles
        labels, rectangles_merged, centroids = merge_rectangles(
            labels, rectangles,
            threshold_ratio_overlap=0.1, threshold_distance=30
        )
        labels_background = set(labels[_xs, _ys])  # labels of background in worm compoenents
        label_values, label_counts = np.unique(labels.flatten(), return_counts=True)
        candidates_info = []
        for label_value, label_count, centroid in zip(label_values, label_counts, centroids):
            # Skip Background's Connected Component
            if label_value in labels_background:
                continue
            # If around the size of a worm
            pixel_ratio = label_count / img_size
            ## TODO: double check if previous condition works better of this one
            ## pixel_ratio <= self.PIXEL_RATIO_WORM_MAX and label_count >= self.SMALLES_TRACKING_OBJECT
            if label_count >= self.SMALLES_TRACKING_OBJECT:
                candidates_info.append([
                    labels == label_value,
                    centroid
                ])
        ## Select Worm from Candidates
        self.found_trackedworm = False
        img_mask_trackedworm = None
        idx_closest = None
        _d_center_closest = None
        if len(candidates_info) > 0:
            _center_previous = self.trackedworm_center \
                if self.tracking and self.trackedworm_center is not None else np.array([nx/2, ny/2])
            _size_lower = self.trackedworm_size*(1.0-self.TRACKEDWORM_SIZE_FLUCTUATIONS) if self.tracking and self.trackedworm_size is not None else 0.0
            _size_upper = self.trackedworm_size*(1.0+self.TRACKEDWORM_SIZE_FLUCTUATIONS) if self.tracking and self.trackedworm_size is not None else 0.0
            for idx, (mask, center) in enumerate(candidates_info):
                # Candidate Info
                _size = mask.sum()
                _d_center = np.max(np.abs(center - _center_previous))
                # Check Distance (center of image or worm if tracked)
                is_close_enough = _d_center <= self.TRACKEDWORM_CENTER_SPEED
                # Check Size if Tracked a Worm
                if _size_upper != 0.0:
                    is_close_enough = is_close_enough and (_size_lower <= _size <= _size_upper)
                # If Close Enough
                if is_close_enough:
                    self.found_trackedworm = True
                    if _d_center_closest is None or _d_center < _d_center_closest:
                        idx_closest = idx
                        _d_center_closest = _d_center
                        img_mask_trackedworm = mask
                        if self.tracking:
                                self.trackedworm_size = _size
                                self.trackedworm_center = center.copy()
        if self.debug_idx == 0:
            # self.print('\n\n###')
            # self.print(f"Mode Pixel: {find_mode_pixel(img)}")
            # self.print(f"Number of candidates: {len(candidates_info)}")
            # self.print(str(candidates_info))
            pass

        # Visualize Informations
        img_annotated = img.copy()

        # if self.data_publisher_debug is not None:
        #     img_debug = img_mask_objects.astype(np.uint8)*255
        #     self.data_publisher_debug.send(img_debug)
        

        # Worm Mask
        if self.found_trackedworm:
            ## Extend Worm Boundary
            img_mask_trackedworm_blurred = cv.blur(
                img_mask_trackedworm.astype(np.float32),
                (self.MASK_KERNEL_BLUR, self.MASK_KERNEL_BLUR)
            ) > 1e-4
            xs, ys = np.where(img_mask_trackedworm_blurred)
            x_min, x_max = minmax(xs)
            y_min, y_max = minmax(ys)
            self.img_trackedworm_cropped = img[
                x_min:(x_max+1),
                y_min:(y_max+1)
            ]
            self.x_worm = (x_min + x_max)//2
            self.y_worm = (y_min + y_max)//2

            # Z Calculations
            self.shrp_hist[self.shrp_idx] = self.calc_img_sharpness(
                self.img_trackedworm_cropped
            )
        else:
            self.x_worm, self.y_worm = None, None
            self.img_trackedworm_cropped = None
            self.vz = None

        return img_annotated

    def set_onnxmodel_path(self, fp_onnx):
        self.ort_session = onnxruntime.InferenceSession(fp_onnx)
        return

    def detect_using_NNModel(self, img):

        self.found_trackedworm = True
        self.idx_detector_send += 1
        self.vz = None

        ## Detect using Model
        if self.ort_session is not None:
            img_cropped = img[56:-56,56:-56]
            batch_1_400_400 = {
                'input': np.repeat(
                    img_cropped[None, None, :, :], 3, 1
                ).astype(np.float32)
            }
            ort_outs = self.ort_session.run( None, batch_1_400_400 )
            self.y_worm, self.x_worm = ort_outs[0][0].astype(np.int64) + 56
        else:  # No ORT Session
            self.y_worm, self.x_worm = self.shape[0]//2, self.shape[1]//2

        # Visualize Informations
        img_annotated = img.copy()
        img_annotated = cv.circle(img_annotated, (int(self.y_worm), int(self.x_worm)), radius=10, color=255, thickness=2)
        img_annotated = cv.circle(img_annotated, (256, 256), radius=2, color=255, thickness=2)  # Center of image

        # if self.data_publisher_debug is not None:
        #     img_debug = img_annotated
        #     self.data_publisher_debug.send(img_debug)

        return img_annotated

    def process(self):
        """This processes the incoming images and sends move commands to zaber."""
        # Get Image Data
        msg = self.data_subscriber.get_last()
        if msg is not None:
            self.data = msg[1]
        
        # Base Cases
        ## None Message
        if msg is None:
            self.print("received message is NONE")
            return
        ## DEBUG GCaMP
        if self.name == "tracker_gcamp":
            self.data_publisher.send(self.data)
            return

        # Find Worm
        img = self.data

        # img_annotated = self.detect_using_XY_tracker(img)
        img_annotated = self.detect_using_NNModel(img)
        
        # Behavior Displayer
        self.data_publisher.send(img_annotated)


        # If no worm and tracking, stop moving to avoid collision
        if self.tracking and not self.found_trackedworm:
            # TODO why this happens that worm seems to be missing?!
            self.missing_worm_idx += 1
            # Interpolation Method
            self.vz = 0
            if self.interpolation_tracking:
                self.estimate_vz_by_interpolation()
            # Panic!!!!!
            if self.missing_worm_idx > self.MISSING_WORM_TOLERANCE:
                # TODO change it so when tracking is lost, we can help it manually. e.g. change camera position closer to worm and it continues to track
                if self.missing_worm_idx <= (self.MISSING_WORM_TOLERANCE+100):  # send command two times to ensure stopping of all motors
                    self.set_velocities(0, 0, 0)
                else:
                    self.set_velocities(None, None, None)
                if not self.interpolation_tracking:
                    self.print("TRACKING AND NO WORM!")
            return
        elif not self.found_trackedworm:
            return
        elif self.tracking and self.found_trackedworm:
            self.missing_worm_idx = 0

        # PID
        ## Velocities XY
        self.vy, self.vx = self.pid_controller.get_velocity(self.y_worm, self.x_worm)

        ## Velocity Z
        if self.interpolation_tracking:
            self.estimate_vz_by_interpolation()
        else:
            self.estimate_vz_by_sharpness()


        # Set Velocities
        if self.tracking:
            # DEBUG
            if self.vz is not None:
                # print(self.vz)
                pass
            
            ##setting PID parameters
            self.set_velocities(-self.vy, self.vx, None)

        # Return
        return


    # DEBUG
    def change(self, value):
        value_new = self.xytracker.DEVIATION_RATIO_THRESHOLD + float(value)
        self.xytracker.DEVIATION_RATIO_THRESHOLD = min(
            max(value_new, 0.0),
            1.0
        )
        print(f"<{self.name}>@XYThresholdRatio: {self.xytracker.DEVIATION_RATIO_THRESHOLD}")

    def change_threshold(self, direction):
        _tmp = self.MASK_WORM_THRESHOLD
        self.MASK_WORM_THRESHOLD = np.clip(
            self.MASK_WORM_THRESHOLD + direction,
            0, 255
        )
        self.MASK_WORM_THRESHOLD_BLURED = self.MASK_WORM_THRESHOLD*1.1

        print(f"<{self.name}>@Threshold: {self.MASK_WORM_THRESHOLD}")
        self.send_log(f"threshold changed {_tmp}->{self.MASK_WORM_THRESHOLD}")


    def start(self):
        if not self.tracking:
            self.shrp_idx = 0
            self.send_log("starting tracking")
        self.missing_worm_idx = 0
        self.trackedworm_center = None
        self.trackedworm_size = None
        self.tracking = True
        self.pid_controller.Ix = 0
        self.pid_controller.Iy = 0
        self.pid_controller.Ex = 0
        self.pid_controller.Ey = 0

    def stop(self):
        if self.tracking:
            self.set_velocities(0, 0, 0)
            self.send_log("stopping tracking")
        self.missing_worm_idx = 0
        self.trackedworm_center = None
        self.trackedworm_size = None
        self.tracking = False
        self.pid_controller.Ix = 0
        self.pid_controller.Iy = 0
        self.pid_controller.Ex = 0
        self.pid_controller.Ey = 0


    def set_shape(self, y ,x):
        self.poller.unregister(self.data_subscriber.socket)

        self.shape = (y, x)
        self.tracker.set_shape(y, x)
        self.out = np.zeros(self.shape, dtype=self.dtype)

        self.data_subscriber.set_shape(self.shape)
        self.data_publisher.set_shape(self.shape)
        # if self.data_publisher_debug is not None and self.name != "tracker_gcamp" :
        #     self.data_publisher_debug.set_shape(self.shape)

        self.poller.register(self.data_subscriber.socket, zmq.POLLIN)
        self.publish_status()

    def shutdown(self):
        """Shutdown the tracking device."""
        self.send_log("shutdown command received")
        self.stop()
        self.running = False
        self.publish_status()

    def update_status(self):
        """updates the status dictionary."""
        self.status["shape"] = self.shape
        self.status["tracking"] = self.tracking
        self.status["device"] = self.running
        self.send_log("status update received")
        self.send_log(self.status)

    def send_log(self, msg_obj):
        """Send log data to logger device."""
        # Cases to Handle
        msg = str(msg_obj)
        if isinstance(msg_obj, dict):  # Dict/JSON
            msg = json.dumps(msg_obj, default=int)
        # Send log
        msg = "{} {} {}".format( time.time(), self.name, msg )
        self.command_publisher.send(f"logger {msg}")
        return
    
    def interpolate_z_tracking(self, yes_no):
        if isinstance(yes_no, bool):
            self.interpolation_tracking = yes_no
        elif isinstance(yes_no, int):
            self.interpolation_tracking = yes_no == 1
        else:
            self.interpolation_tracking = yes_no.lower() == 'true'


    def publish_status(self):
        """Publishes the status to the hub and logger."""
        self.update_status()
        self.command_publisher.send("hub " + json.dumps({self.name: self.status}, default=int))
        self.send_log({
            self.name: self.status
        })
        # self.command_publisher.send("logger " + json.dumps({self.name: self.status}, default=int))

    def run(self):
        """This subscribes to images and adds time stamp
         and publish them with TimeStampedPublisher."""

        while self.running:

            sockets = dict(self.poller.poll())

            if self.command_subscriber.socket in sockets:
                self.command_subscriber.handle()

            elif self.data_subscriber.socket in sockets:
                # DEBUG
                _debug_start = time.time()
                # Process
                self.process()
                # DEBUG
                self.debug_durations[self.debug_idx] = time.time() - _debug_start
                if self.debug_idx >= (self.debug_T-1):
                    _debug_avg = self.debug_durations.mean()*1000
                    msg = f"Average Process Time: {_debug_avg:>6.4f}"
                    # self.print(msg)
                    self.send_log(msg)
                self.debug_idx = (self.debug_idx+1)%self.debug_T
    
    # Processing Methods
    def calc_line_sharpness(self, arr1d):
        i_min, i_max = np.argmin(arr1d), np.argmax(arr1d)
        v_min, v_max = arr1d[i_min], arr1d[i_max]
        il, ir = (i_min, i_max) if i_min <= i_max else (i_max, i_min)
        if v_max == v_min:
            return np.nan
        values = (arr1d[il:(ir+1)] - v_min)/(v_max - v_min)
        shrp = np.sum( np.diff(values)**2 )
        return shrp

    def calc_horizontal_sharpness(self, img, x, points, points_extended):
        # Slice Bounds
        y_min, y_max = minmax(points[1][
            points[0] == x
        ])
        y_ext_min, y_ext_max = minmax(points_extended[1][
            points_extended[0] == x
        ])
        # Enough Points
        if y_max - y_min <= self.SHARPNESS_MIN_LENGTH:
            return np.nan
        # Sharpnesses
        shrp1 = self.calc_line_sharpness(img[
            x, y_ext_min:(y_min+1+self.SHARPNESS_PADDING)
        ])
        shrp2 = self.calc_line_sharpness(img[
            x, (y_max-self.SHARPNESS_PADDING):(y_ext_max+1)
        ])
        shrp = (shrp1+shrp2)/2
        # Return
        return shrp

    def calc_vertical_sharpness(self, img, y, points, points_extended):
        # Slice Bounds
        x_min, x_max = minmax(points[0][
            points[1] == y
        ])
        x_ext_min, x_ext_max = minmax(points_extended[0][
            points_extended[1] == y
        ])
        # Enough Points
        if x_max - x_min <= self.SHARPNESS_MIN_LENGTH:
            return np.nan
        # Sharpnesses
        shrp1 = self.calc_line_sharpness(img[
            x_ext_min:(x_min+1+self.SHARPNESS_PADDING), y
        ])
        shrp2 = self.calc_line_sharpness(img[
            (x_max-self.SHARPNESS_PADDING):(x_ext_max+1), y
        ])
        shrp = (shrp1+shrp2)/2
        # Return
        return shrp

    def calc_img_sharpness(self, img):
        # Threshold and Blur
        # DEBUG TODO: change these calls to 'self.MASK_WORM_THRESHOLD' and 'self.MASK_WORM_THRESHOLD_BLURED'
        # since they are now being dynamically changed -> previously they were constant
        img_thrshld = img <= self.MASK_WORM_THRESHOLD
        img_thrshld_medblur = cv.medianBlur(
            img, self.MASK_MEDIAN_BLUR
        ) <= (self.MASK_WORM_THRESHOLD_BLURED)
        # Mask and Extentsion
        img_mask = img_thrshld & img_thrshld_medblur
        img_mask_extended = cv.blur(
            img_thrshld_medblur.astype(np.float32),
            (self.MASK_KERNEL_BLUR, self.MASK_KERNEL_BLUR)
        ) > 0
        # Points
        xs, ys = np.where(img_mask)
        xs_unique, ys_unique = np.unique(xs), np.unique(ys)
        points = np.array([xs,ys])
        points_extended = np.array(np.where(img_mask_extended))
        # Empty
        if len(points) == 0 or len(points_extended) == 0:
            print(f"<{self.name}>@ZERO POINTS")
            return np.nan
        # Sharpness
        samples_x = np.random.choice(xs_unique, size=self.SHRPNESS_SAMPLES, replace=False) \
            if self.SHRPNESS_SAMPLES < len(xs_unique) else xs_unique
        shrpn_x_avg = np.nanmean([
            self.calc_horizontal_sharpness(
                img,
                x,
                points, points_extended
            ) for x in samples_x
        ])
        samples_y = np.random.choice(ys_unique, size=self.SHRPNESS_SAMPLES, replace=False) \
            if self.SHRPNESS_SAMPLES < len(ys_unique) else ys_unique
        shrpn_y_avg = np.nanmean([  # TODO: This gives warning!!!
            self.calc_vertical_sharpness(
                img,
                y,
                points, points_extended
            ) for y in samples_y
        ])
        shrp = (shrpn_x_avg+shrpn_y_avg)/2
        # Return
        return shrp
    
    # Print
    def print(self, msg):
        print(f"<{self.name}>@ {msg}")
        return
    
    # Set Velocities
    def set_velocities(self, vx, vy, vz):
        if vx is not None:
            self.command_publisher.send("teensy_commands movex {}".format(vx))
        if vy is not None:
            self.command_publisher.send("teensy_commands movey {}".format(vy))
        if vz is not None:
            self.command_publisher.send("teensy_commands movez {}".format(vz))
        self.send_log(f"set velocities ({vx},{vy},{vz})")
        self.get_curr_pos()
        return

    # Estimate Vz by Sharpness
    def estimate_vz_by_sharpness(self):
        if self.shrp_idx == (self.shrp_hist_size-1):
            # Coarse Sharpness Change
            _n = self.shrp_hist_size//2
            shrp_old = np.nanmean(self.shrp_hist[:_n])
            shrp_new = np.nanmean(self.shrp_hist[_n:])
            # Stop or Move
            if np.isnan(shrp_old) or np.isnan(shrp_new) or self.vz is None:
                self.vz = 0
            elif self.vz == 0:
                self.vz = self.VZ_MAX
            elif shrp_old > shrp_new:
                self.vz = -self.vz
            # Shift New to Old
            self.shrp_hist[:_n] = self.shrp_hist[_n:]
            self.shrp_idx = _n
            # DEBUG
            # self.print(f"Sharpness Old/New: {shrp_old:>6.4f}/{shrp_new:>6.4f}, vz: {self.vz}")
        else:
            self.shrp_idx += 1
        return

def main():
    """Create and start auto tracker device."""

    arguments = docopt(__doc__)
    device = TrackerDevice(
        commands_in=parse_host_and_port(arguments["--commands_in"]),
        data_in=parse_host_and_port(arguments["--data_in"]),
        commands_out=parse_host_and_port(arguments["--commands_out"]),
        data_out=parse_host_and_port(arguments["--data_out"]),
        fmt=arguments["--format"],
        interpolation_tracking=arguments["--interpolation_tracking"],
        name=arguments["--name"],
        data_out_debug=parse_host_and_port(arguments["--data_out_debug"]),)

    device.run()

if __name__ == "__main__":
    main()
