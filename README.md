# Installation

# Getting Started

- Once the installation is complete, find the OpenAutoScope2.0 icon, and to launch the software double-click on the icon.
  The software's graphical user interface (GUI) resembles the image displayed below.
  
<p align="center">
  <img src="https://github.com/mtorkashvand/compact-flourescent-microscope/assets/31863323/b4e17906-7630-4475-9443-ec367f55f1e9" alt="Image" width="900" height="650">
</p>

### 1: Recorder Toggle <a name="recorder-toggle"></a>
This button serves as a toggle for recording. The current recording status, indicating whether recording is active or not, is displayed next to the button.
Once the recording begins, two HDF files and a log file are generated at the location specified in [__data directory__](#3-data-directory-).
### 2: Tracker Toggle <a name="tracker-toggle"></a>
This button functions as a toggle for tracking. The tracking method is determined by [__plane interpolation z tracking__](#plane-interpolation-z-tracking),
[__tracking's Z offset__](#tracking-s-z-offset), and [__tracking mode__](#tracking-mode).
The current tracking status, indicating whether tracking is active or not, is displayed next to the button.
### 3: Data Directory
Here, you can specify the location where the log file and data are saved. It is recommended to create separate folders for each recording session.
### 4: Frame Per Second <a name="frame-per-second"></a>
You can set the imaging frame rate here. The minimum and maximum allowed values are indicated in parentheses. For most behavioral experiments, a frame rate of 20Hz is sufficient.
Keep in mind that higher frame rates will increase the size of the recorded data, and certain tracking methods may require more computational resources,
potentially leading to tracking issues.
### 5: GFP Camera Exposure Time <a name="gfp-camera-exposure-time"></a>
This parameter determines the exposure time for the camera recording the behavior (displayer "14"). The minimum and maximum values depend on the current frame rate "4".
Setting a high exposure time can lead to the loss of features due to saturation.
### 6: Behavior Camera Exposure Time <a name="behavior-camera-exposure-time"></a>
This parameter controls the exposure time for the camera recording the GFP signal (displayer "16"). The minimum and maximum values depend on the current frame rate "4".
It is important to avoid increasing this parameter excessively to prevent camera saturation, which can compromise the extraction of meaningful GCaMP signals.
### 7: IR LED Toggle <a name="ir-led-toggle"></a>
This button serves as a toggle for the IR LED, which acts as the light source for behavior recording.
The LED can only be turned on or off; therefore, the intensity is fixed at 100 and cannot be adjusted.
### 8: 470nm LED <a name="470nm-led"></a>
This button functions as a toggle for the 470nm LED, serving as the light source for the green fluorescent proteins. The LED can be switched on or off,
and the intensity can be adjusted from 0 to 100 percent. Setting the intensity to 100 percent corresponds to passing the maximum allowed current to the LED.
### 9: 595nm LED <a name="595nm-led"></a>
This button functions as a toggle for the 595nm LED, serving as the light source for specific optogenetic experiments. The LED can be switched on or off,
and the intensity can be adjusted from 0 to 100 percent. Setting the intensity to 100 percent corresponds to passing the maximum allowed current to the LED.
### 10: Motion Control <a name="motion-control"></a>
These six buttons enable you to move the stage in the X, Y, and Z directions. The velocity for the X and Y directions is the same but can be adjusted
independently from the velocity of the Z direction. Pressing and holding a button initiates motion in the desired direction, and releasing the button stops the stage's movement.
### 11: Plane Interpolation Z Tracking <a name="plane-interpolation-z-tracking"></a>
This mode is designed for precise Z tracking. To utilize this mode, the sample needs to be prepared on the surface of the glass (refer to the sample preparation protocol).
Three points on the glass surface should be marked (using a sharpie, for example). If the user chooses to use this tracking mode, they should manually adjust the motor
in the X, Y, and Z directions to focus on each marked point and then press the 'Set Point' button. These three points define the plane of the glass surface and will be
used to maintain the Z position consistently on this plane.
### 12: Tracking's Z Offset <a name="tracking-s-z-offset"></a>
Regardless of the algorithm utilized for Z tracking, there's a chance that the neurons or fluorescent elements in the sample might be situated at a different Z position
than the target being tracked via the behavior channel. This parameter offers the flexibility to designate an offset, enabling focus on the elements of interest within the GFP
channel, even if they're placed at a Z position different from the predicted Z (the result given by the Z tracking algorithm).
### 13: Tracking Mode <a name="tracking-mode"></a>
There are various tracking algorithms available for capturing motion in the x-y plane. The appropriate method to be used depends on factors such as the magnification of the
objective, imaging conditions (please refer to the sample preparation protocol), and the developmental stage of the animal (L1, L2, L3, L4, YA). In this section,
you can specify the tracking method to be utilized based on the requirements of your experiment.
### 14: Behavior Displayer <a name="behavior-displayer"></a>
This displays the frames overlaid from the behavior channel and the GFP channel. Before overlaying, a background subtraction is applied to the frame from the GFP channel.
To learn more about how to adjust the level of background subtraction, refer to the parameters in the configuration file.The white circle represents the output of the tracking
algorithm. It is advised to initiate tracking only if the output aligns with your expectations. 
### 15: Behavior Camera Cropping Offset <a name="behavior-camera-cropping-offset"></a>
The displayed images are cropped from the full field of view. These parameters determine the deviation from the center of the full field of view, ensuring alignment between
the images from the two channels. After finding the optimal x and y offset pair, you can enter them in the configuration file. Depending on your system, these parameters may
require periodic adjustments to maintain accurate alignment.
### 16: GFP Dispalyer <a name="gfp-dispalyer"></a>
This displays the frame from the GFP channel. The presence of high levels of background noise can be attributed to two main factors: poor sample preparation and/or high LED power.
### 17: GFP Camera Cropping Offset <a name="gfp-camera-cropping-offset"></a>
The displayed images are cropped from the full field of view, and these parameters control the deviation from the center of the full field of view. Their purpose is to achieve
alignment between the images from the two channels, as described in "15". Depending on the sensor size, it may be necessary to adjust the offset in the GFP camera instead
of changing it in the behavior camera in the opposite direction for one particular axis.
### 18: Quit <a name="quit"></a>
This closes the software and, ensures that all open files are properly closed before shutting down the system.
