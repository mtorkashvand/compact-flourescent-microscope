# TODO: add SHABANG! :D
# TODO: add docstring, copy-right, ...
# TODO: functionalities to add
# 1. interface to start and stop the microscope
# 2. set initial paramters of the microscope and then start
# 3. buttons and gauges for controlling LEDs
# 4. button for starting and stopping the recoding
# 5. setting 3 different points for "Plane-Tracking"
# 6. other features? brainstorm
# TODO: add properties like below
# - disallow changing paramters when microscope is running
# - check recording, changing paramters, ... disallow doing things together
# - add status indicator, e.g. like a colored region in LAMBDA
# 
# PySimpleGUI Docs
# https://www.pysimplegui.org/en/latest/call%20reference/#the-elements

# Modules
import time
from collections import defaultdict

import cv2 as cv
import numpy as np
import PySimpleGUI as sg

from cfm.zmq.client_with_gui import GUIClient
from cfm.system.cfm_with_gui import CFMwithGUI
from cfm.devices.utils import array_props_from_string
from cfm.devices.dual_displayer import DualDisplayer
from cfm.ui.elements import (
    InputWithIncrements, ReturnHandler,
    LEDCompound, ExposureCompound, FramerateCompound, LEDIR,
    ZInterpolationTracking, ToggleRecording, ToggleTracking
)

from cfm.icons.icons import (
    ICON_LED_GFP_OFF, ICON_LED_GFP_ON, ICON_LED_OPT_OFF, ICON_LED_OPT_ON,
    ICON_LED_IR_ON, ICON_LED_IR_OFF, ICON_EXPOSURE, ICON_FPS,
    ICON_RECORDING_OFF, ICON_RECORDING_ON,
    ICON_TRACKING_OFF, ICON_TRACKING_ON
)

# Parameters
DEBUG = True
ICON_SIZE = (64, 64)
# These numbers are subject to change based on the camera model and its orientation when mounted.
CAMERA_X_MAX = 1920
CAMERA_Y_MAX = 1200
default_b_x_offset = -10
default_b_y_offset = -44
default_g_x_offset = -36
default_g_y_offset = 44
offset_step_small = 2
offset_step_large = 10
fmt = "UINT8_YX_512_512"
(_, _, shape) = array_props_from_string(fmt)
forwarder_in = str(5000)
forwarder_out = str(5001)
server_client = str(5002)

data_camera_out_behavior = str(5003)
data_stamped_behavior = str(5004)
tracker_out_behavior = str(5005)

data_camera_out_gcamp = str(5006)
data_stamped_gcamp = str(5007)
tracker_out_gcamp = str(5008)
tracker_out_debug  = str(5009)

XInputToZMQPub_out = str(6000)
processor_out = str(6001)

binsize = 2
exposure_behavior = 1
exposure_gcamp = 1
interpolation_tracking = False

y_bound = int((CAMERA_Y_MAX - binsize * shape[0]) / (2 * binsize))
x_bound = int((CAMERA_X_MAX - binsize * shape[1]) / (2 * binsize))

# Methods
## Run CFM with GUI
def run_cfm_with_gui(**kwargs):
    cfm_with_gui = CFMwithGUI(
        name="cfm_with_gui",
        **kwargs
    )
    return cfm_with_gui
##

# Classes

# Main
# Define the window's contents
# layout = [
#     [sg.Text("What's your name?")],
#     [sg.Input(key='-INPUT-')],
#     [sg.Text(size=(40,1), key='-OUTPUT-')],
#     [sg.Button('Ok'), sg.Button('Quit')]
# ]
# TODO: check if using `Column Element` will help organizing
# TODO: use `Image Element` with `update` to show frames?
# TODO: use `Menu Element` to add menu bars on top of the window
# TODO:

elements = []

ui_return_handler = ReturnHandler()
elements.append(ui_return_handler)

ui_recording_toggle = ToggleRecording(
    key="recording",
    icon_off=ICON_RECORDING_OFF,
    icon_on=ICON_RECORDING_ON,
    icon_size=ICON_SIZE
)
elements.append(ui_recording_toggle)

ui_tracking_toggle = ToggleTracking(
    key="tracking",
    icon_off=ICON_TRACKING_OFF,
    icon_on=ICON_TRACKING_ON,
    icon_size=ICON_SIZE
)
elements.append(ui_tracking_toggle)

ui_offset_behavior_x = InputWithIncrements(
    text = "Offset Behavior X: ",
    key="offset_behavior_x",
    default_value=default_b_x_offset,
    bounds=[-x_bound, x_bound],
    increments=[-offset_step_large,
                -offset_step_small,
                offset_step_small,
                offset_step_large],
    type_caster=int
)
elements.append(ui_offset_behavior_x)

ui_offset_behavior_y = InputWithIncrements(
    text = "Offset Behavior Y: ",
    key="offset_behavior_y",
    default_value=default_b_y_offset,
    bounds=[-y_bound, y_bound],
    increments=[-offset_step_large,
                -offset_step_small,
                offset_step_small,
                offset_step_large],
    type_caster=int
)
elements.append(ui_offset_behavior_y)

ui_offset_gcamp_x = InputWithIncrements(
    text = "Offset GCaMP X: ",
    key="offset_gcamp_x",
    default_value=default_g_x_offset,
    bounds=[-x_bound, x_bound],
    increments=[-offset_step_large,
                -offset_step_small,
                offset_step_small,
                offset_step_large],
    type_caster=int
)
elements.append(ui_offset_gcamp_x)

ui_offset_gcamp_y = InputWithIncrements(
    text = "Offset GCaMP Y: ",
    key="offset_gcamp_y",
    default_value=default_g_y_offset,
    bounds=[-y_bound, y_bound],
    increments=[-offset_step_large,
                -offset_step_small,
                offset_step_small,
                offset_step_large],
    type_caster=int
)
elements.append(ui_offset_behavior_y)

ui_led_gfp = LEDCompound(
    text="470nm LED power (%)",
    key='led_gfp',
    led_name='g',
    icon_off=ICON_LED_GFP_OFF,
    icon_on=ICON_LED_GFP_ON,
    icon_size=ICON_SIZE,
    bounds=(0, 100)
)
elements.append(ui_led_gfp)

ui_led_opt = LEDCompound(
    text="595nm LED power (%)",
    key='led_opt',
    led_name='o',
    icon_off=ICON_LED_OPT_OFF,
    icon_on=ICON_LED_OPT_ON,
    icon_size=ICON_SIZE,
    bounds=(0, 100)
)
elements.append(ui_led_opt)

ui_led_ir = LEDIR(
    text="760nm LED power (%)",
    key='led_ir',
    icon_off=ICON_LED_IR_OFF,
    icon_on=ICON_LED_IR_ON,
    icon_size=ICON_SIZE,
)
elements.append(ui_led_ir)

ui_exposure_gfp = ExposureCompound(
    text='Exposure Time',
    key='exposure_gcamp',
    icon=ICON_EXPOSURE,
    icon_size=ICON_SIZE,
    camera_name='gcamp',
    bounds=(250, 478000)
)
elements.append(ui_exposure_gfp)

ui_exposure_behavior = ExposureCompound(
    text='Exposure Time',
    key='exposure_behavior',
    icon=ICON_EXPOSURE,
    icon_size=ICON_SIZE,
    camera_name='behavior',
    bounds=(250, 478000)
)
elements.append(ui_exposure_behavior)

ui_framerate = FramerateCompound(
    ui_exposure_behavior,
    ui_exposure_gfp,
    icon=ICON_FPS,
    icon_size=ICON_SIZE
)
elements.append(ui_framerate)

ui_interpolation_tracking = ZInterpolationTracking()
elements.append(ui_interpolation_tracking)

folder_browser_data = sg.FolderBrowse(
    button_text = "Browse",
    target = "data_directory",
    initial_folder = "."
)
folder_browser_logger = sg.FolderBrowse(
    button_text = "Browse",
    target = "logger_directory",
    initial_folder = "."
)

led_column_layout = sg.Column(
    [[*ui_led_ir.elements], 
     [*ui_led_gfp.elements], 
     [*ui_led_opt.elements]]
)
exp_fps_column_layout = sg.Column(
    [[*ui_framerate.elements],
     [*ui_exposure_behavior.elements],
     [*ui_exposure_gfp.elements]]
)

layout = [
    [
        *ui_return_handler.elements,
    ],[
        *ui_recording_toggle.elements, *ui_tracking_toggle.elements
    ],[
        sg.HorizontalSeparator(),
    ],[
        [exp_fps_column_layout, sg.VSeparator(), led_column_layout, sg.VSeparator()]
    ],[
        sg.HorizontalSeparator(),
    ],[
        *ui_interpolation_tracking.elements,
    ],[
        sg.Text("data_directory: "),
        folder_browser_data, sg.Input(key="data_directory", default_text=r"./", size=30),
        sg.Text("logger_directory: "),
        folder_browser_logger, sg.Input(key="logger_directory", default_text=r"./", size=30),
    ],[
        sg.HorizontalSeparator(),
    ],[
        sg.Image(key="img_frame_r", size=shape),
        sg.Image(key="img_frame_g", size=shape),
    ],[
        *ui_offset_behavior_x.elements, *ui_offset_behavior_y.elements,
    ],[
        *ui_offset_gcamp_x.elements, *ui_offset_gcamp_y.elements,
    ],[
        sg.HorizontalSeparator(),
    ],[
        sg.Button('Quit')
    ]
]

# Register Events
registered_events = defaultdict(list)
for element in elements:
    for event in element.events:
        registered_events[event].append(element)

# Create the window
window = sg.Window(
    'OpenAutoScope2.0 GUI',
    layout,
    finalize=True
)
ui_return_handler.set_window(window)
gui_client = GUIClient(port=server_client, port_forwarder_in=f"L{forwarder_in}")
# Add Client to Elements so they can interact directly with it
for element in elements:
    element.set_client(gui_client)

# Create the dual displayer instance
dual_displayer = DualDisplayer(
    window=window,
    data_r=f"L{tracker_out_behavior}",  # displayer port for the behavior data
    data_g=f"L{tracker_out_gcamp}",  # displayer port for the gcamp data
    fmt=fmt,  # image format accroding to 'array_props_from_string'
    name="displayer"  # image displayers name start with 'displayer' 
    )

def zero_displayers():
    _tmp = np.zeros(shape, dtype=np.uint8)
    _tmp = cv.imencode('.png', _tmp)[1].tobytes()
    window['img_frame_r'].update(data=_tmp)
    window['img_frame_g'].update(data=_tmp)
    window.refresh()
zero_displayers()
# Display and interact with the Window using an Event Loop
# DEBUG TODO:  change code to have different `elements`` for each functionality, e.g. like CombosJoined

event, values = window.read(timeout=0)

offset_bx = x_bound + int(values['offset_behavior_x'])
offset_by = y_bound + int(values['offset_behavior_y'])
offset_gx = x_bound + int(values['offset_gcamp_x'])
offset_gy = y_bound + int(values['offset_gcamp_y'])


cfm_with_gui = run_cfm_with_gui(**values)
cfm_with_gui.run()

client_cli_cmd = "DO _flir_camera_set_region_behavior 1 {} {} {} {} {}".format(
    shape[0], shape[1], binsize, offset_by, offset_bx
)
gui_client.process(client_cli_cmd)

client_cli_cmd = "DO _flir_camera_set_region_gcamp 1 {} {} {} {} {}".format(
    shape[0], shape[1], binsize, offset_gy, offset_gx
)
gui_client.process(client_cli_cmd)

while True:
    event, values = window.read(timeout=10)
    if event == sg.WIN_CLOSED or event == 'Quit':
        gui_client.process("DO shutdown")
        break
        
    # Add Values
    # To be used by handlers.
    for element in elements:
        element.add_values(values)
    # Handle Events
    for element in registered_events[event]:
        element.handle(event = event, **values)
    # Add Values
    # Add values from UI element with expected keys in CFMwithGUI
    for element in elements:
        element.add_values(values)

    img_r, img_g, img_combined = dual_displayer.get_frame(combine=True)
    frame_r = cv.imencode('.png', img_combined)[1].tobytes()
    frame_g = cv.imencode('.png', img_g)[1].tobytes()
    window['img_frame_r'].update(data=frame_r)
    window['img_frame_g'].update(data=frame_g)
# Finish up by removing from the screen
window.close()


# UI Design
# - [/] `input` with autoselect + take effect when return presses
# - menu-tree:
#       \_ set_shape
#       |_ teensy_port
# - [x] leds: toggle + text + value/`input`
# - [x] exposures: icon + text + `input`
# - [x] z-interp: checkbox + text + 3-buttons
# - [x] browse folder: add path to tooltip instead of text box
# - [ ] 4 buttons x-y directions + `input` for speed
# - [ ] 2 buttons for z direction + `input`
# - [ ] start/stop recording same button -> change icon
# - [ ] start/stop tracking same button -> change icon
# - [ ] display radio buttons: overlay or single
# - [ ] vertical offset controllers
# - [ ] crashes after the second stop/start
# - [X] detect the teensy port automatically, and get rid of its element
# - [ ] changing format brings the binsize back to 1 (can't set it to anything other than 1)
# - [ ] bounds for the 'inputwithincrement' should depend on the binsize and the format



if __name__ == '__main__':
    pass