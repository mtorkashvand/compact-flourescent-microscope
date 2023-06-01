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

import numpy as np
import cv2 as cv

import PySimpleGUI as sg

from cfm.zmq.client_with_gui import GUIClient
from cfm.system.cfm_with_gui import CFMwithGUI
from cfm.devices.dual_displayer import DualDisplayer
from cfm.ui.elements import InputSlider, CombosJoined, InputWithIncrements

# Parameters
DEBUG = False

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

# Methods
## Run CFM with GUI
def run_cfm_with_gui(**kwargs):
    cfm_with_gui = CFMwithGUI(
        name="cfm_with_gui",
        **kwargs
    )
    return cfm_with_gui
## 
def sg_input_port(key, port):
    return sg.Input(
        key=key,
        size=5,
        default_text=str(port)
    )

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


ui_framerate = InputSlider('framerate: ', key='framerate', default_value=20, range=(1, 48), type_caster=int)
ui_exposure_behavior = InputSlider('exposure behavior: ', key='exposure_behavior', default_value=18000, range=(1, 48750), type_caster=int)  # TODO: set range_max from frame_rate and lock it
ui_exposure_gfp = InputSlider('exposure gfp: ', key='exposure_gcamp', default_value=48750, range=(1, 48750), type_caster=int)  # TODO: set range_max from frame_rate and lock it
ui_binsize_format = CombosJoined(
    text1="Binsize: ", text2="Format: ",
    v1_to_v2s={
        "1": [
            "UINT8_YX_1024_1024",
            "UINT8_YX_512_512",
            "UINT8_YX_256_256",
            "UINT8_YX_128_128",
        ],
        "2": [
            "UINT8_YX_512_512",
            "UINT8_YX_256_256",
            "UINT8_YX_128_128",
        ],
        "4": [
            "UINT8_YX_256_256",
            "UINT8_YX_128_128",
        ],
    },
    default_v1="2", default_v2="UINT8_YX_512_512",
    key1="binsize", key2="format"
)
ui_offset_behavior_x = InputWithIncrements(
    text = "Offset Behavior X: ",
    key="offset_behavior_x",
    default_value=-10,
    bounds=[-224, 224],
    increments=[-10,-2,2,10],
    type_caster=int
)
ui_offset_behavior_y = InputWithIncrements(
    text = "Offset Behavior Y: ",
    key="offset_behavior_y",
    default_value=-44,
    bounds=[-44, 44],
    increments=[-10,-2,2,10],
    type_caster=int
)
ui_offset_gcamp_x = InputWithIncrements(
    text = "Offset GCaMP X: ",
    key="offset_gcamp_x",
    default_value=-36,
    bounds=[-224, 224],
    increments=[-10,-2,2,10],
    type_caster=int
)
ui_offset_gcamp_y = InputWithIncrements(
    text = "Offset GCaMP Y: ",
    key="offset_gcamp_y",
    default_value=44,
    bounds=[-44, 44],
    increments=[-10,-2,2,10],
    type_caster=int
)
# Add Elements
elements = [
    ui_framerate, ui_exposure_behavior, ui_exposure_gfp, ui_binsize_format,
    ui_offset_behavior_x, ui_offset_behavior_y,
    ui_offset_gcamp_x, ui_offset_gcamp_y,
]


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
progress_bar = sg.ProgressBar(
    max_value=100,
    size=(100,5),
    orientation='horizontal',
    visible=True,
    key='progressbar'
)
layout = [
    # Start & Stop
    [
        [
            sg.Button('Start'),
            sg.Button('Stop'),
            sg.Button('Start Recording', disabled=False),
            sg.Button('Stop Recording', disabled=True),
        ],
    ],
    [
        sg.HorizontalSeparator(),
    ],
    # Camera Lighting
    [
        *ui_framerate.elements, *ui_exposure_behavior.elements, *ui_exposure_gfp.elements
    ],
    # Camera Binning and Size
    [
        *ui_binsize_format.elements
    ],
    [
        sg.Text("teensy_usb_port "), sg_input_port("teensy_usb_port", "COM4"),
        sg.Checkbox(
            text="interpolation_tracking",
            key="interpolation_tracking",
            default=False
        ),
    ],[
        sg.Text("data_directory: "), folder_browser_data, sg.Input(key="data_directory", default_text=r"C:\src\data", size=130),
    ],[
        sg.Text("logger_directory: "), folder_browser_logger, sg.Input(key="logger_directory", default_text=r"C:\src\data", size=130),
    ],
    [
        sg.HorizontalSeparator(),
    ], [  # LED Controls
        sg.Checkbox(
            text="LED IR",
            key="led_ir_checkbox",
            default=False,
            enable_events=True
        ),
        sg.Text("LED GCaMP Power (0-255): "),
        sg.Slider(
            key="led_slider_gcamp",
            range=(0, 255),
            default_value=0,
            resolution=1,
            orientation='h',
            enable_events=True
        ),
        sg.Text("LED Optogenetics Power (0-255): "),
        sg.Slider(
            key="led_slider_optogenetics",
            range=(0, 255),
            default_value=0,
            resolution=1,
            orientation='h',
            enable_events=True
        ),
    ],
    [
        sg.Image(key="img_frame_r", size=(512, 512)),
        sg.Image(key="img_frame_g", size=(512, 512)),
    ],[
        *ui_offset_behavior_x.elements, *ui_offset_behavior_y.elements,
    ],[
        *ui_offset_gcamp_x.elements, *ui_offset_gcamp_y.elements,
    ],[
        sg.HorizontalSeparator(),
    ],
    [
        sg.Button('Ok'),
        sg.Button('Quit'),
        sg.Button('Execute'),
        sg.Input(default_text="DO shutdown", size=50, key="client_cli"),
    ],
    [
        progress_bar
    ],
    
    [
        sg.StatusBar(
            text="Status! :wink:",
            size=(20,1),
            key="statusbar"
        ),
        sg.Sizegrip()
    ]
]

# Register Events
registered_events = defaultdict(list)
for element in elements:
    for event in element.events:
        registered_events[event].append(element)

# Create the window
window = sg.Window(
    'Compact Fluerscence Microscope (CFM) GUI',
    layout
)
window.finalize()
gui_client = GUIClient(port=server_client)

# Create the dual displayer instance
dual_displayer = DualDisplayer(
    window=window,
    data_r=f"L{tracker_out_behavior}",  # displayer port for the behavior data
    data_g=f"L{tracker_out_gcamp}",  # displayer port for the gcamp data
    fmt="UINT8_YX_512_512",  # image format accroding to 'array_props_from_string'
    name="displayer"  # image displayers name start with 'displayer' 
    )

def zero_displayers():
    _tmp = np.zeros((512,512),dtype=np.uint8)
    _tmp = cv.imencode('.png', _tmp)[1].tobytes()
    window['img_frame_r'].update(data=_tmp)
    window['img_frame_g'].update(data=_tmp)
    window.refresh()
zero_displayers()
# Display and interact with the Window using an Event Loop
# DEBUG TODO:  change code to have different `elements`` for each functionality, e.g. like CombosJoined
N = 0
_n, _duration = 0, 0.0
CAMERA_RUNNING = False
while True:
    if CAMERA_RUNNING:
        event, values = window.read(timeout=10)
        img_r, img_g, img_combined = dual_displayer.get_frame(combine=True)
        frame_r = cv.imencode('.png', img_combined)[1].tobytes()
        frame_g = cv.imencode('.png', img_g)[1].tobytes()
        window['img_frame_r'].update(data=frame_r)
        window['img_frame_g'].update(data=frame_g)
    else:
        event, values = window.read()
    if DEBUG:
        print(values)
        print(event)
        _start = time.time()
    # Handle Events
    for element in registered_events[event]:
        element.handle(event = event, **values)
    # Add Values
    # Add values from UI element with expected keys in CFMwithGUI
    for element in elements:
        element.add_values(values)
    if DEBUG:
        _end = time.time()
        _n += 1
        _duration += (_end - _start)
        print(f"DEBUG: Cycles: {_n}, Average Process Time: {_duration/_n}")
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        zero_displayers()
        CAMERA_RUNNING = False
        gui_client.running = False
        break
    elif event == 'Start':
        # Run CLI Client
        CAMERA_RUNNING = True
        gui_client.running = True
        # Run CFM
        if 'cfm_with_gui' not in locals():
            cfm_with_gui = run_cfm_with_gui(**values)
            cfm_with_gui.run()
        else:
            print('Already running!')
    elif event == 'Stop':
        # Stop CLI Client
        zero_displayers()
        CAMERA_RUNNING = False
        gui_client.process("DO shutdown")
        dual_displayer.reset_buffers()
        time.sleep(0.5)
        gui_client.running = False
        # Stop CFM
        if 'cfm_with_gui' in locals():
            cfm_with_gui.kill()
            if DEBUG:
                print("DEBUG cfm killed")
            del cfm_with_gui
            print("Process should be killed.")
        else:
            print('Not running!')
    elif event == 'Execute':
        client_cli_cmd = values['client_cli']
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event.startswith("offset_behavior"):
        print("DEBUG I'M BHEAVIOR!!!")
        offset_bx = 224 + values['offset_behavior_x']
        offset_by = 44 + values['offset_behavior_y']
        client_cli_cmd = "DO _flir_camera_set_region_behavior 1 512 512 2 {} {}".format(
            offset_by, offset_bx
        )
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event.startswith("offset_gcamp"):
        print("DEBUG I'M GCAMP!!!")
        offset_gx = 224 + values['offset_gcamp_x']
        offset_gy = 44 + values['offset_gcamp_y']
        client_cli_cmd = "DO _flir_camera_set_region_gcamp 1 512 512 2 {} {}".format(
            offset_gy, offset_gx
        )
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event.startswith("exposure_behavior"):
        client_cli_cmd = "DO _flir_camera_set_exposure_framerate_behavior {} {}".format(
            values["exposure_behavior"], values["framerate"] 
        )
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event.startswith("exposure_gcamp"):
        client_cli_cmd = "DO _flir_camera_set_exposure_framerate_gcamp {} {}".format(
            values["exposure_gcamp"], values["framerate"] 
        )
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event == 'Start Recording':
        for element in elements:
            element.disable()
        window['Start Recording'].update(disabled=True)
        window['Stop Recording'].update(disabled=False)
        client_cli_cmd = f"DO _writer_start"
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event == 'Stop Recording':
        for element in elements:
            element.enable()
        window['Start Recording'].update(disabled=False)
        window['Stop Recording'].update(disabled=True)
        client_cli_cmd = f"DO _writer_stop"
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event == 'led_ir_checkbox':
        # TODO set LED Optogenetics power
        state_str = "n" if values['led_ir_checkbox'] else "f"
        client_cli_cmd = f"DO _teensy_commands_set_toggle_led {state_str}"
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event.startswith("led_slider_gcamp"):
        # TODO set LED GCaMP power
        led_name, intensity = "g", int(values['led_slider_optogenetics'])
        client_cli_cmd = f"DO _teensy_commands_set_led {led_name} {intensity}"
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    elif event.startswith("led_slider_optogenetics"):
        # TODO set LED Optogenetics power
        led_name, intensity = "o", int(values['led_slider_optogenetics'])
        client_cli_cmd = f"DO _teensy_commands_set_led {led_name} {intensity}"
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    
    # Output a message to the window
    N = min(N+1, 100)
    window.find_element(key="progressbar").update(
        current_count = N
    )

# Finish up by removing from the screen
window.close()

# Elements to add:
# - remove exposure slider
# - z-axis speed -> L1,R1 on GamePad
# - movement in X/Y/Z directions (fast/slow)


if __name__ == '__main__':
    print('It Worked!')