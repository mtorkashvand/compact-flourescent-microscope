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
import signal
from subprocess import Popen

from collections import defaultdict

import PySimpleGUI as sg

from cfm.zmq.client_with_gui import GUIClient
from cfm.system.cfm_with_gui import run as cfm_with_gui_run

from cfm.ui.elements import InputSlider
## TODO DEBUG CONTINUE FROM HERE: run `cfm_with_gui_run` from here with KWARGS

# Parameters
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
    cmd = ["cfm_with_gui"]
    for key, value in kwargs.items():
        cmd += [ f"--{key}={value}" ]
    return Popen(cmd)
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


ui_framerate = InputSlider('framerate: ', key='--FRAMERATE--', default_value=20, range=(1, 48), type_caster=int)
ui_exposure_behavior = InputSlider('exposure behavior: ', key='--EXPOSURE-BEHAVIOR--', default_value=18000, range=(1, 48750), type_caster=int)  # TODO: set range_max from frame_rate and lock it
ui_exposure_gfp = InputSlider('exposure gfp: ', key='--EXPOSURE-GFP--', default_value=48750, range=(1, 48750), type_caster=int)  # TODO: set range_max from frame_rate and lock it
# Add Elements
elements = [
    ui_framerate, ui_exposure_behavior, ui_exposure_gfp,
]


progress_bar = sg.ProgressBar(
    max_value=100,
    size=(100,5),
    orientation='horizontal',
    visible=True,
    key='progressbar'
)
layout = [
    [
        [
            sg.Button('Start'),
            sg.Button('Stop'),
            sg.Button('Execute'),
            sg.Input(default_text="DO shutdown", size=50, key="client_cli"),
        ],
    ],
    [
        sg.HorizontalSeparator(),
    ],
    [
        sg.HorizontalSeparator(),
    ],
    [
        sg.HorizontalSeparator(),
    ],
    [
        sg.HorizontalSeparator(),
    ],
    [
        sg.HorizontalSeparator(),
    ],
    [
        *ui_framerate.elements, *ui_exposure_behavior.elements, *ui_exposure_gfp.elements
    ],
    [
        # sg.Text("framerate "),
        # sg.Input(
        #     default_text="20.0", size=6, key="framerate_input",
        #     enable_events=True
        # ),
        # sg.Slider(
        #     range=(0.0, 48.0),
        #     default_value=20.0,
        #     resolution=1,
        #     orientation='h',
        #     key="framerate_slider"
        # ),
        # sg.Text("binsize "), sg_input_port("binsize", 2),
        # sg.Text("flir_exposure_behavior "), sg_input_port("flir_exposure_behavior", 4500.0),
        # sg.Text("flir_exposure_gcamp "), sg_input_port("flir_exposure_gcamp", 975000/20),
        # sg.Text("fmt "), sg.Listbox(
        #     values=[
        #         "UINT8_YX_512_512",
        #         "UINT8_YX_256_256",
        #         "UINT8_YX_128_128",
        #     ],
        #     select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
        #     size=(20,3),
        #     key="fmt"
        # ),
        sg.Text("format "), sg.Combo(
            values=[
                "UINT8_YX_512_512",
                "UINT8_YX_256_256",
                "UINT8_YX_128_128",
            ],
            default_value="UINT8_YX_512_512",
            size=(20,3),
            key="format"
        ),
        sg.Text("teensy_usb_port "), sg_input_port("teensy_usb_port", "COM4"),
        sg.Checkbox(
            text="interpolation_tracking",
            key="interpolation_tracking",
            default=False
        ),
        sg.Text("data_directory "), sg_input_port("data_directory", "C:\src\data"),
        sg.Text("logger_directory "), sg_input_port("logger_directory", "C:\src\data"),
    ],
    [
        sg.HorizontalSeparator(),
    ],
    # [
    #     sg.Text("Forwarder Ports: "),
    #     sg.Text("forwarder_in "), sg_input_port("forwarder_in", 5000),
    #     sg.Text("forwarder_out "), sg_input_port("forwarder_out", 5001),
    # ],
    # [
    #     sg.HorizontalSeparator(),
    # ],
    # [
    #     sg.Text("Server Client Ports: "),
    #     sg_input_port("server_client", 5002),
    # ],
    # [
    #     sg.HorizontalSeparator(),
    # ],
    # [
    #     sg.Text("Behavior Camera: "),
    #     sg.Text("data_camera_out_behavior "), sg_input_port("data_camera_out_behavior", 5003),
    #     sg.Text("data_stamped_behavior "), sg_input_port("data_stamped_behavior", 5004),
    #     sg.Text("tracker_out_behavior "), sg_input_port("tracker_out_behavior", 5005),
    #     sg.Text("tracker_out_debug "), sg_input_port("tracker_out_debug", 5009),
    #     sg.Text("camera_serial_number_behavior "), sg.Input(
    #         key="camera_serial_number_behavior",
    #         size=8,
    #         default_text="22591117"
    #     ),
    # ],
    # [
    #     sg.HorizontalSeparator(),
    # ],
    # [
    #     sg.Text("GCaMP Camera: "),
    #     sg.Text("data_camera_out_gcamp "), sg_input_port("data_camera_out_gcamp", 5006),
    #     sg.Text("data_stamped_gcamp "), sg_input_port("data_stamped_gcamp", 5007),
    #     sg.Text("tracker_out_gcamp "), sg_input_port("tracker_out_gcamp", 5008),
    #     sg.Text("camera_serial_number_gcamp "), sg.Input(
    #         key="camera_serial_number_gcamp",
    #         size=8,
    #         default_text="22591142"
    #     ),
    # ],
    # [
    #     sg.HorizontalSeparator(),
    # ],
    # [
    #     sg.Text("General: "),
    #     sg.Text("XInputToZMQPub_out "), sg_input_port("XInputToZMQPub_out", 6000),
    #     sg.Text("processor_out "), sg_input_port("processor_out", 6001),
    # ],
    [
        sg.HorizontalSeparator(),
    ],
    [sg.Button('Ok'), sg.Button('Quit')],
    # [
    #     sg.Radio(
    #         text="Option1",
    #         group_id="group_1",
    #         default=True,
    #         key="group_1_option1"
    #     ),
    #     sg.Radio(
    #         text="Option2",
    #         group_id="group_1",
    #         default=False,
    #         key="group_2_option1"
    #     ),
    #     sg.Radio(
    #         text="Option3",
    #         group_id="group_1",
    #         default=False,
    #         key="group_3_option1"
    #     ),
    # ],
    # [
    #     sg.Output(
    #         size=(150,10),
    #         background_color="#000000",
    #         text_color="#ffffff",
    #         echo_stdout_stderr=True,
    #         key="output"
    #     ),
    # ],
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
gui_client = GUIClient(port=server_client)



# Display and interact with the Window using an Event Loop
N = 0
while True:
    event, values = window.read()
    print(values)
    print(event)
    for element in registered_events[event]:
        element.handle(event = event, **values)
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        gui_client.running = False
        break
    elif event == 'Start':
        # Run CLI Client
        gui_client.running = True
        # Run CFM
        if 'cfm_job' not in locals():
            cfm_job = run_cfm_with_gui()
        else:
            print('Already running!')
    elif event == 'Stop':
        # Stop CLI Client
        gui_client.process("DO shutdown")
        time.sleep(3)
        gui_client.running = False
        # Stop CFM
        if 'cfm_job' in locals():
            cfm_job.send_signal(signal.SIGINT)
            # cfm_job.kill()
            del cfm_job
            print("Process should be killed.")
        else:
            print('Not running!')
    elif event == 'Execute':
        client_cli_cmd = values['client_cli']
        print(f"Executing: '{client_cli_cmd}'")
        gui_client.process(client_cli_cmd)
    # Output a message to the window
    print('\n'*3)
    N = min(N+1, 100)
    window.find_element(key="progressbar").update(
        current_count = N
    )

# Finish up by removing from the screen
window.close()



if __name__ == '__main__':
    print('It Worked!')