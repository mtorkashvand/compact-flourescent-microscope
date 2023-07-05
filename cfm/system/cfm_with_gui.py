#! python
#
# This runs wormtracker.
#
# Copyright 2022
# Author: Sina Rasouli, Mahdi Torkashvand (mmt.mahdi@gmail.com)
# Used code developed by "Mahdi Torkashvand" @ https://github.com/mtorkashvand/WormTracker-NSnB

"""
Different Camera Configuration Serial Numbers:
- 17476171, 174757448
- 22591142, 22591117
"""

DEBUG = False

import os
from subprocess import Popen

from serial.tools import list_ports
from cfm.devices.utils import array_props_from_string

def get_teensy_port():
    """
    This returns the port number for the Teensy board assuming there is one 
    (and only one) Teensy board connected to a COM port.

    For the case with multiple Teensy boards, you can get a board if VENDOR_ID, PRODUCT_ID
    and SERIAL_NUMBER are available:

    for port in list(list_ports.comports()):
        if port[1].startswith("Teensy") and if port[2] == "USB VID:PID=%s:%s SNR=%s"%(VENDOR_ID, PRODUCT_ID, SERIAL_NUMBER):
	        return port[0]
    """
    # DEBUG TODO use the following linkes to change this code
    # https://pyserial.readthedocs.io/en/latest/tools.html#serial.tools.list_ports.ListPortInfo
    # https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
    return "COM4"
    for port in list(list_ports.comports()):
        if port[1].startswith("Teensy"):
            return port[0]
    return

class CFMwithGUI:
    DEFAUL_KWARGS = dict(
        format = "UINT8_YX_512_512",
        camera_serial_number_behavior = "22591117",
        camera_serial_number_gcamp = "22591142",
        binsize = 2,
        exposure_behavior = 1,
        exposure_gcamp = 1,
        interpolation_tracking = False,
        forwarder_in = 5000,
        forwarder_out = 5001,
        server_client = 5002,
        processor_out = 6001,
        data_camera_out_behavior = 5003,
        data_stamped_behavior = 5004,
        tracker_out_behavior = 5005,
        data_camera_out_gcamp = 5006,
        data_stamped_gcamp = 5007,
        tracker_out_gcamp = 5008,
        tracker_out_debug  = 5009,
        tracker_out_image = 5010,
        framerate = 20,
        data_directory = r"C:\src\data",
        logger_directory = r"C:\src\data",
    )
    # Constructor
    def __init__(self, name, **kwargs) -> None:
        self.kwargs = CFMwithGUI.DEFAUL_KWARGS.copy()
        for key,value in kwargs.items():
            self.kwargs[key] = value
        self.jobs = []
        self.name = name
        return
    
    # Kill
    def kill(self):
        for job in self.jobs:
            try:
                job.kill()
            except PermissionError as _e:
                print("Received error closing process: ", _e)
        self.jobs = []
        return
    
    # Run
    def run(self):
        """This runs all devices."""

        camera_serial_number_behavior = self.kwargs['camera_serial_number_behavior']
        camera_serial_number_gcamp = self.kwargs['camera_serial_number_gcamp']
        binsize = self.kwargs['binsize']
        exposure_behavior = self.kwargs['exposure_behavior']
        exposure_gcamp = self.kwargs['exposure_gcamp']
        interpolation_tracking =  self.kwargs['interpolation_tracking']
        forwarder_in = self.kwargs['forwarder_in']
        forwarder_out = self.kwargs['forwarder_out']
        server_client = self.kwargs['server_client']
        processor_out = self.kwargs['processor_out']
        data_camera_out_behavior = self.kwargs['data_camera_out_behavior']
        data_stamped_behavior = self.kwargs['data_stamped_behavior']
        tracker_out_behavior = self.kwargs['tracker_out_behavior']
        data_camera_out_gcamp = self.kwargs['data_camera_out_gcamp']
        data_stamped_gcamp = self.kwargs['data_stamped_gcamp']
        tracker_out_gcamp = self.kwargs['tracker_out_gcamp']
        tracker_out_debug  = self.kwargs['tracker_out_debug']
        tracker_out_image = self.kwargs['tracker_out_image']
        data_directory = self.kwargs['data_directory']  # "C:\src\data"
        logger_directory = self.kwargs['logger_directory']  # "C:\src\data"
        framerate = self.kwargs['framerate']
        format = self.kwargs['format']
        teensy_usb_port = get_teensy_port()
        (_, _, shape) = array_props_from_string(format)

        self.jobs.append(Popen(["cfm_hub",
                        f"--server={server_client}",
                        f"--inbound=L{forwarder_out}",
                        f"--outbound=L{forwarder_in}",
                        f"--name=hub",
                        f"--framerate={framerate}"]))

        self.jobs.append(Popen(["cfm_forwarder",
                        f"--inbound={forwarder_in}",
                        f"--outbound={forwarder_out}"]))
            
        if not DEBUG:

            if not os.path.exists(data_directory):
                os.makedirs(data_directory)
            if not os.path.exists(logger_directory):
                os.makedirs(logger_directory)

            self.jobs.append(Popen(["cfm_processor",
                            f"--name=controller",
                            f"--inbound=L{forwarder_out}",
                            f"--outbound={processor_out}",
                            f"--deadzone=5000",
                            f"--threshold=50"]))

            self.jobs.append(Popen(["cfm_commands",
                            f"--inbound=L{processor_out}",
                            f"--outbound=L{forwarder_in}",
                            f"--commands=L{forwarder_out}"]))
            # Behavior Camera
            self.jobs.append(Popen(["flir_camera",
                            f"--serial_number={camera_serial_number_behavior}",
                            f"--commands=L{forwarder_out}",
                            f"--name=FlirCameraBehavior",
                            f"--status=L{forwarder_in}",
                            f"--data=*:{data_camera_out_behavior}",
                            f"--width={shape[1]}",
                            f"--height={shape[0]}",
                            f"--binsize={binsize}",
                            f"--exposure_time={exposure_behavior}",
                            f"--frame_rate={framerate}",]))
            ## GCaMP Camera
            self.jobs.append(Popen(["flir_camera",
                            f"--serial_number={camera_serial_number_gcamp}",
                            f"--commands=L{forwarder_out}",
                            f"--name=FlirCameraGCaMP",
                            f"--status=L{forwarder_in}",
                            f"--data=*:{data_camera_out_gcamp}",
                            f"--width={shape[1]}",
                            f"--height={shape[0]}",
                            f"--binsize={binsize}",
                            f"--exposure_time={exposure_gcamp}",
                            f"--frame_rate={framerate}"]))
            ## Behavior Data Hub
            self.jobs.append(Popen(["cfm_data_hub",
                            f"--data_in=L{data_camera_out_behavior}",
                            f"--commands_in=L{forwarder_out}",
                            f"--status_out=L{forwarder_in}",
                            f"--data_out={data_stamped_behavior}",
                            f"--format={format}",
                            f"--name=data_hub_behavior"]))
            ## GCaMP Data Hub
            self.jobs.append(Popen(["cfm_data_hub",
                            f"--data_in=L{data_camera_out_gcamp}",
                            f"--commands_in=L{forwarder_out}",
                            f"--status_out=L{forwarder_in}",
                            f"--data_out={data_stamped_gcamp}",
                            f"--format={format}",
                            f"--name=data_hub_gcamp",
                            "--flip_image"]))  # TODO: convert to an argument coming from GUI
            ## Behavior Data Writer
            self.jobs.append(Popen(["cfm_writer",
                            f"--data_in=L{data_stamped_behavior}",
                            f"--commands_in=L{forwarder_out}",
                            f"--status_out=L{forwarder_in}",
                            f"--format={format}",
                            f"--directory={data_directory}",
                            f"--video_name=flircamera_behavior",
                            f"--name=writer_behavior"]))
            ## GCaMP Data Writer
            self.jobs.append(Popen(["cfm_writer",
                            f"--data_in=L{data_stamped_gcamp}",
                            f"--commands_in=L{forwarder_out}",
                            f"--status_out=L{forwarder_in}",
                            f"--format={format}",
                            f"--directory={data_directory}",
                            f"--video_name=flircamera_gcamp",
                            f"--name=writer_gcamp"]))
            ## Debug Display
            self.jobs.append(Popen(["cfm_displayer",
                            f"--inbound=L{tracker_out_debug}",
                            f"--format={format}",
                            f"--commands=L{forwarder_out}",
                            f"--name=displayer_debug"]))
            # Logger
            self.jobs.append(Popen(["cfm_logger",
                            f"--inbound={forwarder_out}",
                            f"--directory={logger_directory}"]))
            # TODO: add selection between only one tracker being activated
            ## Behavior Tracker
            self.jobs.append(Popen(["cfm_tracker",
                            f"--commands_in=L{forwarder_out}",
                            f"--commands_out=L{forwarder_in}",
                            f"--data_in=L{data_stamped_behavior}",
                            f"--data_out={tracker_out_behavior}",
                            f"--format={format}",
                            f"--interpolation_tracking={interpolation_tracking}",
                            f"--name=tracker_behavior",
                            f"--data_out_debug={tracker_out_debug}",
                            f"--tracker_out_image={tracker_out_image}",
                        ]))
            ## GCaMP  Tracker
            self.jobs.append(Popen(["cfm_tracker",
                            f"--commands_in=L{forwarder_out}",
                            f"--commands_out=L{forwarder_in}",
                            f"--data_in=L{data_stamped_gcamp}",
                            f"--data_out={tracker_out_gcamp}",
                            f"--format={format}",
                            f"--interpolation_tracking={interpolation_tracking}",
                            f"--name=tracker_gcamp",
                            f"--data_out_debug={tracker_out_debug}",
                            f"--tracker_out_image={tracker_out_image+2000}",
                        ]))
            ## Detector Pharynx
            # self.jobs.append(Popen(["detector_pharynx",
            #                 f"--input_image=L{tracker_out_image}",
            #                 f"--to_forwarder=L{forwarder_in}",
            #                 f"--input_commands=L{forwarder_out}",
            #                 ]))

            self.jobs.append(Popen(["cfm_teensy_commands",
                            f"--inbound=L{forwarder_out}",
                            f"--outbound=L{forwarder_in}",
                            f"--port={teensy_usb_port}"]))

        return