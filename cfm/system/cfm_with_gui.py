#! python
#
# This runs wormtracker.
#
# Copyright 2022
# Author: Sina Rasouli
# Used code developed by "Mahdi Torkashvand" @ https://github.com/mtorkashvand/WormTracker-NSnB

"""
Different Camera Configuration Serial Numbers:
- 17476171, 174757448
- 22591142, 22591117
"""


import os

from subprocess import Popen
from cfm.devices.utils import array_props_from_string


class CFMwithGUI:
    DEFAUL_KWARGS = dict(
        format = "UINT8_YX_512_512",
        camera_serial_number_behavior = "22591117",
        camera_serial_number_gcamp = "22591142",
        binsize = "2",
        exposure_behavior = "1",
        exposure_gcamp = "1",
        interpolation_tracking = "False",
        forwarder_in = "5000",
        forwarder_out = "5001",
        server_client = "5002",
        XInputToZMQPub_out = "6000",
        processor_out = "6001",
        data_camera_out_behavior = "5003",
        data_stamped_behavior = "5004",
        tracker_out_behavior = "5005",
        data_camera_out_gcamp = "5006",
        data_stamped_gcamp = "5007",
        tracker_out_gcamp = "5008",
        tracker_out_debug  = "5009",
        teensy_usb_port = "COM4",
        framerate = "20",
        data_directory = "C:\src\data",
        logger_directory = "C:\src\data",
    )
    # Constructor
    def __init__(self, name, **kwargs) -> None:
        # DEBUG
        print("\n\nDEBUG CFM WITH GUI KWARGS:")
        print(kwargs)
        print("\n\n")
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
    def run( self ):
        """This runs all devices."""

        camera_serial_number_behavior = self.kwargs['camera_serial_number_behavior']
        camera_serial_number_gcamp = self.kwargs['camera_serial_number_gcamp']
        binsize = self.kwargs['binsize']
        exposure_behavior = self.kwargs['exposure_behavior']
        exposure_gcamp = self.kwargs['exposure_gcamp']
        interpolation_tracking =  self.kwargs['interpolation_tracking'].lower() == 'true'

        forwarder_in = self.kwargs['forwarder_in']
        forwarder_out = self.kwargs['forwarder_out']
        server_client = self.kwargs['server_client']
        XInputToZMQPub_out = self.kwargs['XInputToZMQPub_out']
        processor_out = self.kwargs['processor_out']
        data_camera_out_behavior = self.kwargs['data_camera_out_behavior']
        data_stamped_behavior = self.kwargs['data_stamped_behavior']
        tracker_out_behavior = self.kwargs['tracker_out_behavior']
        data_camera_out_gcamp = self.kwargs['data_camera_out_gcamp']
        data_stamped_gcamp = self.kwargs['data_stamped_gcamp']
        tracker_out_gcamp = self.kwargs['tracker_out_gcamp']
        tracker_out_debug  = self.kwargs['tracker_out_debug']

        teensy_usb_port = self.kwargs['teensy_usb_port']  # default: "COM4"
        framerate = self.kwargs['framerate']

        format = self.kwargs['format']
        (_, _, shape) = array_props_from_string(format)
        flir_exposure_behavior = exposure_behavior
        flir_exposure_gcamp = exposure_gcamp
        # exposure_time = 975000.0 / float(framerate) ## Maximum Possible Exposure
        # Maximum acceptable framerate based on exposure
        # framerate_max = 1000000 / (1.02 * exposure_time)
        # flir_exposure_behavior = str(4500)  # micro-seconds
        # flir_exposure_gcamp = str( 975000.0 / float(framerate) )
        framerate = str(20)
        flir_exposure_behavior = str(18000)  # micro-seconds
        flir_exposure_gcamp = str( 975000.0 / float(framerate) )

        data_directory = self.kwargs['data_directory']  # "C:\src\data"
        logger_directory = self.kwargs['logger_directory']  # "C:\src\data"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        if not os.path.exists(logger_directory):
            os.makedirs(logger_directory)

        self.jobs.append(Popen(["cfm_processor",
                        "--name=controller",
                        "--inbound=L" + forwarder_out,
                        "--outbound=" + processor_out,
                        "--deadzone=5000",
                        "--threshold=50"]))

        self.jobs.append(Popen(["cfm_commands",
                        "--inbound=L" + processor_out,
                        "--outbound=L" + forwarder_in,
                        "--commands=L" + forwarder_out]))


        self.jobs.append(Popen(["cfm_hub",
                        "--server=" + server_client,
                        "--inbound=L" + forwarder_out,
                        "--outbound=L" + forwarder_in,
                        "--name=hub",
                        "--framerate="+ framerate]))

        self.jobs.append(Popen(["cfm_forwarder",
                        "--inbound=" + forwarder_in,
                        "--outbound=" + forwarder_out]))

        # Cameras
        # Behavior
        self.jobs.append(Popen(["flir_camera",
                        "--serial_number=" + camera_serial_number_behavior,
                        "--commands=localhost:" + forwarder_out,
                        "--name=FlirCameraBehavior",
                        "--status=localhost:" + forwarder_in,
                        "--data=*:" + data_camera_out_behavior,
                        "--width=" + str(shape[1]),
                        "--height=" + str(shape[0]),
                        "--binsize=" + binsize,
                        "--exposure_time=" + flir_exposure_behavior,
                        "--frame_rate=" + framerate,]))
        ## GCaMP
        self.jobs.append(Popen(["flir_camera",
                        "--serial_number=" + camera_serial_number_gcamp,
                        "--commands=localhost:" + forwarder_out,
                        "--name=FlirCameraGCaMP",
                        "--status=localhost:" + forwarder_in,
                        "--data=*:" + data_camera_out_gcamp,
                        "--width=" + str(shape[1]),
                        "--height=" + str(shape[0]),
                        "--binsize=" + binsize,
                        "--exposure_time=" + flir_exposure_gcamp,
                        "--frame_rate=" + framerate]))
        

        # Data Hubs
        ## Behavior
        self.jobs.append(Popen(["cfm_data_hub",
                            "--data_in=L" + data_camera_out_behavior,
                            "--commands_in=L" + forwarder_out,
                            "--status_out=L" + forwarder_in,
                            "--data_out=" + data_stamped_behavior,
                            "--format=" + format,
                            "--name=data_hub_behavior"]))
        ## GCaMP
        self.jobs.append(Popen(["cfm_data_hub",
                            "--data_in=L" + data_camera_out_gcamp,
                            "--commands_in=L" + forwarder_out,
                            "--status_out=L" + forwarder_in,
                            "--data_out=" + data_stamped_gcamp,
                            "--format=" + format,
                            "--name=data_hub_gcamp"]))

        # Writers
        ## Behavior
        self.jobs.append(Popen(["cfm_writer",
                            "--data_in=L" + data_stamped_behavior,
                            "--commands_in=L" + forwarder_out,
                            "--status_out=L" + forwarder_in,
                            "--format=" + format,
                            "--directory="+ data_directory,
                            "--video_name=flircamera_behavior",
                            "--name=writer_behavior"]))
        ## GCaMP
        self.jobs.append(Popen(["cfm_writer",
                            "--data_in=L" + data_stamped_gcamp,
                            "--commands_in=L" + forwarder_out,
                            "--status_out=L" + forwarder_in,
                            "--format=" + format,
                            "--directory="+ data_directory,
                            "--video_name=flircamera_gcamp",
                            "--name=writer_gcamp"]))

        # Display
        ## Behavior
        self.jobs.append(Popen(["cfm_displayer",
                            "--inbound=L" + tracker_out_behavior,
                            "--format=" + format,
                            "--commands=L" + forwarder_out,
                            "--name=displayer_behavior"]))
        ## GCaMP
        self.jobs.append(Popen(["cfm_displayer",
                            "--inbound=L" + tracker_out_gcamp,
                            "--format=" + format,
                            "--commands=L" + forwarder_out,
                            "--name=displayer_gcamp"]))
        ## Debug Display
        self.jobs.append(Popen(["cfm_displayer",
                            "--inbound=L" + tracker_out_debug,
                            "--format=" + format,
                            "--commands=L" + forwarder_out,
                            "--name=displayer_debug"]))

        # Logger
        self.jobs.append(Popen(["cfm_logger",
                        "--inbound=" + forwarder_out,
                        "--directory=" + logger_directory]))

        # Trackers
        # TODO: add selection between only one tracker being activated
        ## Behavior
        self.jobs.append(Popen(["cfm_tracker",
                        "--commands_in=L" + forwarder_out,
                        "--commands_out=L" + forwarder_in,
                        "--data_in=L" + data_stamped_behavior,
                        "--data_out=" + tracker_out_behavior,
                        "--format=" + format,
                        "--interpolation_tracking=" + str(interpolation_tracking),
                        "--name=tracker_behavior",
                        "--data_out_debug=" + tracker_out_debug]))
        ## GCaMP
        self.jobs.append(Popen(["cfm_tracker",
                        "--commands_in=L" + forwarder_out,
                        "--commands_out=L" + forwarder_in,
                        "--data_in=L" + data_stamped_gcamp,
                        "--data_out=" + tracker_out_gcamp,
                        "--format=" + format,
                        "--interpolation_tracking=" + str(interpolation_tracking),
                        "--name=tracker_gcamp",
                        "--data_out_debug=" + tracker_out_debug]))

        self.jobs.append(Popen(["cfm_teensy_commands",
                        "--inbound=L" + forwarder_out,
                        "--outbound=L" + forwarder_in,
                        "--port=" + teensy_usb_port]))
        # Return
        return