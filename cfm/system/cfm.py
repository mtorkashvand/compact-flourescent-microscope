#! python
#
# This runs wormtracker.
#
# Copyright 2022
# Author: Sina Rasouli
# Used code developed by "Mahdi Torkashvand" @ https://github.com/mtorkashvand/WormTracker-NSnB

"""
Run all wormtracker components.

Usage:
    cfm.py                              [options]

Options:
    -h --help                           Show this help.
    --format=FORMAT_STR                 Image format.
                                            [default: UINT8_YX_512_512]
    --camera_serial_number_behavior=SN  Camera Serial Number.
                                            [default: 22591117]
    --camera_serial_number_gcamp=SN     Camera Serial Number.
                                            [default: 22591142]
    --binsize=NUMBER                    Specify camera bin size.
                                            [default: 2]
    --exposure_behavior=VALUE           Exposure time of flircamera in us.
                                            [default: 1]
    --exposure_gcamp=VALUE              Exposure time of flircamera in us.
                                            [default: 1]
    --interpolation_tracking=BOOL       Uses user-specified points to interpolate z.
                                            [default: False]
"""

"""
Different Camera Configuration Serial Numbers:
- 17476171, 174757448
- 22591142, 22591117
"""

import time
import os
import signal
from subprocess import Popen

from docopt import docopt

from cfm.devices.utils import array_props_from_string

def execute(
        job,
        fmt: str,
        camera_serial_number_behavior: str,
        camera_serial_number_gcamp: str,
        binsize: str,
        exposure_behavior: str,
        exposure_gcamp: str,
        interpolation_tracking: bool,
    ):
    """This runs all devices."""

    forwarder_in = str(5000)
    forwarder_out = str(5001)
    server_client = str(5002)
    XInputToZMQPub_out = str(6000)
    processor_out = str(6001)
    data_camera_out_behavior = str(5003)
    data_stamped_behavior = str(5004)
    tracker_out_behavior = str(5005)
    data_camera_out_gcamp = str(5006)
    data_stamped_gcamp = str(5007)
    tracker_out_gcamp = str(5008)
    tracker_out_debug  = str(5009)

    (_, _, shape) = array_props_from_string(fmt)
    teensy_usb_port = "COM4"
    flir_exposure_behavior = exposure_behavior
    flir_exposure_gcamp = exposure_gcamp
    framerate = str(18)
    # exposure_time = 975000.0 / float(framerate) ## Maximum Possible Exposure
    # Maximum acceptable framerate based on exposure
    # framerate_max = 1000000 / (1.02 * exposure_time)
    flir_exposure_behavior = str(18000)  # micro-seconds
    flir_exposure_gcamp = str( 975000.0 / float(framerate) )
    # flir_exposure_gcamp = str(5000)
    binsize = str(binsize)

    data_directory = "C:\src\data"
    logger_directory = "C:\src\data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    if not os.path.exists(logger_directory):
        os.makedirs(logger_directory)

    job.append(Popen(["cfm_client",
                      "--port=" + server_client]))

    job.append(Popen(["xinput_pub",
                      "--outbound=*:" + XInputToZMQPub_out]))

    job.append(Popen(["cfm_processor",
                      "--inbound=L" + XInputToZMQPub_out,
                      "--outbound=" + processor_out,
                      "--deadzone=5000",
                      "--threshold=50"]))

    job.append(Popen(["cfm_commands",
                      "--inbound=L" + processor_out,
                      "--outbound=L" + forwarder_in]))


    job.append(Popen(["cfm_hub",
                       "--server=" + server_client,
                       "--inbound=L" + forwarder_out,
                       "--outbound=L" + forwarder_in,
                       "--name=hub",
                       "--framerate="+ framerate]))

    job.append(Popen(["cfm_forwarder",
                      "--inbound=" + forwarder_in,
                      "--outbound=" + forwarder_out]))

    # Cameras
    # Behavior
    job.append(Popen(["flir_camera",
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
    job.append(Popen(["flir_camera",
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
    job.append(Popen(["cfm_data_hub",
                        "--data_in=L" + data_camera_out_behavior,
                        "--commands_in=L" + forwarder_out,
                        "--status_out=L" + forwarder_in,
                        "--data_out=" + data_stamped_behavior,
                        "--format=" + fmt,
                        "--name=data_hub_behavior"]))
    ## GCaMP
    job.append(Popen(["cfm_data_hub",
                        "--data_in=L" + data_camera_out_gcamp,
                        "--commands_in=L" + forwarder_out,
                        "--status_out=L" + forwarder_in,
                        "--data_out=" + data_stamped_gcamp,
                        "--format=" + fmt,
                        "--name=data_hub_gcamp",
                        "--flip_image"]))

    # Writers
    ## Behavior
    job.append(Popen(["cfm_writer",
                        "--data_in=L" + data_stamped_behavior,
                        "--commands_in=L" + forwarder_out,
                        "--status_out=L" + forwarder_in,
                        "--format=" + fmt,
                        "--directory="+ data_directory,
                        "--video_name=flircamera_behavior",
                        "--name=writer_behavior"]))
    ## GCaMP
    job.append(Popen(["cfm_writer",
                        "--data_in=L" + data_stamped_gcamp,
                        "--commands_in=L" + forwarder_out,
                        "--status_out=L" + forwarder_in,
                        "--format=" + fmt,
                        "--directory="+ data_directory,
                        "--video_name=flircamera_gcamp",
                        "--name=writer_gcamp"]))

    # Display
    ## Behavior
    job.append(Popen(["cfm_displayer",
                          "--inbound=L" + tracker_out_behavior,
                          "--format=" + fmt,
                          "--commands=L" + forwarder_out,
                          "--name=displayer_behavior"]))
    ## GCaMP
    job.append(Popen(["cfm_displayer",
                          "--inbound=L" + tracker_out_gcamp,
                          "--format=" + fmt,
                          "--commands=L" + forwarder_out,
                          "--name=displayer_gcamp"]))
    ## Debug Display
    job.append(Popen(["cfm_displayer",
                          "--inbound=L" + tracker_out_debug,
                          "--format=" + fmt,
                          "--commands=L" + forwarder_out,
                          "--name=displayer_debug"]))

    # Logger
    job.append(Popen(["cfm_logger",
                      "--inbound=" + forwarder_out,
                      "--directory=" + logger_directory]))

    # Trackers
    # TODO: add selection between only one tracker being activated
    ## Behavior
    job.append(Popen(["cfm_tracker",
                      "--commands_in=L" + forwarder_out,
                      "--commands_out=L" + forwarder_in,
                      "--data_in=L" + data_stamped_behavior,
                      "--data_out=" + tracker_out_behavior,
                      "--format=" + fmt,
                      "--interpolation_tracking=" + str(interpolation_tracking),
                      "--name=tracker_behavior",
                      "--data_out_debug=" + tracker_out_debug]))
    ## GCaMP
    job.append(Popen(["cfm_tracker",
                      "--commands_in=L" + forwarder_out,
                      "--commands_out=L" + forwarder_in,
                      "--data_in=L" + data_stamped_gcamp,
                      "--data_out=" + tracker_out_gcamp,
                      "--format=" + fmt,
                      "--interpolation_tracking=" + str(interpolation_tracking),
                      "--name=tracker_gcamp",
                      "--data_out_debug=" + tracker_out_debug]))

    job.append(Popen(["cfm_teensy_commands",
                      "--inbound=L" + forwarder_out,
                      "--outbound=L" + forwarder_in,
                      "--port=" + teensy_usb_port]))





def run(
    fmt: str,
    camera_serial_number_behavior: str,
    camera_serial_number_gcamp: str,
    binsize: str,
    exposure_behavior: str,
    exposure_gcamp: str,
    interpolation_tracking: bool,
):
    """Run all system devices."""

    jobs = []

    def finish(*_):
        for job in jobs:
            try:
                job.kill()
            except PermissionError as _e:
                print("Received error closing process: ", _e)

        raise SystemExit

    signal.signal(signal.SIGINT, finish)

    execute(
        jobs, fmt,
        camera_serial_number_behavior,
        camera_serial_number_gcamp,
        binsize,
        exposure_behavior,
        exposure_gcamp,
        interpolation_tracking
    )

    while True:
        time.sleep(1)


def main():
    """CLI entry point."""
    args = docopt(__doc__)

    run(
        fmt=args["--format"],
        camera_serial_number_behavior=args["--camera_serial_number_behavior"],
        camera_serial_number_gcamp=args["--camera_serial_number_gcamp"],
        binsize=args["--binsize"],
        exposure_behavior=args["--exposure_behavior"],
        exposure_gcamp=args["--exposure_gcamp"],
        interpolation_tracking=args["--interpolation_tracking"]
    )

if __name__ == "__main__":
    main()
