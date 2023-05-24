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
    --forwarder_in=FORWARDER_IN         FORWARDER_IN.
                                            [default: 5000]
    --forwarder_out=FORWARDER_OUT         FORWARDER_OUT.
                                            [default: 5001]
    --server_client=SERVER_CLIENT         SERVER_CLIENT.
                                            [default: 5002]
    --XInputToZMQPub_out=XInputToZMQPub_OUT         XInputToZMQPub_OUT.
                                            [default: 6000]
    --processor_out=PROCESSOR_OUT         PROCESSOR_OUT.
                                            [default: 6001]
    --data_camera_out_behavior=DATA_CAMERA_OUT_BEHAVIOR         DATA_CAMERA_OUT_BEHAVIOR.
                                            [default: 5003]
    --data_stamped_behavior=DATA_STAMPED_BEHAVIOR         DATA_STAMPED_BEHAVIOR.
                                            [default: 5004]
    --tracker_out_behavior=TRACKER_OUR_BEHAVIOR         TRACKER_OUR_BEHAVIOR.
                                            [default: 5005]
    --data_camera_out_gcamp=DATA_CAMERA_OUT_GCAMP         DATA_CAMERA_OUT_GCAMP.
                                            [default: 5006]
    --data_stamped_gcamp=DATA_STAMPED_GCAMP         DATA_STAMPED_GCAMP.
                                            [default: 5007]
    --tracker_out_gcamp=TRACKER_OUT_GCAMP         TRACKER_OUT_GCAMP.
                                            [default: 5008]
    --tracker_out_debug=TRACKER_OUT_DEBUG   TRACKER_OUT_DEBUG.
                                            [default: 5009]
    --teensy_usb_port=TEENSY_USB_PORT   TRACKER_OUT_DEBUG.
                                            [default: COM4]
    --framerate=FRAMERATE   TRACKER_OUT_DEBUG.
                                            [default: 20]
    --data_directory=DATA_DIRECTORY   TRACKER_OUT_DEBUG.
                                            [default: C:\src\data]
    --logger_directory=LOGGER_DIRECTORY   TRACKER_OUT_DEBUG.
                                            [default: C:\src\data]
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
        **kwargs
    ):
    """This runs all devices."""

    camera_serial_number_behavior = kwargs['camera_serial_number_behavior']
    camera_serial_number_gcamp = kwargs['camera_serial_number_gcamp']
    binsize = kwargs['binsize']
    exposure_behavior = kwargs['exposure_behavior']
    exposure_gcamp = kwargs['exposure_gcamp']
    interpolation_tracking =  kwargs['interpolation_tracking'].lower() == 'true'

    forwarder_in = kwargs['forwarder_in']
    forwarder_out = kwargs['forwarder_out']
    server_client = kwargs['server_client']
    XInputToZMQPub_out = kwargs['XInputToZMQPub_out']
    processor_out = kwargs['processor_out']
    data_camera_out_behavior = kwargs['data_camera_out_behavior']
    data_stamped_behavior = kwargs['data_stamped_behavior']
    tracker_out_behavior = kwargs['tracker_out_behavior']
    data_camera_out_gcamp = kwargs['data_camera_out_gcamp']
    data_stamped_gcamp = kwargs['data_stamped_gcamp']
    tracker_out_gcamp = kwargs['tracker_out_gcamp']
    tracker_out_debug  = kwargs['tracker_out_debug']

    teensy_usb_port = kwargs['teensy_usb_port']  # default: "COM4"
    framerate = kwargs['framerate']

    format = kwargs['format']
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

    data_directory = kwargs['data_directory']  # "C:\src\data"
    logger_directory = kwargs['logger_directory']  # "C:\src\data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    if not os.path.exists(logger_directory):
        os.makedirs(logger_directory)

    job.append(Popen(["cfm_processor",
                      "--name=controller",
                      "--inbound=L" + forwarder_out,
                      "--outbound=" + processor_out,
                      "--deadzone=5000",
                      "--threshold=50"]))

    job.append(Popen(["cfm_commands",
                      "--inbound=L" + processor_out,
                      "--outbound=L" + forwarder_in,
                      "--commands=L" + forwarder_out]))


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
                        "--format=" + format,
                        "--name=data_hub_behavior"]))
    ## GCaMP
    job.append(Popen(["cfm_data_hub",
                        "--data_in=L" + data_camera_out_gcamp,
                        "--commands_in=L" + forwarder_out,
                        "--status_out=L" + forwarder_in,
                        "--data_out=" + data_stamped_gcamp,
                        "--format=" + format,
                        "--name=data_hub_gcamp"]))

    # Writers
    ## Behavior
    job.append(Popen(["cfm_writer",
                        "--data_in=L" + data_stamped_behavior,
                        "--commands_in=L" + forwarder_out,
                        "--status_out=L" + forwarder_in,
                        "--format=" + format,
                        "--directory="+ data_directory,
                        "--video_name=flircamera_behavior",
                        "--name=writer_behavior"]))
    ## GCaMP
    job.append(Popen(["cfm_writer",
                        "--data_in=L" + data_stamped_gcamp,
                        "--commands_in=L" + forwarder_out,
                        "--status_out=L" + forwarder_in,
                        "--format=" + format,
                        "--directory="+ data_directory,
                        "--video_name=flircamera_gcamp",
                        "--name=writer_gcamp"]))

    # Display
    ## Behavior
    job.append(Popen(["cfm_displayer",
                          "--inbound=L" + tracker_out_behavior,
                          "--format=" + format,
                          "--commands=L" + forwarder_out,
                          "--name=displayer_behavior"]))
    ## GCaMP
    job.append(Popen(["cfm_displayer",
                          "--inbound=L" + tracker_out_gcamp,
                          "--format=" + format,
                          "--commands=L" + forwarder_out,
                          "--name=displayer_gcamp"]))
    ## Debug Display
    job.append(Popen(["cfm_displayer",
                          "--inbound=L" + tracker_out_debug,
                          "--format=" + format,
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
                      "--format=" + format,
                      "--interpolation_tracking=" + str(interpolation_tracking),
                      "--name=tracker_behavior",
                      "--data_out_debug=" + tracker_out_debug]))
    ## GCaMP
    job.append(Popen(["cfm_tracker",
                      "--commands_in=L" + forwarder_out,
                      "--commands_out=L" + forwarder_in,
                      "--data_in=L" + data_stamped_gcamp,
                      "--data_out=" + tracker_out_gcamp,
                      "--format=" + format,
                      "--interpolation_tracking=" + str(interpolation_tracking),
                      "--name=tracker_gcamp",
                      "--data_out_debug=" + tracker_out_debug]))

    job.append(Popen(["cfm_teensy_commands",
                      "--inbound=L" + forwarder_out,
                      "--outbound=L" + forwarder_in,
                      "--port=" + teensy_usb_port]))





def run(
        **kwargs
    ):
    """Run all system devices."""

    jobs = []

    def finish(*_):
        print("INTERRUPTEDDDD!!!")
        for job in jobs:
            try:
                job.kill()
            except PermissionError as _e:
                print("Received error closing process: ", _e)

        raise SystemExit

    signal.signal(signal.SIGINT, finish)

    execute(
        jobs, **kwargs
    )

    while True:
        time.sleep(1)


def main():
    """CLI entry point."""
    args = docopt(__doc__)

    

    kwargs = dict(
        camera_serial_number_behavior = args["--camera_serial_number_behavior"],
        camera_serial_number_gcamp = args["--camera_serial_number_gcamp"],
        binsize = args["--binsize"],
        exposure_behavior = args["--exposure_behavior"],
        exposure_gcamp = args["--exposure_gcamp"],
        interpolation_tracking =  args["--interpolation_tracking"],
        format = args["--format"],

        forwarder_in = args['--forwarder_in'],
        forwarder_out = args['--forwarder_out'],
        server_client = args['--server_client'],
        XInputToZMQPub_out = args['--XInputToZMQPub_out'],
        processor_out = args['--processor_out'],
        data_camera_out_behavior = args['--data_camera_out_behavior'],
        data_stamped_behavior = args['--data_stamped_behavior'],
        tracker_out_behavior = args['--tracker_out_behavior'],
        data_camera_out_gcamp = args['--data_camera_out_gcamp'],
        data_stamped_gcamp = args['--data_stamped_gcamp'],
        tracker_out_gcamp = args['--tracker_out_gcamp'],
        tracker_out_debug  = args['--tracker_out_debug'],

        teensy_usb_port = args['--teensy_usb_port'],  # default: "COM4"
        framerate = args['--framerate'],

        data_directory = args['--data_directory'],  # "C:\src\data"
        logger_directory = args['--logger_directory'],  # "C:\src\data"
    )

    run(
        **kwargs
    )

if __name__ == "__main__":
    main()
