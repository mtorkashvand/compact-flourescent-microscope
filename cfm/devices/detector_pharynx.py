#! python
#
# Copyright 2021
# Author: Sina Rasouli, Mahdi Torkashvand
#
# This is a NeuralNetwork interface for processing images and returning position of the pharynx in the image.

"""
Process image, detech pharynx and return position of the pharynx.

Usage:
    detector_pharynx.py             [options]

Options:
    -h --help               Show this help.
    --input_image=HOST:PORT     Connection for receiving images.
                                [default: L6001]
    --to_forwarder=HOST:PORT    Connection for outputing positions to forwarder, e.g. setting position for tracker.
                                [default: 6002]
    --input_commands=HOST:PORT  Connection for receiving commands, e.g. start and shutdown.
                                [default: 5001]
"""


import onnxruntime

import signal
from typing import Tuple

import numpy as np

import zmq
from docopt import docopt

from cfm.zmq.publisher import Publisher
from cfm.zmq.subscriber import ObjectSubscriber
from cfm.zmq.array import Subscriber as ArraySubscriber
from cfm.zmq.utils import parse_host_and_port

class DetectorPharynx():

    def __init__(self,
                 input_image: Tuple[str, int],
                 to_forwarder: Tuple[str, int],
                 input_commands: Tuple[str, int]):


        self.subscriber_input_image = ArraySubscriber(
            host=input_image[0],
            port=input_image[1],
            bound=input_image[2],
            shape=(512,512),
            datatype=np.uint8
        )

        self.publisher_to_forwarder = Publisher(to_forwarder[1],
                                   to_forwarder[0],
                                   to_forwarder[2])
        self.subscriber_commands = ObjectSubscriber(
            self,
            input_commands[1],
            input_commands[0],
            input_commands[2],
            name = "detector_pharynx"  # TODO: change this hard coded name -> also in `cfm_with_gui.py`
        )


        self.poller = zmq.Poller()
        self.poller.register(self.subscriber_input_image.socket)
        self.poller.register(self.subscriber_commands.socket)

        self.ort_session = onnxruntime.InferenceSession(
            r"C:\src\wormtracker\model_detection_pharynx.onnx"
        )
        self.running = True
        

    def shutdown(self):
        self.running = False
        return
    
    def detect_pharynx(self, img):
        img_cropped = img[56:-56,56:-56]
        batch_1_400_400 = {
            'input': np.repeat(
                img_cropped[None, None, :, :], 3, 1
            ).astype(np.float32)
        }
        ort_outs = self.ort_session.run( None, batch_1_400_400 )
        x, y = ort_outs[0][0].astype(np.int64) + 56
        self.publisher_to_forwarder.send(f"tracker_behavior set_worm_xy {x} {y}")
        return

    def run(self):
        def _finish(*_):
            raise SystemExit

        signal.signal(signal.SIGINT, _finish)

        while self.running:

            sockets = dict(self.poller.poll())

            if self.subscriber_input_image.socket in sockets:
                img = self.subscriber_input_image.get_last()
                self.detect_pharynx(img)

            if self.subscriber_commands.socket in sockets:
                self.subscriber_commands.handle()

def main():
    """CLI entry point."""
    arguments = docopt(__doc__)

    input_image = parse_host_and_port(arguments["--input_image"])
    to_forwarder = parse_host_and_port(arguments["--to_forwarder"])
    input_commands = parse_host_and_port(arguments["--input_commands"])

    detector = DetectorPharynx(
        input_image=input_image,
        to_forwarder=to_forwarder,
        input_commands=input_commands
    )

    detector.run()

if __name__ == "__main__":
    main()






