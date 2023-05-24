#! python
#
# Copyright 2021
# Author: Mahdi Torkashvand, Vivek Venkatachalam

"""
This subscribes to a binary stream bound to a socket and displays each message
as an image.

Usage:
    displayer.py                    [options]

Options:
    -h --help                       Show this help.
    --inbound=HOST:PORT             Connection for inbound messages.
                                        [default: L5005]
    --commands=HOST:PORT            Connection to recieve messages.
                                        [default: L5001]
    --format=FORMAT                 Size and type of image being sent.
                                        [default: UINT16_ZYX_25_512_1024]
    --name=STRING                   Name of image window.
                                        [default: displayer]
"""

from typing import Tuple

import cv2  # install specific version: pip install opencv-python==4.5.2.54
import zmq
import numpy as np
from docopt import docopt

from cfm.zmq.utils import parse_host_and_port
from cfm.zmq.subscriber import ObjectSubscriber
from cfm.zmq.array import TimestampedSubscriber
from cfm.devices.utils import array_props_from_string

class Displayer:
    """This creates a displayer with 2 subscribers, one for images
    and one for commands."""
    def __init__(
            self,
            inbound: Tuple[str, int],
            commands: Tuple[str, int, bool],
            fmt: str,
            name: str):

        (_, _, self.shape) = array_props_from_string(fmt)
        self.dtype = np.uint8
        self.image = np.zeros(self.shape, self.dtype)

        self.name = name
        self.running = True
        self.inbound = inbound
        self._DEBUG = 'debug' in self.name.lower() and False
        self._DEBUG_IDX = 0

        self.poller = zmq.Poller()

        self.command_subscriber = ObjectSubscriber(
            obj=self,
            name=name,
            host=commands[0],
            port=commands[1],
            bound=commands[2])

        self.data_subscriber = TimestampedSubscriber(
            host=self.inbound[0],
            port=self.inbound[1],
            shape=self.shape,
            datatype=self.dtype,
            bound=self.inbound[2])

        self.poller.register(self.command_subscriber.socket, zmq.POLLIN)
        self.poller.register(self.data_subscriber.socket, zmq.POLLIN)

        cv2.namedWindow(self.name)
        cv2.resizeWindow(self.name, self.shape[1], self.shape[0])
        if self.name[-5:] == "gcamp":  # TODO DEBUG WHAT IS THIS?
            cv2.moveWindow(self.name, self.shape[1] + 8, 0)
        else:
            cv2.moveWindow(self.name, 0, 0)
    def set_shape(self, y, x):
        self.poller.unregister(self.data_subscriber.socket)

        self.shape = (y, x)

        self.data_subscriber.set_shape(self.shape)

        self.image = np.zeros(self.shape, self.dtype)

        self.poller.register(self.data_subscriber.socket, zmq.POLLIN)

        cv2.namedWindow(self.name)
        cv2.resizeWindow(self.name, self.shape[1], self.shape[0])
        if self.name[-5:] == "gcamp":
            cv2.moveWindow(self.name, self.shape[1] + 8, 0)
        else:
            cv2.moveWindow(self.name, 0, 0)

    def process(self):
        msg = self.data_subscriber.get_last()

        if msg is not None:
            self.image = msg[1]
        
        self._DEBUG_IDX += 1
        if self._DEBUG and self._DEBUG_IDX%40 == 0:
            print(f"{self.name}: runnning {self._DEBUG_IDX:>9}")

        cv2.imshow(self.name, self.image)
        cv2.waitKey(1)

    def run(self):
        while self.running:

            sockets = dict(self.poller.poll())

            if self.command_subscriber.socket in sockets:
                self.command_subscriber.handle()

            elif self.data_subscriber.socket in sockets:
                self.process()

    def shutdown(self):
        self.running = False

def main():
    """CLI entry point."""

    args = docopt(__doc__)

    displayer = Displayer(inbound=parse_host_and_port(args["--inbound"]),
                       commands=parse_host_and_port(args["--commands"]),
                       fmt=args["--format"],
                       name=args["--name"])

    displayer.run()

if __name__ == "__main__":
    main()
