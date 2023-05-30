#! python
#
# Copyright 2021
# Author: Mahdi Torkashvand, Vivek Venkatachalam

"""
ZMQ Client.

Usage:
    client.py             [options]

Options:
    -h --help             Show this help.
    --port=PORT           [default: 5002]
"""

import time
import signal

import zmq
from docopt import docopt

from cfm.zmq.utils import (
    coerce_string,
    coerce_bytes
)

class GUIClient():
    """This is a wrapped ZMQ client that can send requests to a server."""

    def __init__(
            self,
            port):

        self.port = port
        self.running = False

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)

        address = "tcp://localhost:{}".format(self.port)
        self.socket.connect(address)

        self.control_socket = self.context.socket(zmq.PUB)
        self.control_socket.bind("tcp://*:4862")  # DEBUG TODO change this to an argument for the class

    def recv(self) -> bytes:
        """Receive a reply."""

        return self.socket.recv()

    def send(self, req: bytes):
        """Send a request."""

        self.socket.send(req)

    def process(self, req_str):
        """Take a single request from stdin, send
        it to a server, and return the reply."""
        if not self.running:
            return
        if req_str == "DO shutdown":
            self.running = False
        self.send(coerce_bytes(req_str))
        print(coerce_string(self.recv()))
        time.sleep(5)
        self.control_socket.send_string("TERMINATE")


    def run(self):
        """Start looping."""

        self.running = True
        def _finish(*_):
            raise SystemExit

        signal.signal(signal.SIGINT, _finish)

def main():
    """CLI entry point."""

    args = docopt(__doc__)
    port = int(args["--port"])

    client = GUIClient(port)
    client.run()


if __name__ == "__main__":
    main()
