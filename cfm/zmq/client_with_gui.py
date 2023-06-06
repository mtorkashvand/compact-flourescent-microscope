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
    --port_forwarder_in=PORT           [default: L5000]
"""

DEBUG = True

import time
import signal

import zmq
from docopt import docopt

from cfm.zmq.publisher import Publisher
from cfm.zmq.utils import parse_host_and_port

from cfm.zmq.utils import (
    coerce_string,
    coerce_bytes
)

class GUIClient():
    """This is a wrapped ZMQ client that can send requests to a server."""

    def __init__(
            self,
            port: int,
            port_forwarder_in: str
        ):

        self.port = port
        self.bounds_forwarder_in = parse_host_and_port(port_forwarder_in)
        self.running = False

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)

        address = "tcp://localhost:{}".format(self.port)
        self.socket.connect(address)

        self.control_socket = self.context.socket(zmq.PUB)
        self.control_socket.bind("tcp://*:4862")  # DEBUG TODO change this to an argument for the class

        self.forwarder_publisher = Publisher(
            host=self.bounds_forwarder_in[0],
            port=self.bounds_forwarder_in[1],
            bound=self.bounds_forwarder_in[2]
        )

    def recv(self) -> bytes:
        """Receive a reply."""

        return self.socket.recv()

    def send(self, req: bytes):
        """Send a request."""

        self.socket.send(req)
    
    def log(self, msg):
        self.forwarder_publisher.send("logger "+ str(msg))

    def process(self, req_str):
        """Take a single request from stdin, send
        it to a server, and return the reply."""
        if not self.running:
            return
        self.send(coerce_bytes(req_str))
        self.log(f"<CLIENT WITH GUI> command sent: {req_str}")
        if not DEBUG:
            resp_str = coerce_string(self.recv())
            print(resp_str)
            self.log(f"<CLIENT WITH GUI> response received: {resp_str}")
        if req_str == "DO shutdown":
            time.sleep(1)
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
    port_forwarder_in = args["--port_forwarder_in"]

    client = GUIClient(port=port, port_forwarder_in=port_forwarder_in)
    client.run()


if __name__ == "__main__":
    main()
