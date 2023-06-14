#! python
#
# Copyright 2023
# Author: Mahdi Torkashvand

"""
Query controller information and publish them via ZMQ PUB.

Usage:
    xinput_pub.py                       [options]

Options:
    -h --help                           Show this help.
    --name=NAME                         Name of the XBox controller subscriber.
                                            [default: XInput]
    --outbound=HOST:PORT                Connection for outbound messages.
                                            [default: *:5557]
    --inbound=HOST:PORT                Connection for outbound messages.
                                            [default: *:5558]
"""

import time
import signal
import struct
from typing import Tuple

import zmq
import XInput
from docopt import docopt


from cfm.zmq.subscriber import ObjectSubscriber
from cfm.zmq.publisher import Publisher
from cfm.zmq.utils import parse_host_and_port



class  XInputToZMQPub():
    """This is xinput class"""

    def __init__(
            self,
            name: str,
            outbound: Tuple[str, int],
            inbound: Tuple[str, int]):
        self.name = name
        self.publisher = Publisher(
            host=outbound[0],
            port=outbound[1],
            bound=outbound[2]
        )
        self.publisher.socket.setsockopt(zmq.CONFLATE, 1)
        
        self.subscriber = ObjectSubscriber(
            obj=self,
            host=inbound[0],
            port=inbound[1],
            bound=inbound[2],
            name=self.name
        )

        self.running = True
        
        self.packet_number = 0
    
    def shutdown(self):
        self.running = False
        msg = struct.pack(b'HBBhhhh', 0, 0, 0, 0, 0, 0, 0)
        self.publisher.socket.send(msg, flags=0)

    def run(self):

        def _finish(*_):
            raise SystemExit

        signal.signal(signal.SIGINT, _finish)

        while self.running:
            # Receive Command
            msg = self.subscriber.recv_last()
            if msg is not None:
                print(msg)
                self.subscriber.process(msg)
            # Get XInput States
            state = XInput.get_state(0)
            if state.dwPacketNumber != self.packet_number:
                self.packet_number = state.dwPacketNumber
                result = state.Gamepad
                msg = struct.pack(b'HBBhhhh',
                                  result.wButtons,
                                  result.bLeftTrigger, result.bRightTrigger,
                                  result.sThumbLX, result.sThumbLY, result.sThumbRX, result.sThumbRY)
                self.publisher.socket.send(msg, flags=0)
            time.sleep(0.01)

def main():
    """CLI entry point."""

    args = docopt(__doc__)

    xinput_to_zmq_pub = XInputToZMQPub(
        name=args["--name"],
        outbound=parse_host_and_port(args["--outbound"]),
        inbound=parse_host_and_port(args["--inbound"])
    )
    xinput_to_zmq_pub.run()


if __name__ == "__main__":
    main()