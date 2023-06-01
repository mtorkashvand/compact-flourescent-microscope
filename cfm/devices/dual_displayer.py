#! python
#
# Copyright 2023
# Author: Mahdi Torkashvand

from typing import Tuple
import PySimpleGUI as sg

import cv2 as cv  # install specific version: pip install opencv-python==4.5.2.54
import zmq
import numpy as np

from cfm.zmq.utils import parse_host_and_port
from cfm.zmq.array import TimestampedSubscriber
from cfm.devices.utils import array_props_from_string

DEBUG = False

class DualDisplayer:
    def __init__(
            self,
            window: sg.Window,
            data_r: str,
            data_g: str,
            fmt: str,
            name: str):

        self.window = window
        (_, _, self.shape) = array_props_from_string(fmt)
        self.dtype = np.uint8
        self.reset_buffers()

        self.name = name
        self.data_r = parse_host_and_port(data_r)
        self.data_g = parse_host_and_port(data_g)
        self.channel = [1, 1]

        self.poller = zmq.Poller()

        #TODO: how to update? how to read it from the checkpoint?
        self.warp_matrix = np.eye(2,3)

        self.subscriber_r = TimestampedSubscriber(
            host=self.data_r[0],
            port=self.data_r[1],
            shape=self.shape,
            datatype=self.dtype,
            bound=self.data_r[2])

        self.subscriber_g = TimestampedSubscriber(
            host=self.data_g[0],
            port=self.data_g[1],
            shape=self.shape,
            datatype=self.dtype,
            bound=self.data_g[2])

        self.poller.register(self.subscriber_r.socket, zmq.POLLIN)
        self.poller.register(self.subscriber_g.socket, zmq.POLLIN)
    
    def reset_buffers(self):
        self.image_r = np.zeros(self.shape, dtype=self.dtype)
        self.image_g = np.zeros(self.shape, dtype=self.dtype)
        self.image = np.zeros((*self.shape, 3), dtype=self.dtype)

    def set_shape(self, y, x):
        self.poller.unregister(self.subscriber_r.socket)
        self.poller.unregister(self.subscriber_g.socket)

        self.shape = (y, x)

        self.subscriber_r.set_shape(self.shape)
        self.subscriber_g.set_shape(self.shape)

        self.image = np.zeros(self.shape, self.dtype)

        self.poller.register(self.subscriber_r.socket, zmq.POLLIN)
        self.poller.register(self.subscriber_g.socket, zmq.POLLIN)

    #  TODO: add methods to load peviously save warp matrix or to update it
    def warp_r(self):
        # return cv.warpAffine(self.image_r, self.warp_matrix, *self.shape[::-1])
        return self.image_r

    def get_frame(self, combine=False):
        """
        Gets the last data from each camera, converts them into arrays, 
        returns individual arrays as well as a combination of them 
        if requested.
     
        params:
            combine: specifies the last returned value
                True: combined arrays
                False: none
        
        return:
            image_r: 2D array of data collected from behavior camera
            image_g: 2D array of data collected from the gcamp camera
            frame: A combination of image_r and image_g decided with param 'channel'
        """

        if DEBUG:
            self.image_r = np.random.randint(0, 256, size=self.shape)
            self.image_g = np.random.randint(0, 256, size=self.shape)
        else:
            sockets = dict(self.poller.poll())
            if self.subscriber_r.socket in sockets:
                msg_r = self.subscriber_r.get_last()
                if msg_r is not None:
                    self.image_r = msg_r[1]
            if self.subscriber_g.socket in sockets:
                msg_g = self.subscriber_g.get_last()
                if msg_g is not None:
                    self.image_g = msg_g[1]

        image_r_warped = self.warp_r()
        
        if combine:
            self.image[..., :] = (image_r_warped / 2).astype(np.uint8)[..., None]
            self.image[..., 1] += ( self.image_g / 2 ).astype(np.uint8)
            return image_r_warped, self.image_g, self.image
        return image_r_warped, self.image_g, None
