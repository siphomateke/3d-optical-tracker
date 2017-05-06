import time
from urllib2 import urlopen
from urllib2 import HTTPError, URLError
import xml.etree.ElementTree as ET
from threading import Thread, ThreadError
import cv2
import numpy as np


class CamData:
    def __init__(self, filename):
        self.camera_settings = np.load(filename)
        self.mtx = self.camera_settings["mtx"]
        self.fx = self.mtx[0][0]
        self.fy = self.mtx[1][1]
        self.cx = self.mtx[0][2]
        self.cy = self.mtx[1][2]
        self.dist = self.camera_settings["dist"]

        self.rvec = None
        self.tvec = None


class Cam:
    def __init__(self, url, data_filename="", name="Cam1"):
        self.url = url
        self.url_video = self.url + "video"
        self.name = name
        self.capture = cv2.VideoCapture()
        self.thread_cancelled = False
        self.thread = Thread(target=self.run)
        self.frame = None
        self.opened = False

        self.torch_on = False
        self.mean_brightness = None
        self.prev_mean_brightness = None

        self.data_filename = data_filename
        if len(self.data_filename) > 0:
            self.data = CamData(data_filename)
        else:
            self.data = None

        if self.capture.open(self.url_video):
            ret, img = self.capture.read()
            self.shape = img.shape
        else:
            self.shape = None

        print "Camera initialised."

    def start(self):
        print("Attempting connection to {} ...".format(self.url_video))
        if self.capture.open(self.url_video):
            self.request_action("disabletorch")
            self.thread.start()
            print "Camera stream started."
            return True
        else:
            print "Error opening camera."
            return False

    def check_lighting(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.mean_brightness = np.average(np.average(hsv[:, :, 2], axis=0), axis=0)
        if self.prev_mean_brightness is not None:
            diff = self.mean_brightness - self.prev_mean_brightness
            if self.mean_brightness < 2:
                if not self.torch_on:
                    self.request_action("enabletorch")
            elif self.mean_brightness > 90:
                if self.torch_on:
                    self.request_action("disabletorch")
        self.prev_mean_brightness = self.mean_brightness

    def run(self):
        while not self.thread_cancelled:
            try:
                ret, img = self.capture.read()
                self.frame = img.copy()
                self.opened = True
                # self.check_lighting()
            except ThreadError:
                self.thread_cancelled = True

    def is_opened(self):
        return self.opened

    def get_frame(self):
        return self.frame

    def is_running(self):
        return self.thread.isAlive()

    def shut_down(self):
        self.thread_cancelled = True
        # block while waiting for thread to terminate
        while self.thread.isAlive():
            time.sleep(1)
        return True

    def imshow(self, name, img):
        if len(name) > 0:
            cv2.imshow("{} - {}".format(self.name, name), img)
        else:
            cv2.imshow("{}".format(self.name, name), img)

    def request_action(self, command):
        success = True
        try:
            response = urlopen(self.url + command)
            xml_root = ET.fromstring(str(response.read()))
            if xml_root.tag.lower() == "result":
                if xml_root.text.lower() != "ok":
                    success = False
        except HTTPError as err:
            print("HTTP error: {}".format(err))
            success = False
        except URLError as err:
            print("URL error: {}".format(err))
            success = False
        if success:
            if command == "enabletorch":
                self.torch_on = True
            elif command == "disabletorch":
                self.torch_on = False
        return success


class CamManager:
    def __init__(self, cameras):
        self.cameras = cameras  # type: list[Cam]
        self._available_cameras = 0

    def start(self):
        self._available_cameras = 0
        for cam in self.cameras:
            cam_available = cam.start()
            if cam_available:
                self._available_cameras += 1

    @property
    def available_cameras(self):
        return self._available_cameras

    def get_frames(self):
        for cam in self.cameras:
            if cam.is_opened():
                frame = cam.get_frame()
                yield cam, frame

    def shut_down(self):
        for cam in self.cameras:
            cam.shut_down()
