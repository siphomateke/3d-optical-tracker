from urllib2 import urlopen
from urllib2 import HTTPError, URLError
import xml.etree.ElementTree as ET
from threadutil import ProgramThread
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


class CamBase:
    def __init__(self, data_filename="", name="Cam1"):
        self.name = name
        self.capture = cv2.VideoCapture()
        self.frame = None
        self.ready = False

        # TODO: Add assertions for loading camera data

        self.data_filename = data_filename
        if len(self.data_filename) > 0:
            self.data = CamData(data_filename)
        else:
            self.data = None

        print "{} initialised.".format(self.name)

    def start(self):
        """
        Initialize camera settings or connections
        """
        return True

    def stop(self):
        """
        Wrap up  execution or stop threads
        """
        return True

    def imshow(self, name, img):
        if len(name) > 0:
            cv2.imshow("{} - {}".format(self.name, name), img)
        else:
            cv2.imshow("{}".format(self.name, name), img)


class ImgCam(CamBase):
    def __init__(self, filename, data_filename="", name="ImgCam1"):
        CamBase.__init__(self, data_filename=data_filename, name=name)
        # TODO: Add file checks for reading image file
        self.frame = cv2.imread(filename)
        self.ready = True


class IPCam(CamBase, ProgramThread):
    def __init__(self, url, data_filename="", name="IPCam1"):
        CamBase.__init__(self, data_filename=data_filename, name=name)
        ProgramThread.__init__(self, self.run)
        self.url = url
        self.url_video = self.url + "video"

        self.torch_on = False
        self.mean_brightness = None
        self.prev_mean_brightness = None

    def start(self):
        print("{} attempting connection to {} ...".format(self.name, self.url_video))
        if self.capture.open(self.url_video):
            self.request_action("disabletorch")
            self.start_thread()
            print "Camera stream started."
            return True
        else:
            print "Error opening camera."
            return False

    def run(self):
        ret, img = self.capture.read()
        if ret:
            self.frame = img.copy()
            self.ready = True
            # self.check_lighting()

    def stop(self):
        self.stop_thread()

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

    def request_action(self, command):
        """
        Sends a http command to the ip camera
        :param command: The command to execute. e.g 'enabletorch'
        :return: Whether the command was successfully executed
        """
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
    """
    Controls and stores a group of cameras and their settings
    """
    def __init__(self, cameras):
        self.cameras = cameras  # type: list[CamBase]
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
            if cam.ready:
                yield cam

    def stop(self):
        for cam in self.cameras:
            cam.stop()
