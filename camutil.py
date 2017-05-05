import time
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
        self.url = url + "video"
        self.name = name
        self.capture = cv2.VideoCapture()
        self.thread_cancelled = False
        self.thread = Thread(target=self.run)
        self.frame = None
        self.opened = False

        self.data_filename = data_filename
        if len(self.data_filename) > 0:
            self.data = CamData(data_filename)
        else:
            self.data = None

        if self.capture.open(self.url):
            ret, img = self.capture.read()
            self.shape = img.shape
        else:
            self.shape = None

        print "Camera initialised."

    def start(self):
        print("Attempting connection to {} ...".format(self.url))
        if self.capture.open(self.url):
            self.thread.start()
            print "Camera stream started."
            return True
        else:
            print "Error opening camera."
            return False

    def run(self):
        while not self.thread_cancelled:
            try:
                ret, img = self.capture.read()
                self.frame = img.copy()
                self.opened = True
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
