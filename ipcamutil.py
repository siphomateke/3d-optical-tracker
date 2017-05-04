import time
from threading import Thread, ThreadError
import cv2


class Cam:
    def __init__(self, url):
        self.url = url + "video"
        self.capture = cv2.VideoCapture()
        self.thread_cancelled = False
        self.thread = Thread(target=self.run)
        self.frame = None
        self.opened = False

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
