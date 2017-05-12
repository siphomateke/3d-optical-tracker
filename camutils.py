import logging
from urllib2 import urlopen
from urllib2 import HTTPError, URLError
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import math

from threadutils import ProgramThread
import sfmutils


class CamData:
    def __init__(self):
        self.rms = -1  # Root Mean Square (RMS) represents re-projection error
        self.mtx = None
        self.dist = None

        self.rvec = None
        self.tvec = None

        self.pnp_rms = -1
        self.pnp_solved = False
        self.pos = None
        self.euler = None

        self._to_save = ["rms", "mtx", "dist", "rvec", "tvec", "pnp_rms", "pnp_solved", "pos", "euler"]

        self._cache = {}

    @staticmethod
    def create_from_data(rms, mtx, dist):
        cam_data = CamData()
        cam_data.rms = rms
        cam_data.mtx = mtx
        cam_data.dist = dist
        return cam_data

    @staticmethod
    def create_from_file(filename):
        cam_data = CamData()
        camera_settings = np.load(filename)
        if "mtx" not in camera_settings:
            err = "camera intrinsic matrix not found in camera settings file"
            raise ValueError(err)
        if "dist" not in camera_settings:
            err = "CamDat load error: distortion matrix not found in camera settings file"
            raise ValueError(err)
        # Load all parameters from file
        for attr in camera_settings:
            val = camera_settings[attr]
            setattr(cam_data, attr, val)
        return cam_data

    def save(self, path):
        # TODO: Add assertions for saving camera data
        dict = {}
        for attr in self._to_save:
            val = getattr(self, attr)
            dict[attr] = val
        # np.savez(path, rms=self.rms, mtx=self.mtx, dist=self.dist, rvec=self.rvec, tvec=self.tvec)
        np.savez(path, **dict)
        return True

    def _cache_get(self, name, func):
        """
        Gets a value from the cache if it exists. Otherwise set it to func
        :param name:
        :param func:
        :return:
        """
        if name not in self._cache:
            self._cache[name] = func()
        return self._cache[name]

    @property
    def fx(self):
        return self.mtx[0][0]

    @property
    def fy(self):
        return self.mtx[1][1]

    @property
    def cx(self):
        return self.mtx[0][2]

    @property
    def cy(self):
        return self.mtx[1][2]

    @property
    def rotation_matrix(self):
        return self._cache_get("r", lambda: cv2.Rodrigues(self.rvec)[0])

    def rotation_matrix_transpose(self):
        return self._cache_get("r_T", lambda: self.rotation_matrix.T)

    @property
    def rotation_matrix_inv(self):
        return self._cache_get("r_inv", lambda: np.matrix(np.linalg.inv(self.rotation_matrix)))

    @property
    def Rt(self):
        assert self.tvec.shape == (1, 3) or self.tvec.shape == (3, 1), "translation vectors must be a 1x3 or 3x1 matrix"
        return self._cache_get("Rt", lambda: np.hstack((self.rotation_matrix, self.tvec.reshape(3, 1))))

    @property
    def proj_mtx(self):
        return self._cache_get("proj_mtx", lambda: np.dot(self.mtx, self.Rt))

    def update_pnp_data(self, object_points, image_points):
        """
        Calculates PnP RMS error and camera position and rotation
        """
        object_points_screen, jac = cv2.projectPoints(object_points, self.rvec, self.tvec, self.mtx, None)
        self.pnp_rms = sfmutils.calc_reprojection_error(object_points_screen, image_points)

        r = self.rotation_matrix_transpose()
        self.pos = -np.matrix(r) * np.matrix(self.tvec)

        yaw = math.atan2(-r[1][0], r[0][0])
        pitch = math.atan2(-r[2][1], r[2][2])
        roll = math.asin(r[2][0])
        self.euler = np.array([yaw, pitch, roll]) * (180 / math.pi)

    def solve_pnp(self, object_points, image_points):
        ret, self.rvec, self.tvec, inlears = cv2.solvePnPRansac(object_points.reshape(-1, 3),
                                                                image_points.reshape(-1, 1, 2), self.mtx, None)
        self.update_pnp_data(object_points, image_points)

        return self.pnp_rms

    def solve_pnp_iterative(self, object_points, image_points):
        """
        Currently not used
        """
        if self.rvec is not None and self.tvec is not None:
            ret, self.rvec, self.tvec, inlears = cv2.solvePnPRansac(object_points.reshape(-1, 3),
                                                                    image_points.reshape(-1, 1, 2), self.mtx, None,
                                                                    rvec=self.rvec, tvec=self.tvec,
                                                                    useExtrinsicGuess=True)
            self.update_pnp_data(object_points, image_points)
        else:
            self.solve_pnp(object_points, image_points)

        return self.pnp_rms


class CamBase:
    def __init__(self, data_filename="", name="Cam1"):
        self.name = name
        self.ready = False
        self.frame = None
        self.frame_ready = False
        self.data_loaded = False

        # TODO: Improve exception handling and logging

        self.data = None
        self.data_filename = data_filename
        if len(self.data_filename) > 0:
            try:
                self.data = CamData.create_from_file(data_filename)
                self.data_loaded = True
            except IOError as err:
                print "Error loading camera data for {}: {} \n " \
                      "Make sure the path is correct and that the file exists.".format(self.name, err)
            except ValueError as err:
                print "Error loading camera data for {}: {}".format(self.name, err)

        else:
            self.data_loaded = False

        if self.data_loaded:
            print "{} initialised.".format(self.name)
        else:
            print "Failed to initialise {} data.".format(self.name)

    def start(self):
        """
        Initialize camera settings or connections
        """
        return self.ready

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
        self.img_filename = filename

    def start(self):
        """
        Loads the image
        :return: Whether the image is ready for reading frames
        """
        # TODO: Add file checks for reading image file
        self.frame = cv2.imread(self.img_filename)
        self.ready = False
        if not self.data_loaded:
            logging.error("ImgCam {} could not be initialised".format(self.name))
        else:
            if self.frame is not None:
                self.ready = True
                self.frame_ready = True
            else:
                logging.error("Error loading image for camera {}. " \
                              "Make sure the path is correct and that the file exists.".format(self.name))
        return self.ready


class IPCam(CamBase, ProgramThread):
    def __init__(self, url, data_filename="", name="IPCam1"):
        CamBase.__init__(self, data_filename=data_filename, name=name)
        ProgramThread.__init__(self, self.run)
        self.capture = cv2.VideoCapture()
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
            self.ready = True
        else:
            print "Error opening camera."
            self.ready = False
        return self.ready

    def run(self):
        ret, img = self.capture.read()
        if ret:
            self.frame = img.copy()
            self.frame_ready = True
            # self.check_lighting()

    def stop(self):
        self.stop_thread()
        self.capture.release()
        return True

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
            if cam.ready and cam.frame_ready:
                yield cam

    def stop(self):
        for cam in self.cameras:
            cam.stop()
