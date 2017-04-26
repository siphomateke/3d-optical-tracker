import numpy as np
import cv2
import ipcamutil
import math
from urllib2 import urlopen
import json
import serial_utils
import serial


def get_jsonparsed_data(url):
    response = urlopen(url)
    data = str(response.read())
    return json.loads(data)


def rotate_img(img, x_rot=90, y_rot=90, z_rot=90, dx=0, dy=0, dz=200, f=200):
    x_rot = (x_rot - 90.) * math.pi / 180.
    y_rot = (y_rot - 90.) * math.pi / 180.
    z_rot = (z_rot - 90.) * math.pi / 180.
    # get width and height for ease of use in matrices
    w, h = img.shape[1], img.shape[0]
    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 0],
                   [0, 0, 1]])
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([
        [1, 0, 0, 0],
        [0, math.cos(x_rot), -math.sin(x_rot), 0],
        [0, math.sin(x_rot), math.cos(x_rot), 0],
        [0, 0, 0, 1]])
    RY = np.array([
        [math.cos(y_rot), 0, -math.sin(y_rot), 0],
        [0, 1, 0, 0],
        [math.sin(y_rot), 0, math.cos(y_rot), 0],
        [0, 0, 0, 1]])
    RZ = np.array([
        [math.cos(z_rot), -math.sin(z_rot), 0, 0],
        [math.sin(z_rot), math.cos(z_rot), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)
    # Translation matrix
    T = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]])
    # 3D -> 2D matrix
    A2 = np.array([
        [f, 0, w / 2, 0],
        [0, f, h / 2, 0],
        [0, 0, 1, 0]])
    # Final transformation matrix
    trans = np.dot(A2, np.dot(T, (np.dot(R, A1))))
    # Apply matrix transformation
    return cv2.warpPerspective(img, trans, (w, h), flags=cv2.INTER_LANCZOS4)


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    """# compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY"""

    nW = image.shape[1]
    nH = image.shape[0]

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


class Sensor:
    def __init__(self, url):
        self.url = url
        self.accel = None
        self.cur_accel = None
        self.lin_accel = None
        self.cur_lin_accel = None
        self.yaw = AngleFilter(0.5)
        self.pitch = AngleFilter(0.5)
        self.roll = AngleFilter(0.5)
        self.translate = 0
        self.G = 9.81
        self.gps = None
        self.network_loc = None
        self.loc = None
        self.mag = None

    def get_sensor_reading(self, sensor):
        return sensor[len(sensor) - 1][1]

    def get_accel(self):
        return self.cur_accel / self.G

    def update(self):
        sensors_data = get_jsonparsed_data(self.url + "sensors.json")
        gps_data = get_jsonparsed_data(self.url + "gps.json")
        if "gps" in gps_data:
            self.gps = gps_data["gps"]
            self.network_loc = gps_data["network"]
            if self.gps["accuracy"] > self.network_loc["accuracy"]:
                self.loc = [self.gps["latitude"], self.gps["longitude"], self.gps["altitude"]]
            else:
                self.loc = [self.network_loc["latitude"], self.network_loc["longitude"], self.gps["altitude"]]
        if "mag" in sensors_data:
            self.mag = sensors_data["mag"]["data"]
            self.cur_mag = self.get_sensor_reading(self.mag)
        self.accel = sensors_data["accel"]["data"]
        self.cur_accel = np.array(self.get_sensor_reading(self.accel))
        if "lin_accel" in sensors_data:
            self.lin_accel = sensors_data["lin_accel"]["data"]
            self.cur_lin_accel = np.array(self.get_sensor_reading(self.lin_accel))
        ax = self.cur_accel[0]
        ay = self.cur_accel[1]
        az = self.cur_accel[2]
        self.G = math.sqrt((math.pow(ax, 2) + math.pow(ay, 2) + math.pow(az, 2)))

        new_yaw = math.atan2(ax, ay)
        new_pitch = math.atan2(ay, math.sqrt(math.pow(ax, 2) + math.pow(az, 2)))
        self.pitch2 = (-(math.acos(ay / self.G) - (math.pi / 2.0))) * (180.0 / math.pi)
        new_roll = math.atan2(ax, az)
        self.yaw.update(new_yaw)
        self.pitch.update(new_pitch)
        self.roll.update(new_roll)
        self.translate += ((self.cur_accel[2] / self.G) - self.translate) / 4.0

    def list_to_str(self, list):
        str = ""
        for i in range(len(list)):
            str += "{}".format(list[i])
            if i < len(list) - 1:
                str += ","
        return str

    def display(self):
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 150), (450, 450), (0, 0, 255), 2)
        cv2.circle(img, (int((self.cur_accel[0] / self.G) * -150) + 300, int((self.cur_accel[1] / self.G) * 150) + 300),
                   5, (0, 255, 255), -1)
        cv2.circle(img, (int(self.roll.get_angle() / 180 * -300) + 300, int(self.pitch.get_angle() / 180 * 300) + 300),
                   5, (0, 255, 0), -1)
        cv2.circle(img, (int(self.roll.get_angle() / 180 * -300) + 300, int(self.pitch2 / 180 * 300) + 300),
                   5, (255, 0, 0), -1)
        cv2.imshow("Accel", img)


ipcamera_url = "http://192.168.8.100:8080/"
cam = ipcamutil.Cam(ipcamera_url + "video")
cam.start()
sensor = Sensor(ipcamera_url)

then = cv2.getTickCount()
ser = serial_utils.SerialStream(serial.Serial(baudrate=9600, port="COM3", timeout=0), "a")
ser.open()

timer = 0

orientation = np.array([0, 0, 0])
accelerometer = np.array([0, 0, 0])

while True:
    now = cv2.getTickCount()
    delta = ((now - then) / cv2.getTickFrequency())
    then = now

    sensor.update()
    timer += delta
    if timer * 1000.0 > 10:
        if not ser.reading:
            ser.write("m" + sensor.list_to_str(sensor.cur_mag))
            timer = 0
    lines = ser.get_data()
    if len(lines) > 0:
        line = lines[len(lines) - 1]
        data = line.split(",")
        #orientation = np.float64([float(data[0]), float(data[1]), float(data[2])])
        #accelerometer = np.float64([float(data[3]), float(data[4]), float(data[5])])
    # agent.update(sensor.cur_accel, delta)
    loc = np.array([0, 0, 0])
    if cam.is_opened():
        frame = cam.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        transformed = rotate_bound(frame, -sensor.yaw.get_angle() + 90)
        if len(orientation) > 0:
            imu_img = np.zeros((600, 600, 3), dtype=np.uint8)
            cv2.circle(imu_img, (300, 300), 150, (0, 255, 255), 1)
            cv2.ellipse(imu_img, (300, 300), (150, 150), angle=-90.0 + float(orientation[0]), startAngle=0.0,
                        endAngle=5.0, color=(0, 0, 255), thickness=-1)
            cv2.putText(imu_img, str(sensor.cur_accel), (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5,
                        (255, 255, 255))
            cv2.putText(imu_img, str(loc), (50, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5,
                        (255, 255, 255))
            scale = 1000
            cv2.circle(imu_img, (300 + int(loc[0] * scale), 300 + -int(loc[1] * scale)), 5, (0, 255, 255),
                       -1)
            cv2.imshow("IMU", imu_img)
        cv2.imshow('Camera', transformed)

        sensor.display()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

ser.close()
cam.shut_down()
cv2.destroyAllWindows()
