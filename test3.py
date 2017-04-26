import cv2
from camera_sensors import CameraSensors
from ipcamutil import Cam

ipcamera_url = "http://192.168.8.100:8080/"
cam = Cam(ipcamera_url + "video")
cam.start()
sensor = CameraSensors(ipcamera_url)

while True:
    now = cv2.getTickCount()
    delta = ((now - then) / cv2.getTickFrequency())
    then = now

    sensor.update()
    if cam.is_opened():
        frame = cam.get_frame()

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cam.shut_down()
cv2.destroyAllWindows()