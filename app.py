import cv2
import numpy as np

import sfmutils
from camutils import IPCam, CamManager
from marker_utils import MarkerFinder
from network import NetworkSocket
from threadutils import ProgramThread


def pt_arr_to_tuple(pt_arr):
    ravelled = pt_arr.ravel()
    return int(ravelled[0]), int(ravelled[1])


def draw_grid(img, pts, color):
    layers = np.array([pts[:4], pts[4:]])
    thickness = 2
    for i in xrange(2):
        layer = layers[i]
        pt1 = pt_arr_to_tuple(layer[0])
        pt2 = pt_arr_to_tuple(layer[1])
        cv2.line(img, pt1, pt2, color, thickness)
        pt1 = pt_arr_to_tuple(layer[1])
        pt2 = pt_arr_to_tuple(layer[3])
        cv2.line(img, pt1, pt2, color, thickness)
        pt1 = pt_arr_to_tuple(layer[3])
        pt2 = pt_arr_to_tuple(layer[2])
        cv2.line(img, pt1, pt2, color, thickness)
        pt1 = pt_arr_to_tuple(layer[2])
        pt2 = pt_arr_to_tuple(layer[0])
        cv2.line(img, pt1, pt2, color, thickness)

    layer1 = layers[0]
    layer2 = layers[1]
    pt1 = pt_arr_to_tuple(layer1[0])
    pt2 = pt_arr_to_tuple(layer2[0])
    cv2.line(img, pt1, pt2, color, thickness)
    pt1 = pt_arr_to_tuple(layer1[1])
    pt2 = pt_arr_to_tuple(layer2[1])
    cv2.line(img, pt1, pt2, color, thickness)
    pt1 = pt_arr_to_tuple(layer1[2])
    pt2 = pt_arr_to_tuple(layer2[2])
    cv2.line(img, pt1, pt2, color, thickness)
    pt1 = pt_arr_to_tuple(layer1[3])
    pt2 = pt_arr_to_tuple(layer2[3])
    cv2.line(img, pt1, pt2, color, thickness)


class CVThread(ProgramThread):
    def __init__(self):
        ProgramThread.__init__(self, self.run)
        self.scene_data = {}
        self.markers_found = 0
        self.quit = False

    def start(self):
        for cam in cam_manager.cameras:
            if cam.name not in self.scene_data:
                self.scene_data[cam.name] = {}
            if "world" not in self.scene_data:
                self.scene_data["world"] = {}
                self.scene_data["world"]["markers3d"] = np.array([])
            self.scene_data[cam.name]["pos"] = cam.data.pos.copy()
            self.scene_data[cam.name]["euler"] = cam.data.euler.copy()
        self.start_thread()

    def run(self):
        if cam_manager.available_cameras > 0:
            all_markers = []
            num_markers_found = np.array([])
            for cam in cam_manager.get_frames():
                # Prepare scene data for this camera
                if "markers3d" not in self.scene_data[cam.name]:
                    self.scene_data[cam.name]["markers3d"] = np.array([])

                # frame = resize_img(cam.frame, width=960)
                frame = cam.frame.copy()
                undistorted = cv2.undistort(frame, cam.data.mtx, cam.data.dist)
                img = undistorted.copy()

                found_markers, markers = marker_finder.find_markers(undistorted)
                if found_markers:
                    for marker in markers:
                        cv2.circle(img, (int(marker.x), int(marker.y)), 5, (0, 255, 0), -1)
                    self.scene_data[cam.name]["markers"] = markers
                    all_markers.append(markers)

                    num_markers_found = np.append(num_markers_found, np.array([len(markers)]))

                    # region Find marker intersection with z=0 plane
                    # Where the ray passing through the markers intersects with the z=0 plane
                    temp_markers_zplane = np.array([])

                    for marker in markers:
                        P = sfmutils.find_point_zplane(marker.x, marker.y, cam.data)

                        # changing values in another thread must be done immediately
                        # hence we store it in a temporary array before transferring to the public one
                        if temp_markers_zplane.shape[0] > 0:
                            temp_markers_zplane = np.vstack((temp_markers_zplane, P))
                        else:
                            temp_markers_zplane = np.array([P])

                        new_points = np.float32([[P[0], P[1], P[2]]])
                        screen_points, jac = cv2.projectPoints(new_points, cam.data.rvec, cam.data.tvec, cam.data.mtx,
                                                               None)
                        backprojection_error = np.linalg.norm(
                            np.array([screen_points[0].ravel()[0], screen_points[0].ravel()[1]]) - np.array(
                                [marker.x, marker.y]))
                        # print "Backprojection Error: {}".format(round(backprojection_error, 3))
                        center = (int(screen_points[0].ravel()[0]), int(screen_points[0].ravel()[1]))
                        cv2.circle(img, center, 5, (255, 0, 0), -1)

                    self.scene_data[cam.name]["markers_zplane"] = temp_markers_zplane
                    # endregion
                else:
                    num_markers_found = np.append(num_markers_found, np.array([0]))
                cam.imshow("Visualization", img)

            if len(num_markers_found) > 0:
                self.markers_found = np.min(num_markers_found)
                if len(all_markers) > 0 and self.markers_found > 0:
                    markers3d = sfmutils.triangulate(all_markers, cam_manager.cameras)
                    self.scene_data["world"]["markers3d"] = markers3d
                    self.markers_found = True
            else:
                self.markers_found = 0

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                self.quit = True

    def stop(self):
        self.stop_thread()


ipcam_url = "http://192.168.8.100:8080/"
cam_manager = CamManager([
    # IPCam(ipcam_url, data_filename="camera/LG-K8_scaled2.npz", name="LG_K8"),
    # ImgCam("img\\cam_rotated.jpg", data_filename="camera/LG-K8_scaled2.npz", name="CamRotated"),
    # ImgCam("img\\cam_right.jpg", data_filename="camera/LG-K8_scaled2.npz", name="CamRight"),
    # ImgCam("img\\cam_mid.jpg", data_filename="camera/LG-K8_scaled2.npz", name="CamMid"),
    # ImgCam("img\\cam_left.jpg", data_filename="camera/LG-K8_scaled2.npz", name="CamLeft"),
    # ImgCam("img\\cam_center.jpg", data_filename="camera/LG-K8_scaled2.npz", name="CamCenter")
    IPCam("http://192.168.8.100:8080/", data_filename="camera2/LG_K8.npz", name="LG_K8"),
    # IPCam("http://192.168.8.103:8080/", data_filename="camera2/samsung_galaxy.npz", name="samsung_galaxy")
])
cam_manager.start()

marker_finder = MarkerFinder()

cv_thread = CVThread()
cv_thread.start()

net_socket = NetworkSocket()
net_socket.listen()

while True:
    if net_socket.open:
        all_data = {
            "cameras": [],
            "markers3d": []
        }
        world_scale = 4 / 1000.0
        for cam in cam_manager.cameras:
            if cam.name in cv_thread.scene_data:
                scene_data = cv_thread.scene_data[cam.name]

                pos = cam.data.pos
                euler = cam.data.euler
                pos_world = pos * world_scale

                if cv_thread.markers_found > 0:
                    markers_zplane = scene_data["markers_zplane"].reshape(-1, 3)
                    markers_zplane_world = markers_zplane * world_scale
                else:
                    markers_zplane_world = np.array([])

                all_data["cameras"].append({
                    "name": cam.name,
                    "pos": np.array(pos_world).ravel().tolist(),
                    "euler": euler.ravel().tolist(),
                    "markers_zplane": markers_zplane_world.tolist()
                })
        if cv_thread.markers_found > 0:
            temp_markers3d = cv_thread.scene_data["world"]["markers3d"] * world_scale
            all_data["markers3d"] = temp_markers3d.tolist()
        net_socket.send(all_data, is_json=True)
    if cv_thread.quit:
        break

print "Terminating program..."
net_socket.close()
print "Network socket closed"
cam_manager.stop()
print "Camera stream stopped"
cv_thread.stop()
print "CV thread killed"
cv2.destroyAllWindows()
print "Windows destroyed"
