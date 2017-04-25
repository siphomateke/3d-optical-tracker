import numpy as np
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 2000

lk_params = dict(winSize=(21, 21),
                 # maxLevel = 3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def feature_tracking(prev_img, next_img, prev_pts):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, None, **lk_params)  # shape: [k,2] [k,1] [k,1]

    if st is not None:
        st = st.reshape(st.shape[0])
        kp1 = prev_pts[st == 1]
        kp2 = kp2[st == 1]

        return True, kp1, kp2
    else:
        return False, None, None


class PinholeCamera:
    def __init__(self, filename):
        camera_settings = np.load(filename)
        self.mtx = camera_settings["mtx"]
        self.fx = self.mtx[0][0]
        self.fy = self.mtx[1][1]
        self.cx = self.mtx[0][2]
        self.cy = self.mtx[1][2]
        self.dist = camera_settings["dist"]


class VisualOdometry:
    def __init__(self, cam):
        self.cam = cam
        self.last_frame = None
        self.new_frame = None
        self.frame_stage = 0
        self.detector = cv2.FastFeatureDetector_create(2, True)
        self.points = None
        self.prev_points = None
        self.cur_R = None
        self.cur_t = None
        self.optical_flow = None
        self.delta = None

    def undistort(self, img):
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.cam.mtx, self.cam.dist, (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(img, self.cam.mtx, self.cam.dist, None, new_camera_mtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        return dst

    def detect_features(self, img):
        keypoints = self.detector.detect(img)
        # Convert keypoints to 2d points
        keypoints = np.array([x.pt for x in keypoints], dtype=np.float32)
        return keypoints

    def process_first_frame(self):
        self.prev_points = self.detect_features(self.new_frame)
        if len(self.prev_points) > 0:
            self.frame_stage = STAGE_SECOND_FRAME

    def process_second_frame(self):
        ret, self.prev_points, self.points = feature_tracking(self.last_frame, self.new_frame, self.prev_points)
        if ret and self.points is not None and len(self.points) > 0:
            # Determine best points and then the essential matrix
            E, mask = cv2.findEssentialMat(self.points, self.prev_points, self.cam.mtx, method=cv2.RANSAC,
                                           prob=0.999, threshold=1.0)
            try:
                # Rotation and translation matrices
                _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.points, self.prev_points, self.cam.mtx)

                self.frame_stage = STAGE_DEFAULT_FRAME
                self.prev_points = self.points
            except cv2.error as e:
                print e.message

    def process_frame(self):
        ret, self.prev_points, self.points = feature_tracking(self.last_frame, self.new_frame, self.prev_points)
        if ret and self.points is not None and len(self.points) > 0:
            # Determine best points and then the essential matrix
            E, mask = cv2.findEssentialMat(self.points, self.prev_points, self.cam.mtx, method=cv2.RANSAC,
                                           prob=0.999, threshold=1.0)
            # Rotation and translation matrices
            try:
                _, R, t, mask = cv2.recoverPose(E, self.points, self.prev_points, self.cam.mtx)
                # absolute_scale = self.getAbsoluteScale(frame_id)

                for (pt1, pt2) in zip(self.points, self.prev_points):
                    self.optical_flow = cv2.line(self.optical_flow, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1)

                absolute_scale = self.delta * 10
                if absolute_scale > 0.1:
                    self.cur_t += absolute_scale * self.cur_R.dot(t)
                    self.cur_R = R.dot(self.cur_R)
                if self.prev_points.shape[0] < kMinNumFeature:
                    self.points = self.detect_features(self.new_frame)
                    self.optical_flow = None
                self.prev_points = self.points
            except cv2.error as e:
                print e.message

    def update(self, img, delta):
        self.delta = delta
        self.new_frame = img
        self.new_frame = self.undistort(self.new_frame)

        if self.frame_stage == STAGE_DEFAULT_FRAME:
            if len(self.prev_points) > 0:
                self.process_frame()
            else:
                self.frame_stage = 0
        elif self.frame_stage == STAGE_SECOND_FRAME:
            if len(self.prev_points) > 0:
                self.process_second_frame()
            else:
                self.frame_stage = 0
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.process_first_frame()

        self.last_frame = self.new_frame
