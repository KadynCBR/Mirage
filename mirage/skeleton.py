from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
import numpy as np
from cv2.typing import MatLike
from copy import deepcopy


class JointDetection:
    def __init__(self, name: str, color, display):
        self.name: str = name
        self.color: tuple[int, int, int] = color
        self.display: bool = display
        self.filter = KalmanFilter(dim_x=4, dim_z=2)  # 4 dimension, x pos, x vel, y pos, y vel. 2 measurement, xy
        self._filter_initialized = False
        self.current_xy: list[float, float] = [0.0, 0.0]
        self.saved_states: list[np.array] = []
        self.confidence = 0

    def initialize_filter_position(self, x_pos: float, y_pos: float, dt_: float):
        # initial state
        self.filter.x = np.array([x_pos, 0, y_pos, 0])
        # measurement function
        self.filter.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        # intitial state covariance matrix
        self.filter.P *= 0.1
        # state transition matrix
        self.filter.F = np.array(
            [
                [1.0, dt_, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt_],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # measurement noise covariance matrix
        # Uncertainty in measurements (trust predictions more?)
        self.filter.R = np.array(
            [
                [0.05, 0.0],
                [0.0, 0.05],
            ]
        )
        # process noise covariance matrix
        # Uncertainty in process model (trust measurement more?)
        self.filter.Q *= 0.01
        self._filter_initialized = True

    def update(self, detection: list[float, float, float], dt: float = 0.05):
        self.current_xy = [detection[1], detection[0]]
        self.confidence: float = detection[2]
        if not self._filter_initialized:
            self.initialize_filter_position(detection[1], detection[0], dt)
        self.filter.predict()
        self.filter.update(np.array(self.current_xy))
        self.estimate = self.filter.x
        self.saved_states.append(self.estimate)

    def get_estimate_at(self, n: int):
        return self.saved_states[n]


class JointDetection3D:
    def __init__(self, name: str, color, display):
        self.name: str = name
        self.color: tuple[int, int, int] = color
        self.display: bool = display
        self.filter = KalmanFilter(
            dim_x=6, dim_z=3
        )  # 6 dimension, x pos, x vel, y pos, y vel z pos z vel. 3 measurement, xyz
        self._filter_initialized = False
        self.current_xyz: list[float, float, float] = [0.0, 0.0, 0.0]
        self.saved_states: list[np.array] = []

    def initialize_filter_position(self, x_pos: float, y_pos: float, z_pos: float, dt_: float):
        # initial state
        self.filter.x = np.array([x_pos, 0, y_pos, 0, z_pos, 0])
        # measurement function
        self.filter.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        # intitial state covariance matrix
        self.filter.P *= 0.1
        # state transition matrix
        self.filter.F = np.array(
            [
                [1.0, dt_, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt_, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, dt_],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        # measurement noise covariance matrix
        # Uncertainty in measurements (trust predictions more?)
        self.filter.R = np.array(
            [
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05],
            ]
        )
        # process noise covariance matrix
        # Uncertainty in process model (trust measurement more?)
        self.filter.Q *= 0.01
        self._filter_initialized = True

    def update(self, detection: list[float, float, float], dt: float = 0.05):
        self.current_xyz = [detection[0], detection[1], detection[2]]
        if not self._filter_initialized:
            self.initialize_filter_position(detection[0], detection[1], detection[2], dt)
        self.filter.predict()
        self.filter.update(np.array(self.current_xyz))
        self.estimate = self.filter.x
        self.saved_states.append(self.estimate)

    def get_estimate_at(self, n: int):
        return self.saved_states[n]


KeypointMappings = {
    0: JointDetection("nose", (0, 0, 255), True),
    1: JointDetection("lefteye", (0, 0, 255), True),  # face
    2: JointDetection("righteye", (0, 0, 255), True),  # face
    3: JointDetection("leftear", (0, 0, 255), True),  # face
    4: JointDetection("rightear", (0, 0, 255), True),  # face
    5: JointDetection("leftshoulder", (0, 255, 255), True),
    6: JointDetection("rightshoulder", (0, 255, 255), True),
    7: JointDetection("leftelbow", (0, 255, 255), True),
    8: JointDetection("rightelbow", (0, 255, 255), True),
    9: JointDetection("leftwrist", (100, 100, 255), True),
    10: JointDetection("rightwrist", (100, 100, 255), True),
    11: JointDetection("lefthip", (100, 0, 100), True),
    12: JointDetection("righthip", (100, 0, 100), True),
    13: JointDetection("leftknee", (100, 0, 100), True),
    14: JointDetection("rightknee", (100, 0, 100), True),
    15: JointDetection("leftankle", (100, 0, 100), True),
    16: JointDetection("rightankle", (100, 0, 100), True),
}

KeypointMappings3D = {
    0: JointDetection3D("nose", (0, 0, 255), True),
    1: JointDetection3D("lefteye", (0, 0, 255), True),  # face
    2: JointDetection3D("righteye", (0, 0, 255), True),  # face
    3: JointDetection3D("leftear", (0, 0, 255), True),  # face
    4: JointDetection3D("rightear", (0, 0, 255), True),  # face
    5: JointDetection3D("leftshoulder", (0, 255, 255), True),
    6: JointDetection3D("rightshoulder", (0, 255, 255), True),
    7: JointDetection3D("leftelbow", (0, 255, 255), True),
    8: JointDetection3D("rightelbow", (0, 255, 255), True),
    9: JointDetection3D("leftwrist", (100, 100, 255), True),
    10: JointDetection3D("rightwrist", (100, 100, 255), True),
    11: JointDetection3D("lefthip", (100, 0, 100), True),
    12: JointDetection3D("righthip", (100, 0, 100), True),
    13: JointDetection3D("leftknee", (100, 0, 100), True),
    14: JointDetection3D("rightknee", (100, 0, 100), True),
    15: JointDetection3D("leftankle", (100, 0, 100), True),
    16: JointDetection3D("rightankle", (100, 0, 100), True),
}


class SkeletonDetection:
    def __init__(self):
        self.joints: dict[int, JointDetection] = deepcopy(KeypointMappings)

    def update_predictions(self, keypoints: MatLike, dt: float):
        for i, keypoint in enumerate(keypoints):
            self.joints[i].update(keypoint, dt)


class SkeletonDetection3D:
    def __init__(self):
        self.joints: dict[int, JointDetection3D] = deepcopy(KeypointMappings3D)

    def update_predictions(self, homogeneous_skeleton_dict: dict, dt: float):
        for i, homogeneous_coordinant in homogeneous_skeleton_dict.items():
            self.joints[i].update(homogeneous_coordinant, dt)
