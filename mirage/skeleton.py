from dataclasses import dataclass
from filterpy.kalman import KalmanFilter


class JointDetection:
    def __init__(self, name: str, color, display):
        self.name: str = name
        self.color: tuple[int, int, int]
        self.display: bool
        # self.filter = KalmanFilter


KeypointMappings = {
    0: JointDetection("nose", (0, 0, 255), True),
    1: JointDetection("lefteye", (0, 0, 255), False),
    2: JointDetection("righteye", (0, 0, 255), False),
    3: JointDetection("leftear", (0, 0, 255), False),
    4: JointDetection("rightear", (0, 0, 255), False),
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


class SkeletonDetection:
    def __init__(self):
        pass
