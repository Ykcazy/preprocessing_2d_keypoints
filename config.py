# FILE: config.py

class Config:
    # --- Video & Display ---
    VIDEO_PATHS = ["dataset - extraction\elbow_extension\correct\cam0\elbow_extension_correct_cam0_1.mp4", "dataset - extraction\elbow_extension\correct\cam1\elbow_extension_correct_cam1_1.mp4"]
    OUTPUT_VIDEO_PATH = "output_visualization.mp4"
    MAX_DISPLAY_WIDTH = 1920
    MAX_DISPLAY_HEIGHT = 1080

    # --- Detection, Tracking, and Pose ---
    MAX_PEOPLE_PER_FRAME = 2
    DET_CONF_THRESHOLD = 0.4
    POSE_CONFIDENCE_THRESHOLD = 0.3
    HISTORY_SIZE = 27
    BBOX_PADDING = 20
 

    # --- ByteTrack Parameters ---
    TRACK_THRESH = 0.4
    TRACK_BUFFER = 100
    MATCH_THRESH = 0.8

    # --- Global Re-Identification (Re-ID) ---
    REID_SIM_THRESH = 0.5
    REID_EMB_SIZE = (128, 64)
    REID_UPDATE_ALPHA = 0.15


# --- SKELETON INFO (COCO - for 2D Pose) ---
# Using the official dataset_info structure for robustness
dataset_info = dict(
    dataset_name='coco',
    keypoint_info={
        0: dict(name='nose', id=0, color=[51, 153, 255]),
        1: dict(name='left_eye', id=1, color=[51, 153, 255]),
        2: dict(name='right_eye', id=2, color=[51, 153, 255]),
        3: dict(name='left_ear', id=3, color=[51, 153, 255]),
        4: dict(name='right_ear', id=4, color=[51, 153, 255]),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0]),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0]),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0]),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0]),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0]),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0]),
        11: dict(name='left_hip', id=11, color=[0, 255, 0]),
        12: dict(name='right_hip', id=12, color=[255, 128, 0]),
        13: dict(name='left_knee', id=13, color=[0, 255, 0]),
        14: dict(name='right_knee', id=14, color=[255, 128, 0]),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0]),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0])
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee')),
        1: dict(link=('left_knee', 'left_hip')),
        2: dict(link=('right_ankle', 'right_knee')),
        3: dict(link=('right_knee', 'right_hip')),
        4: dict(link=('left_hip', 'right_hip')),
        5: dict(link=('left_shoulder', 'left_hip')),
        6: dict(link=('right_shoulder', 'right_hip')),
        7: dict(link=('left_shoulder', 'right_shoulder')),
        8: dict(link=('left_shoulder', 'left_elbow')),
        9: dict(link=('right_shoulder', 'right_elbow')),
        10: dict(link=('left_elbow', 'left_wrist')),
        11: dict(link=('right_elbow', 'right_wrist')),
        12: dict(link=('left_eye', 'right_eye')),
        13: dict(link=('nose', 'left_eye')),
        14: dict(link=('nose', 'right_eye')),
        15: dict(link=('left_eye', 'left_ear')),
        16: dict(link=('right_eye', 'right_ear')),
    }
)

# Create lookup dictionaries and skeleton links from the dataset_info
kpt_name_to_id = {v['name']: k for k, v in dataset_info['keypoint_info'].items()}
skeleton_links = [
    (kpt_name_to_id[info['link'][0]], kpt_name_to_id[info['link'][1]])
    for _, info in dataset_info['skeleton_info'].items()
]


# --- SKELETON INFO (Human3.6M - for 3D Pose) ---
kpt_name_to_id_h36m = {
    'pelvis': 0, 'right_hip': 1, 'right_knee': 2, 'right_ankle': 3,
    'left_hip': 4, 'left_knee': 5, 'left_ankle': 6, 'spine': 7,
    'thorax': 8, 'neck': 9, 'head': 10, 'left_shoulder': 11,
    'left_elbow': 12, 'left_wrist': 13, 'right_shoulder': 14,
    'right_elbow': 15, 'right_wrist': 16
}
skeleton_links_h36m = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
    (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
]
