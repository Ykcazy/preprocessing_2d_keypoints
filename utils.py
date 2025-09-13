# FILE: utils.py

import cv2
import numpy as np
from typing import Union, List

# Local imports
from config import Config

def create_tiled_display(frames: List[np.ndarray], max_w: int, max_h: int) -> np.ndarray:
    if not frames: return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    num_frames = len(frames)
    if num_frames == 0 or frames[0] is None or frames[0].shape[0] == 0 or frames[0].shape[1] == 0:
        return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    frame_h, frame_w, _ = frames[0].shape
    aspect_ratio = frame_h / frame_w
    target_w = int(min(max_w / num_frames, max_h / aspect_ratio))
    target_h = int(target_w * aspect_ratio)
    if target_w == 0 or target_h == 0: return np.zeros((100, 100, 3), dtype=np.uint8)
    resized_frames = [cv2.resize(f, (target_w, target_h)) for f in frames if f is not None]
    if not resized_frames: return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    return np.hstack(resized_frames)

def get_embedding(crop: np.ndarray) -> Union[np.ndarray, None]:
    if crop is None or crop.size == 0: return None
    try:
        resized = cv2.resize(crop, (Config.REID_EMB_SIZE[1], Config.REID_EMB_SIZE[0]))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else None
    except cv2.error:
        return None

def get_pose_embedding(keypoints: np.ndarray) -> Union[np.ndarray, None]:
    """
    Creates a normalized feature vector from 2D keypoints for robust re-identification.
    """
    if keypoints is None or keypoints.shape[0] < 17:
        return None

    # Use torso to normalize: mid-point of shoulders to mid-point of hips
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]

    # Check for invalid keypoints (which are often [0, 0])
    if np.all(left_hip == 0) or np.all(right_hip == 0) or np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return None
        
    origin = (left_hip + right_hip) / 2
    shoulder_center = (left_shoulder + right_shoulder) / 2
    torso_height = np.linalg.norm(shoulder_center - origin)
    
    if torso_height < 1e-4: 
        return None

    normalized_keypoints = (keypoints - origin) / torso_height
    return normalized_keypoints.flatten()

def get_keypoint_distance(
    kpts1: np.ndarray, scores1: np.ndarray, kpt1_name: str,
    kpts2: np.ndarray, scores2: np.ndarray, kpt2_name: str
) -> float:
    from config import kpt_name_to_id, Config
    try:
        kpt1_idx = kpt_name_to_id[kpt1_name]
        kpt2_idx = kpt_name_to_id[kpt2_name]
    except KeyError:
        return -1.0
    if scores1[kpt1_idx] < Config.POSE_CONFIDENCE_THRESHOLD or scores2[kpt2_idx] < Config.POSE_CONFIDENCE_THRESHOLD:
        return -1.0
    p1 = kpts1[kpt1_idx]
    p2 = kpts2[kpt2_idx]
    distance = np.linalg.norm(p1 - p2)
    return distance

def refine_bbox_with_keypoints(keypoints: np.ndarray, previous_bbox: np.ndarray = None) -> np.ndarray:
    """
    Refines a bounding box to tightly enclose valid pose keypoints.
    """
    if keypoints is None or keypoints.size == 0:
        return None

    valid_kpts = keypoints[np.any(keypoints != 0, axis=1)]
    if valid_kpts.shape[0] == 0:
        return None

    min_x, max_x = np.min(valid_kpts[:, 0]), np.max(valid_kpts[:, 0])
    min_y, max_y = np.min(valid_kpts[:, 1]), np.max(valid_kpts[:, 1])
    
    pad = Config.BBOX_PADDING
    refined_bbox = np.array([
        min_x - pad,
        min_y - pad,
        max_x + pad,
        max_y + pad
    ])
    
    return refined_bbox

def calculate_torso_angle(keypoints: np.ndarray) -> float:
    """Calculates the angle of the torso relative to the vertical axis."""
    if keypoints is None:
        return 90.0
    
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]

    if np.all(left_shoulder == 0) or np.all(right_shoulder == 0) or \
       np.all(left_hip == 0) or np.all(right_hip == 0):
        return 90.0

    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2
    
    torso_vector = mid_shoulder - mid_hip
    vertical_vector = np.array([0, -1])
    
    unit_torso = torso_vector / np.linalg.norm(torso_vector)
    dot_product = np.dot(unit_torso, vertical_vector)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return np.degrees(angle_rad)