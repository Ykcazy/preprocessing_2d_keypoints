# FILE: model_loader.py

from mmdet.apis import init_detector
from mmpose.apis import init_model
from mmengine.registry import init_default_scope

def initialize_models():
    """Initializes and returns the detection and pose estimation models."""
    print("Initializing models...")
    
    # Initialize detector
    init_default_scope('mmdet')
    det_model = init_detector(
        'mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py',
        'mmdetection/checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
        device='cuda:0'
    )

    # Initialize pose estimator
    init_default_scope('mmpose')
    pose_model = init_model(
        'mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-384x288.py',
        'mmpose/checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth',
        device='cuda:0'
    )
       
    print("✅ Models initialized successfully.")
    return det_model, pose_model