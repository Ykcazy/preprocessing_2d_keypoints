import cv2
import numpy as np
import torch
from types import SimpleNamespace
from typing import List, Dict, Any
import os

# mmdet / mpose imports
from mmdet.apis import inference_detector
from mmpose.apis import inference_topdown
from mmengine.registry import init_default_scope

# ByteTrack tracker import
from yolox.tracker.byte_tracker import BYTETracker

# Local imports
from config import Config, skeleton_links
from model_loader import initialize_models
from utils import (create_tiled_display, get_pose_embedding,
                   get_keypoint_distance, refine_bbox_with_keypoints)

# =================================================================================
# MAIN VISUALIZATION FUNCTION (BACKGROUND / HEADLESS)
# =================================================================================
def run_visualization():
    print("🚀 Starting background processing. A video will be generated without a live preview.")

    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = float(boxAArea + boxBArea - interArea)
        if denominator == 0: return 0.0
        iou = interArea / denominator
        return iou

    # 1. --- INITIALIZATION ---
    det_model, pose2d_model = initialize_models()
    caps = [cv2.VideoCapture(p) for p in Config.VIDEO_PATHS]
    if not all(c.isOpened() for c in caps):
        print(f"Error: Could not open one or more videos in {Config.VIDEO_PATHS}")
        return

    tracker_args = SimpleNamespace(track_thresh=Config.TRACK_THRESH, match_thresh=Config.MATCH_THRESH, track_buffer=Config.TRACK_BUFFER, mot20=False)
    trackers = [BYTETracker(tracker_args, frame_rate=30) for _ in Config.VIDEO_PATHS]

    role_dbs = [{r: {'embedding': None, 'last_center': None} for r in ["Patient", "Physical Therapist"]} for _ in Config.VIDEO_PATHS]
    
    video_writer = None
    frame_idx = 0
    all_online_targets_prev = [[] for _ in Config.VIDEO_PATHS]

    while True:
        frames = [cap.read()[1] for cap in caps]
        if any(f is None for f in frames): break

        all_views_data, all_online_targets = [], []

        for i, frame in enumerate(frames):
            frame_h, frame_w, _ = frame.shape
            
            # --- STEP 1: Always run the detector ---
            init_default_scope('mmdet')
            det_result = inference_detector(det_model, frame)
            pred_instances = det_result.pred_instances.cpu()
            person_mask = (pred_instances.labels == 0) & (pred_instances.scores > Config.DET_CONF_THRESHOLD)
            
            scores = pred_instances.scores[person_mask]
            sorted_indices = scores.argsort(descending=True)
            top_indices = sorted_indices[:Config.MAX_PEOPLE_PER_FRAME]
            
            if len(top_indices) > 0:
                final_person_indices = person_mask.nonzero(as_tuple=True)[0][top_indices]
                final_mask = torch.zeros_like(person_mask, dtype=torch.bool)
                final_mask[final_person_indices] = True
            else:
                final_mask = torch.zeros_like(person_mask, dtype=torch.bool)

            detector_bboxes = pred_instances.bboxes[final_mask].numpy()
            scores_for_tracker = pred_instances.scores[final_mask].numpy()

            # --- Hybrid Box logic for "padding"/stability ---
            hybrid_bboxes_for_pose = []
            if len(all_online_targets_prev[i]) > 0 and len(detector_bboxes) > 0:
                matched_det_indices = set()
                for track in all_online_targets_prev[i]:
                    best_iou, best_det_idx = 0, -1
                    for det_idx, det_box in enumerate(detector_bboxes):
                        if det_idx in matched_det_indices: continue
                        iou = calculate_iou(track.tlbr, det_box)
                        if iou > best_iou:
                            best_iou, best_det_idx = iou, det_idx
                    
                    if best_iou > 0.3:
                        matched_det_indices.add(best_det_idx)
                        det_box = detector_bboxes[best_det_idx]
                        det_center_x = (det_box[0] + det_box[2]) / 2
                        det_center_y = (det_box[1] + det_box[3]) / 2
                        track_w = track.tlbr[2] - track.tlbr[0]
                        track_h = track.tlbr[3] - track.tlbr[1]
                        x1 = max(0, det_center_x - track_w / 2)
                        y1 = max(0, det_center_y - track_h / 2)
                        x2 = min(frame_w, det_center_x + track_w / 2)
                        y2 = min(frame_h, det_center_y + track_h / 2)
                        hybrid_bboxes_for_pose.append([x1, y1, x2, y2])
                
                for det_idx, det_box in enumerate(detector_bboxes):
                    if det_idx not in matched_det_indices:
                        hybrid_bboxes_for_pose.append(det_box)
                
                bboxes_for_pose = np.array(hybrid_bboxes_for_pose)
            else:
                bboxes_for_pose = detector_bboxes

            pose_results_2d = []
            if len(bboxes_for_pose) > 0:
                init_default_scope('mmpose')
                pose_results_2d = inference_topdown(pose2d_model, frame, bboxes_for_pose)

            refined_dets_for_tracker, view_people = [], []
            if len(pose_results_2d) > 0:
                for bbox_idx, pose_res in enumerate(pose_results_2d):
                    kpts = pose_res.pred_instances.keypoints[0]
                    scores = pose_res.pred_instances.keypoint_scores[0]
                    refined_box = refine_bbox_with_keypoints(kpts)
                    
                    if refined_box is not None and np.isfinite(refined_box).all():
                        score_to_use = 0.99
                        if bbox_idx < len(scores_for_tracker):
                            score_to_use = scores_for_tracker[bbox_idx]

                        new_det = np.hstack([refined_box, np.array([score_to_use])])
                        refined_dets_for_tracker.append(new_det)
                        
                        person_data = {'bbox': refined_box, 'embedding': get_pose_embedding(kpts),
                                       'keypoints': kpts, 'scores': scores, 'role': None}
                        view_people.append(person_data)
            
            dets_array = np.array(refined_dets_for_tracker)
            online_targets = trackers[i].update(dets_array, frame.shape, frame.shape)
            all_online_targets.append(online_targets)
            all_views_data.append(view_people)

        all_online_targets_prev = all_online_targets
        
        # --- (SMART ROLE ASSIGNMENT & ID MATCHING IS UNCHANGED) ---
        for i, view_people in enumerate(all_views_data):
            role_db = role_dbs[i]; unassigned, assigned_roles = [], set()
            online_tracks = all_online_targets[i]; id_offset = i * 1000

            for person in view_people:
                best_iou, best_track_id = 0, None
                for track in online_tracks:
                    iou = calculate_iou(person['bbox'], track.tlbr)
                    if iou > best_iou: best_iou, best_track_id = iou, track.track_id
                if best_iou > 0.3: person['track_id'] = best_track_id + id_offset
            
            for p_idx, person in enumerate(view_people):
                if 'track_id' not in person: person['track_id'] = f'PENDING-{p_idx + id_offset}'

            for p in view_people:
                if p['embedding'] is None: unassigned.append(p); continue
                best_role, best_sim = None, -1
                for r, d in role_db.items():
                    if d['embedding'] is not None:
                        p_emb, d_emb = p['embedding'], d['embedding']
                        norm_p, norm_d = np.linalg.norm(p_emb), np.linalg.norm(d_emb)
                        if norm_p > 0 and norm_d > 0:
                            sim = np.dot(p_emb, d_emb) / (norm_p * norm_d)
                            if sim > best_sim: best_sim, best_role = sim, r
                if best_sim >= Config.REID_SIM_THRESH and best_role not in assigned_roles:
                    p['role'] = best_role; assigned_roles.add(best_role)
                else: unassigned.append(p)
            
            if unassigned:
                available = [r for r in ["Patient", "Physical Therapist"] if r not in assigned_roles]
                if role_db["Patient"]['embedding'] is None and len(unassigned) >= 2 and "Patient" in available:
                    unassigned.sort(key=lambda p: (p['bbox'][2]-p['bbox'][0])/(p['bbox'][3]-p['bbox'][1]), reverse=True)
                    patient = unassigned.pop(0); patient['role']="Patient"; assigned_roles.add("Patient"); available.remove("Patient")
                    if unassigned and "Physical Therapist" in available:
                        therapist = unassigned.pop(0); therapist['role']="Physical Therapist"; assigned_roles.add("Physical Therapist"); available.remove("Physical Therapist")
                else:
                    for p in unassigned:
                        if not available: break
                        p_center = np.array([(p['bbox'][0]+p['bbox'][2])/2, (p['bbox'][1]+p['bbox'][3])/2])
                        best_role, min_dist = None, float('inf')
                        for role in available:
                            if role_db[role]['last_center'] is not None:
                                dist = np.linalg.norm(p_center - role_db[role]['last_center'])
                                if dist < min_dist: min_dist, best_role = dist, role
                        if best_role and min_dist < 250:
                            p['role'] = best_role; available.remove(best_role); assigned_roles.add(best_role)

        for i, view_people in enumerate(all_views_data):
            role_db = role_dbs[i]
            for p in view_people:
                if p.get('role') and p.get('embedding') is not None:
                    role, alpha = p['role'], Config.REID_UPDATE_ALPHA
                    center = np.array([(p['bbox'][0]+p['bbox'][2])/2, (p['bbox'][1]+p['bbox'][3])/2])
                    if role_db[role]['embedding'] is None:
                        role_db[role]['embedding'], role_db[role]['last_center'] = p['embedding'], center
                    else:
                        role_db[role]['embedding'] = alpha*p['embedding'] + (1-alpha)*role_db[role]['embedding']
                        role_db[role]['last_center'] = alpha*center + (1-alpha)*role_db[role]['last_center']
        
        # --- FRAME DRAWING ---
        processed_frames = []
        for i, frame in enumerate(frames):
            for person in all_views_data[i]:
                person_color = (0,255,0) if person.get('role')=="Physical Therapist" else ((255,0,0) if person.get('role')=="Patient" else (0,0,255))
                x1, y1, x2, y2 = map(int, person['bbox'])
                label = f"{person.get('role', '...')} (ID: {person.get('track_id', 'N/A')})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, person_color, 2, cv2.LINE_AA)
                
                if person.get('keypoints') is not None:
                    for p1_idx, p2_idx in skeleton_links:
                        if person['scores'][p1_idx] > Config.POSE_CONFIDENCE_THRESHOLD and person['scores'][p2_idx] > Config.POSE_CONFIDENCE_THRESHOLD:
                            p1 = tuple(map(int, person['keypoints'][p1_idx]))
                            p2 = tuple(map(int, person['keypoints'][p2_idx]))
                            cv2.line(frame, p1, p2, (255,255,255), 2)
                    for (x,y), s in zip(person['keypoints'], person['scores']):
                        if s > Config.POSE_CONFIDENCE_THRESHOLD:
                            cv2.circle(frame, (int(x), int(y)), 4, person_color, -1)
            
            processed_frames.append(frame)

        display_img = create_tiled_display(processed_frames, Config.MAX_DISPLAY_WIDTH, Config.MAX_DISPLAY_HEIGHT)
        
        # --- VIDEO WRITING ---
        if video_writer is None and display_img.size > 0:
            h, w, _ = display_img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
            video_writer = cv2.VideoWriter(Config.OUTPUT_VIDEO_PATH, fourcc, fps, (w,h))
        
        if video_writer is not None:
            video_writer.write(display_img)
            
        # --- REMOVED: The following lines for displaying the window are now gone ---
        # cv2.imshow("Patient and Therapist Tracking (2D)", display_img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_idx += 1
        # Add a print statement to show progress in the terminal
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")


    if video_writer:
        video_writer.release()
    for cap in caps:
        cap.release()
    
    # --- REMOVED: This is no longer needed as no windows were created ---
    # cv2.destroyAllWindows()
    
    print(f"\n✅ Processing finished. Video saved to {Config.OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    run_visualization()