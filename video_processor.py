# FILE: video_processor.py (Final, with Automatic Codec Fallback)

import cv2
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from mmdet.apis import inference_detector
from mmpose.apis import inference_topdown
from mmengine.registry import init_default_scope
from yolox.tracker.byte_tracker import BYTETracker

from config import Config, skeleton_links
from model_loader import initialize_models
from utils import (create_tiled_display, get_pose_embedding,
                   get_keypoint_distance, refine_bbox_with_keypoints)

def process_videos(video_paths, exercise_label, correctness_label, output_csv_path=None, output_video_path=None, st_progress_bar=None, st_status_text=None):
    if not output_csv_path and not output_video_path:
        print("⚠️ Warning: No output path provided. Function will have no effect.")
        return

    print("🚀 Starting unified video processing...")

    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = float(boxAArea + boxBArea - interArea)
        if denominator == 0: return 0.0
        return interArea / denominator

    det_model, pose2d_model = initialize_models()
    caps = [cv2.VideoCapture(p) for p in video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_paths[i]}")

    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    tracker_args = SimpleNamespace(track_thresh=Config.TRACK_THRESH, match_thresh=Config.MATCH_THRESH, track_buffer=Config.TRACK_BUFFER, mot20=False)
    trackers = [BYTETracker(tracker_args, frame_rate=30) for _ in video_paths]
    role_dbs = [{r: {'embedding': None, 'last_center': None} for r in ["Patient", "Physical Therapist"]} for _ in video_paths]

    all_frames_data = []
    video_writer = None
    frame_idx = 0
    all_online_targets_prev = [[] for _ in video_paths]

    while True:
        frames = []
        all_videos_read_successfully = True
        for cap in caps:
            success, frame = cap.read()
            if not success:
                all_videos_read_successfully = False
                break
            frames.append(frame)
        if not all_videos_read_successfully:
            break

        # --- (Core processing logic is unchanged) ---
        all_views_data, all_online_targets = [], []
        for i, frame in enumerate(frames):
            frame_h, frame_w, _ = frame.shape
            init_default_scope('mmdet')
            det_result = inference_detector(det_model, frame)
            pred_instances = det_result.pred_instances.cpu()
            person_mask = (pred_instances.labels == 0) & (pred_instances.scores > Config.DET_CONF_THRESHOLD)
            scores = pred_instances.scores[person_mask]
            sorted_indices = scores.argsort(descending=True)
            top_indices = sorted_indices[:Config.MAX_PEOPLE_PER_FRAME]
            final_mask = torch.zeros_like(person_mask, dtype=torch.bool)
            if len(top_indices) > 0:
                final_person_indices = person_mask.nonzero(as_tuple=True)[0][top_indices]
                final_mask[final_person_indices] = True
            detector_bboxes = pred_instances.bboxes[final_mask].numpy()
            scores_for_tracker = pred_instances.scores[final_mask].numpy()
            hybrid_bboxes_for_pose = []
            if len(all_online_targets_prev[i]) > 0 and len(detector_bboxes) > 0:
                matched_det_indices = set()
                for track in all_online_targets_prev[i]:
                    best_iou, best_det_idx = 0, -1
                    for det_idx, det_box in enumerate(detector_bboxes):
                        if det_idx in matched_det_indices: continue
                        iou = calculate_iou(track.tlbr, det_box)
                        if iou > best_iou: best_iou, best_det_idx = iou, det_idx
                    if best_iou > 0.3:
                        matched_det_indices.add(best_det_idx)
                        det_box = detector_bboxes[best_det_idx]
                        det_center_x, det_center_y = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
                        track_w, track_h = track.tlbr[2] - track.tlbr[0], track.tlbr[3] - track.tlbr[1]
                        x1, y1 = max(0, det_center_x - track_w / 2), max(0, det_center_y - track_h / 2)
                        x2, y2 = min(frame_w, det_center_x + track_w / 2), min(frame_h, det_center_y + track_h / 2)
                        hybrid_bboxes_for_pose.append([x1, y1, x2, y2])
                for det_idx, det_box in enumerate(detector_bboxes):
                    if det_idx not in matched_det_indices: hybrid_bboxes_for_pose.append(det_box)
                bboxes_for_pose = np.array(hybrid_bboxes_for_pose)
            else:
                bboxes_for_pose = detector_bboxes
            init_default_scope('mmpose')
            pose_results_2d = inference_topdown(pose2d_model, frame, bboxes_for_pose) if len(bboxes_for_pose) > 0 else []
            refined_dets_for_tracker, view_people = [], []
            if len(pose_results_2d) > 0:
                for bbox_idx, pose_res in enumerate(pose_results_2d):
                    kpts, scores = pose_res.pred_instances.keypoints[0], pose_res.pred_instances.keypoint_scores[0]
                    refined_box = refine_bbox_with_keypoints(kpts)
                    if refined_box is not None and np.isfinite(refined_box).all():
                        score_to_use = scores_for_tracker[bbox_idx] if bbox_idx < len(scores_for_tracker) else 0.99
                        refined_dets_for_tracker.append(np.hstack([refined_box, np.array([score_to_use])]))
                        view_people.append({'bbox': refined_box, 'embedding': get_pose_embedding(kpts), 'keypoints': kpts, 'scores': scores, 'role': None})
            online_targets = trackers[i].update(np.array(refined_dets_for_tracker), frame.shape, frame.shape)
            all_online_targets.append(online_targets)
            all_views_data.append(view_people)
        all_online_targets_prev = all_online_targets
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
        if output_csv_path:
            therapist_2d = next((p for p in all_views_data[0] if p.get('role') == 'Physical Therapist'), None)
            patient_2d = next((p for p in all_views_data[1] if p.get('role') == 'Patient'), None)
            if therapist_2d and patient_2d:
                frame_data = {'frame': frame_idx}
                h, w, _ = frames[0].shape
                therapist_prox_view = next((p for p in all_views_data[0] if p.get('role') == 'Physical Therapist'), None)
                patient_prox_view = next((p for p in all_views_data[0] if p.get('role') == 'Patient'), None)
                if therapist_prox_view and patient_prox_view and therapist_prox_view.get('keypoints') is not None and patient_prox_view.get('keypoints') is not None:
                    for hand in ['left_wrist', 'right_wrist']:
                        for arm_part in ['left_wrist', 'left_elbow', 'right_wrist', 'right_elbow']:
                            dist = get_keypoint_distance(therapist_prox_view['keypoints'], therapist_prox_view['scores'], hand, patient_prox_view['keypoints'], patient_prox_view['scores'], arm_part)
                            frame_data[f"dist_{hand}_to_{arm_part}_norm"] = dist / np.sqrt(h**2 + w**2) if dist != -1 else -1
                therapist_kpts = therapist_2d['keypoints']
                patient_kpts = patient_2d['keypoints']
                for i in range(17):
                    for j, axis in enumerate('xy'):
                        frame_data[f"pt_kpt{i}_{axis}"] = therapist_kpts[i][j]
                        frame_data[f"patient_kpt{i}_{axis}"] = patient_kpts[i][j]
                all_frames_data.append(frame_data)
        if output_video_path:
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
                                p1 = tuple(map(int, person['keypoints'][p1_idx])); p2 = tuple(map(int, person['keypoints'][p2_idx]))
                                cv2.line(frame, p1, p2, (255,255,255), 2)
                        for (x,y), s in zip(person['keypoints'], person['scores']):
                            if s > Config.POSE_CONFIDENCE_THRESHOLD: cv2.circle(frame, (int(x), int(y)), 4, person_color, -1)
                processed_frames.append(frame)
            display_img = create_tiled_display(processed_frames, Config.MAX_DISPLAY_WIDTH, Config.MAX_DISPLAY_HEIGHT)
            if video_writer is None and display_img.size > 0:
                h, w, _ = display_img.shape
                fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
                # --- AUTOMATIC CODEC FALLBACK LOGIC ---
                try:
                    # Try the best, web-compatible codec first
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    if not video_writer.isOpened():
                        raise RuntimeError("Writer failed to open with avc1")
                    print("✅ Successfully initialized video writer with 'avc1' (H.264) codec.")
                except Exception as e:
                    print(f"⚠️ Failed to initialize with 'avc1' codec: {e}. Trying fallback 'mp4v'.")
                    # If H.264 fails, fall back to the most basic codec
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    if not video_writer.isOpened():
                        print("❌ CRITICAL ERROR: Could not initialize video writer with any codec.")
                        video_writer = None # Ensure it's None so we don't try to write
            if video_writer: 
                video_writer.write(display_img)

        frame_idx += 1
        if st_progress_bar and total_frames > 0:
            progress = min(1.0, (frame_idx / total_frames))
            st_progress_bar.progress(progress * 0.8, text=f"Step 1/3: Processing video frame {frame_idx}/{total_frames}")

    # --- FINAL SAVE ---
    if output_csv_path:
        if all_frames_data:
            df = pd.DataFrame(all_frames_data)
            df['exercise'] = exercise_label
            df['correctness'] = correctness_label
            df.to_csv(output_csv_path, index=False)
            print(f"✅ CSV data saved to {output_csv_path}")
        else:
            print("⚠️ No valid CSV data was extracted.")
    if video_writer:
        video_writer.release()
        print(f"✅ Visualization video saved to {output_video_path}")
    for cap in caps:
        cap.release()

