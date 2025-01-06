import pandas as pd
import numpy as np
import onnxruntime as ort
import os
from scipy.optimize import linear_sum_assignment

import cv2
import torch

from scipy.spatial.distance import cosine

# CONSTANTES
roi_means = np.array([123.675, 116.28, 103.53], dtype=np.float32)
roi_stds = np.array([58.395, 57.12, 57.375], dtype=np.float32)
onnx_model_path = "../../Filemail.com-TP4-TP5/reid_osnet_x025_market1501.onnx"
model = ort.InferenceSession(onnx_model_path)
print("Le modèle ONNX est chargé.")

def draw_tracks(frame, tracks):
    for track in tracks:
        x, y, w, h = track['bbox']
        track_id = track['id']
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"{track_id}", (int(x), int(y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

def load_det(file_path):
    columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    det = pd.read_csv(file_path, sep=' ', header=None, names=columns)
    return det

class KalmanFilter:
    def __init__(self, dt=1, u_x=0, u_y=0, std_acc=1, x_std_meas=1, y_std_meas=1):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas

        self.u = np.array([[u_x], [u_y]])
        self.x = np.zeros((4, 1))

        self.A = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        self.B = np.array([[(dt**2)/2, 0],
                            [0, (dt**2)/2],
                            [dt, 0],
                            [0, dt]])
        
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        
        self.R = np.array([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])
        
        self.P = np.eye(4)

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
    
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[:2].flatten()

    def update(self, z):
        z = np.array(z).reshape(-1, 1)
        
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x[:2].flatten()

def calculate_iou(box1, box2):
    b1_x1, b1_y1 = box1[0], box1[1]
    b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1 = box2[0], box2[1]
    b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * \
                 max(inter_rect_y2 - inter_rect_y1, 0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou

def preprocess_patch(im_crops): 
    roi_input = cv2.resize(im_crops, (64, 128)) 
    roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB) 
    roi_input = (np.asarray(roi_input).astype(np.float32) - roi_means) / roi_stds 
    roi_input = np.moveaxis(roi_input, -1, 0) 
    object_patch = roi_input.astype('float32') 
    return object_patch 

def compute_reid_features(tracker_box, frame):
    #print(tracker_box)
    bb_left, bb_top, bb_width, bb_height = tracker_box

    patch = frame[int(bb_top):int(bb_top + bb_height), int(bb_left):int(bb_left + bb_width)]

    if patch.size == 0:
        return None
    
    processed_patch = preprocess_patch(patch)
    
    with torch.no_grad():
        tensor = torch.from_numpy(processed_patch).unsqueeze(0)
        input_data = tensor.numpy()
        input_name = model.get_inputs()[0].name
        features = model.run(None, {input_name: input_data})[0]
    return features

def compute_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
        
    distance = cosine(features1.flatten(), features2.flatten())
    similarity = 1 / (1 + distance)
    return similarity

def associate_detections_to_tracks_with_reid(tracks, old_reid_vector, detections, frame, iou_threshold=0.3):
    ALPHA = 0.7
    BETA = 0.3
    
    #tracks = [t['predicted_bbox'] for t in tracker]

    if len(tracks) == 0 or len(detections) == 0 or len(old_reid_vector) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
            
    # Calculer la matrice IOU
    iou_matrix = np.zeros((len(tracks), len(detections)))

    for i, detection in enumerate(detections):
        detections_features = compute_reid_features(detection, frame)
        for j, track in enumerate(tracks):
            iou = calculate_iou(track['predicted_bbox'], detection)
            track_id = track['id']
            track_features = old_reid_vector.get(track_id)
            similarity = compute_similarity(track_features, detections_features)

            iou_matrix[j, i] = ALPHA * iou + BETA * similarity

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices).T
    
    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))
    
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] >= iou_threshold:
            matches.append(m.tolist())
            unmatched_tracks.remove(m[0])
            unmatched_detections.remove(m[1])
            
    return matches, unmatched_tracks, unmatched_detections

class Tracker_with_reid:
    def __init__(self, max_missed_frames=15):
        self.tracks = []
        self.next_id = 1
        self.max_missed_frames = max_missed_frames
        self.reid_features = {}

    def bbox_to_centroid(self, bbox):
        x = bbox[0] + bbox[2]/2
        y = bbox[1] + bbox[3]/2
        return np.array([[x], [y]])

    def centroid_to_bbox(self, centroid, width, height):
        x = centroid[0] - width/2
        y = centroid[1] - height/2
        return [x, y, width, height]

    def update(self, detected_boxes, frame):
        for track in self.tracks:
            predicted_centroid = track['kalman'].predict()
            track['predicted_bbox'] = self.centroid_to_bbox(predicted_centroid,
                                                          track['bbox'][2],
                                                          track['bbox'][3])


        matched_indices, unmatched_tracks, unmatched_detections = associate_detections_to_tracks_with_reid(
            self.tracks,
            self.reid_features,
            detected_boxes,
            frame
        )

        updated_features = {}

        for track_idx, det_idx in matched_indices:
            det_centroid = self.bbox_to_centroid(detected_boxes[det_idx])
            
            self.tracks[track_idx]['kalman'].update(det_centroid)
            
            self.tracks[track_idx]['bbox'] = detected_boxes[det_idx]
            self.tracks[track_idx]['missed_frames'] = 0

            reid_features_add = compute_reid_features(self.tracks[track_idx]['bbox'], frame)
            if reid_features_add is not None:
                        updated_features[track_idx] = reid_features_add

        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['missed_frames'] += 1
            
            if (self.tracks[track_idx]['missed_frames'] <= self.max_missed_frames):
                reid_features_add = compute_reid_features(self.tracks[track_idx]['bbox'], frame)
                if reid_features_add is not None:
                    updated_features[self.next_id] = reid_features_add

        self.tracks = [t for t in self.tracks if t['missed_frames'] <= self.max_missed_frames]

        for det_idx in unmatched_detections:
            kf = KalmanFilter(dt=1, std_acc=1, x_std_meas=1, y_std_meas=1)
            
            centroid = self.bbox_to_centroid(detected_boxes[det_idx])
            kf.x[:2] = centroid
            
            self.tracks.append({
                'id': self.next_id,
                'bbox': detected_boxes[det_idx],
                'kalman': kf,
                'missed_frames': 0
            })

            reid_features_add = compute_reid_features(detected_boxes[det_idx], frame)
            if reid_features_add is not None:
                updated_features[self.next_id] = reid_features_add
            

            self.next_id += 1

        self.reid_features = updated_features

if __name__ == "__main__":

    img_folder = "img1"
    output_video = "output/Yolov5l_Reid_video.mp4"

    file_path = "det/Yolov5l/det.txt"

    detections = load_det(file_path)

    tracker = Tracker_with_reid()

    frame_files = sorted(os.listdir(img_folder))
    frame_count = 1

    first_frame_path = os.path.join(img_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)


    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(img_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        current_detections = detections[detections['frame'] == frame_count]
        detected_boxes = current_detections[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()

        tracker.update(detected_boxes, frame)

        frame = draw_tracks(frame, tracker.tracks)
        video_writer.write(frame)
        frame_count += 1

    video_writer.release()
    print(f"Vidéo annotée générée : {output_video}")
