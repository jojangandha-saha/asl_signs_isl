import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

train = pd.read_csv('../train_new_csv.csv')
pq_file = '../asl_dataset_new/kaggle/new_asl_signs/train_landmark_files/62590/1031693648.parquet'
xyz = pd.read_parquet(pq_file)

def create_frame_landmark_df(results,frame):
    xyz_skel = xyz[['type','landmark_index']].drop_duplicates().reset_index(drop = True) \
    .copy()
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()
    
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i,['x','y','z']] = [point.x,point.y,point.z]
        
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i,['x','y','z']] = [point.x,point.y,point.z]
        
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i,['x','y','z']] = [point.x,point.y,point.z]
    
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i,['x','y','z']] = [point.x,point.y,point.z]
    face = face.reset_index() \
        .rename(columns = {'index' : 'landmark_index'}) \
        .assign(type = 'face')
    
    pose = pose.reset_index() \
        .rename(columns = {'index' : 'landmark_index'}) \
        .assign(type = 'pose')
    left_hand = left_hand.reset_index() \
        .rename(columns = {'index' : 'landmark_index'}) \
        .assign(type = 'left_hand')
    right_hand = right_hand.reset_index() \
        .rename(columns = {'index' : 'landmark_index'}) \
        .assign(type = 'right_hand')
    
    landmarks = pd.concat([face,pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks,on = ['type', 'landmark_index'], how= 'left')
    landmarks = landmarks.assign(frame = frame)
    return landmarks


def do_capture_loop():
    all_landmarks = []
    try:
        last_time = time.time()
       # For webcam input:
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
            frame = 0
            while cap.isOpened():
                frame += 1
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
            
                
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                #Create landmark df
                landmarks = create_frame_landmark_df(results,frame)
                all_landmarks.append(landmarks)
            
                # # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                delta_time = time.time() - last_time
                cur_fps = np.around(frame / delta_time, 1)
                print(cur_fps)
                print(cap.CAP_PROP_FPS)#0.6
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    except:
        return all_landmarks
    return all_landmarks


if __name__ == '__main__':
    landmarks = do_capture_loop()
    landmarks = pd.concat(landmarks).reset_index(drop=True) \
                .to_parquet('output.parquet')