# monitoring/anomaly_detection.py
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model(r'C:/Users/SRAVA/action_recog.h5')

# Define the categories
categories = ['wave', 'walk', 'turn', 'throw', 'talk', 'stand', 'smile', 'situp', 'sit', 'shake_hands', 
              'run', 'push', 'punch', 'pour', 'pick', 'laugh', 'jump', 'hug', 'hit', 'handstand', 
              'fall_floor', 'eat', 'drink', 'dribble', 'climb_stairs', 'climb', 'clap', 'chew', 
              'catch', 'brush_hair']

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Adjust size based on model requirements
    normalized_frame = resized_frame / 255.0
    return normalized_frame

def predict_action(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    predicted_class = np.argmax(predictions, axis=1)
    return categories[predicted_class[0]]

def detect_anomalies(video_path):
    cap = cv2.VideoCapture(video_path)
    anomaly_detected = False
    start_time = None
    end_time = None
    detected_action = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        action = predict_action(frame)

        if action == 'fall_floor':  # Example anomaly
            if not anomaly_detected:
                start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - 15  # 15 seconds before
                anomaly_detected = True
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 + 15  # 15 seconds after
            detected_action = action

        if anomaly_detected and end_time:
            save_clip(video_path, start_time, end_time)
            break

    cap.release()
    return detected_action

def save_clip(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = video_path.replace('.mp4', '_anomaly.avi')  # Save as a new file
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    while cap.isOpened():
        ret, frame = cap.read()
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not ret or current_time > end_time:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path