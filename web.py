import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180 else angle

class BicepCurlCounter(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate and show angle
            angle = calculate_angle(shoulder, elbow, wrist)
            cv2.putText(img, f"{int(angle)}°", 
                        tuple(np.multiply(elbow, [img.shape[1], img.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Count reps
            if angle > 160: self.stage = "down"
            if angle < 35 and self.stage == "down":
                self.stage = "up"
                self.counter += 1
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        
        # Display counter
        cv2.putText(img, f"Reps: {self.counter}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Simple Bicep Curl Counter")
st.write("Perform bicep curls with your left arm to count reps")

webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=BicepCurlCounter,
    media_stream_constraints={"video": True, "audio": False}
)

st.info("Tip: Keep your elbow stationary and move only your forearm")