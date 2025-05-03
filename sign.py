import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define a dictionary to map landmark patterns to specific gestures
gesture_dict = {
    "thumbs_up": "Thumbs Up 👍",
    "peace_sign": "Peace ✌️",
    # Add more gestures and their corresponding text
}

def recognize_gesture(landmarks):
    """
    Function to recognize a gesture based on hand landmarks.
    You can implement your own logic here by comparing landmark positions.
    For simplicity, this is a placeholder function.
    """
    # Example: Checking if thumb tip is above index finger tip
    if landmarks[4].y < landmarks[8].y:
        return "thumbs_up"
    elif landmarks[12].y < landmarks[8].y and landmarks[16].y < landmarks[8].y:
        return "peace_sign"
    return None

# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to read from the video feed. Exiting...")
            break

        # Convert BGR to RGB for Mediapipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Optimize performance

        # Process the image with the Hands model
        results = hands.process(image)

        # Draw hand landmarks and recognize gestures
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks.landmark)
                if gesture:
                    cv2.putText(frame, gesture_dict.get(gesture, "Unknown Gesture"), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the processed frame
        cv2.imshow('Sign Language Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Sign language detection completed!")
