import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels for gestures
labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: '7'}

# Set recognition threshold
THRESHOLD = 90  # Percentage threshold for recognition

def draw_bounding_box(frame, box, color=(0, 255, 0), label=None):
    """Draw a bounding box on the frame."""
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 4)
    if label:
        cv2.putText(frame, label, (x_min + 5, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

def get_hand_bounding_box(hand_landmarks):
    """Calculate bounding box for a hand based on landmarks."""
    h, w, _ = frame.shape
    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
    return (x_min, y_min, x_max, y_max)

def predict_gesture(data_aux):
    """Make a prediction based on hand landmarks."""
    prediction = model.predict([np.asarray(data_aux)])
    prediction_probabilities = model.predict_proba([np.asarray(data_aux)])
    predicted_class = int(prediction[0])
    confidence_score = prediction_probabilities[0][predicted_class] * 100
    return predicted_class, confidence_score

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)

            # Get bounding box and prediction
            box = get_hand_bounding_box(hand_landmarks)
            predicted_class, confidence_score = predict_gesture(data_aux)

            # Debugging: Print predictions and confidence scores
            print(f"Predicted Class: {predicted_class}, Confidence Score: {confidence_score:.2f}%")

            # Check if confidence score meets the threshold
            if confidence_score >= THRESHOLD:
                label = labels_dict.get(predicted_class, "Unknown")
                draw_bounding_box(frame, box, label=label)
            else:
                draw_bounding_box(frame, box, label="Unrecognizable")

    else:
        cv2.putText(frame,"No hands detected",(50 ,50),
                    cv2.FONT_HERSHEY_SIMPLEX ,1.5,(0 ,0 ,255),3)

    # Display the frame with all drawings and text
    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
