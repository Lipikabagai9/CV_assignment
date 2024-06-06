import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Step 1: Data Collection and Annotation
video_dataset = [
    ('correct.mp4', 0),
    ('failed.mp4', 1),
    ('cheating.mp4', 2)
]

# Step 2: Preprocessing
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Step 3: Feature Extraction
def detect_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    hand_landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            for lm in hand_landmark.landmark:
                h, w, _ = frame.shape
                hand_landmarks.append((int(lm.x * w), int(lm.y * h)))
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
    return hand_landmarks

def detect_bottle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([30, 150, 50])
    upper_color = np.array([255, 255, 180])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bottle_pos = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bottle_pos.append((x, y, w, h))
    return bottle_pos

def extract_features(frames):
    features = []
    for frame in frames:
        hand_pos = detect_hand(frame)
        bottle_pos = detect_bottle(frame)
        if hand_pos and bottle_pos:
            # Example feature: distance between hand and bottle
            hand_x, hand_y = hand_pos[0]
            bottle_x, bottle_y, bottle_w, bottle_h = bottle_pos[0]
            distance = np.sqrt((hand_x - bottle_x) ** 2 + (hand_y - bottle_y) ** 2)
            features.append(distance)
        else:
            print("Hand or bottle not detected in frame.")
    return features


# Step 4: Model Training
X = []  # Features
y = []  # Labels: 0 - Correct, 1 - Failed, 2 - Cheating

# Load and process videos, and extract features
for video, label in video_dataset:
    frames = extract_frames(video)
    features = extract_features(frames)
    for feature in features:
        X.append(feature)
        y.append(label)

# Ensure X and y have the same length
if len(X) != len(y):
    raise ValueError(f"Found input variables with inconsistent numbers of samples: {len(X)} and {len(y)}")

# Convert to numpy arrays for model input
X = np.array(X).reshape(-1, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Step 5: Real-time Detection
def real_time_detection(video_path, clf):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_features([frame])
        if features:
            # Reshape features to 2D array
            features = np.array(features).reshape(1, -1)

            prediction = clf.predict(features)
            if prediction == 0:
                print("Correct Flip")
            elif prediction == 1:
                print("Failed Flip")
            elif prediction == 2:
                print("Cheating")

            # Display frame with prediction
            label = "Correct Flip" if prediction == 0 else "Failed Flip" if prediction == 1 else "Cheating"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            print("Hand or bottle not detected in frame.")

    cap.release()
    cv2.destroyAllWindows()

# Example of running real-time detection
real_time_detection('test_video(correct).mp4', clf)
