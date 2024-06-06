import cv2
import numpy as np

#color range for yellow
yellow_range = ((20, 100, 100), (30, 255, 255))

def detect_yellow_circle(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only yellow color
    mask = cv2.inRange(hsv, yellow_range[0], yellow_range[1])
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    # Find contours in the threshold image
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_circle = None
    largest_radius = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out very small contours
            # Fit a circle to each contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > largest_radius:
                largest_radius = radius
                largest_circle = (int(x), int(y)), int(radius)
    return largest_circle

def estimate_angular_velocity(current_center, previous_center):
    if previous_center is not None and current_center is not None:
        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        angle = np.arctan2(dy, dx)
        time_interval = 1  # Assuming each frame corresponds to 1 unit of time
        angular_velocity = angle / time_interval
        return angular_velocity
    return 0

def predict_next_center(current_center, angular_velocity):
    if current_center:
        center_x, center_y = current_center
        next_x = int(center_x + 30 * np.cos(angular_velocity))
        next_y = int(center_y + 30 * np.sin(angular_velocity))
        return (next_x, next_y)
    return None

# Read video file
cap = cv2.VideoCapture('archery.mp4')

# Initialize variables to store previous center and angular velocity
previous_center = None
angular_velocity = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the largest yellow circle in the current frame
    largest_circle = detect_yellow_circle(frame)

    if largest_circle:
        current_center, radius = largest_circle
        cv2.circle(frame, current_center, radius, (0, 0, 0), 2)  # Draw detected circle
        cv2.circle(frame, current_center, 7, (0, 0, 0), -1)  # Draw detected center

        # Estimate angular velocity
        if previous_center:
            angular_velocity = estimate_angular_velocity(current_center, previous_center)

        # Predict the next center based on angular velocity
        predicted_center = predict_next_center(current_center, angular_velocity)
        if predicted_center:
            cv2.circle(frame, predicted_center, 7, (220, 0, 0), -1)  # Draw predicted center

        # Update previous center
        previous_center = current_center

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
