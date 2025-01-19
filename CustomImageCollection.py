import os
import cv2
import time

# Directory to store collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 4  # Number of classes for gestures
dataset_size = 100  # Number of images per class

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Get ready to collect data for class {j}')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display instructions to start data collection
        cv2.putText(frame, f"Class {j}: Press 'q' to start", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        # Wait for user to press 'q' to start
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Countdown before starting data collection
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break

        # Display countdown on the screen
        cv2.putText(frame, f'Starting in {i}', (100, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)
        cv2.waitKey(1000)  # Wait for 1 second

    # Start collecting images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        # Display progress on the screen
        cv2.putText(frame, f'Collecting: Class {j} ({counter + 1}/{dataset_size})', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                    cv2.LINE_AA)
        
        # Show live feed and save the captured image
        cv2.imshow('Data Collection', frame)

        # Save the captured image every iteration
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        
        counter += 1
        
        # Check for 'q' key press to exit early if needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Data collection for class {j} completed!')

cap.release()
cv2.destroyAllWindows()
