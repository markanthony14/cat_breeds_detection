from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("cat_breed_model.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Use the model to make a prediction on the captured frame
        result = model.predict(frame)

        # Display the result
        cv2.imshow("YOLO Prediction", result[0].plot())

        # Print result (optional)
        print(result)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Stopping the script...")
finally:
    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released and script stopped.")