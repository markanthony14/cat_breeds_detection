import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

def main():
    st.title("YOLO Cat Breed Detection")

    # Sidebar for settings
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

    # Unique key for the "Stop" button
    stop_button = st.sidebar.button("Stop", key="stop_button")

    # Webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    # Load the YOLO model
    model = YOLO("cat_breed_22k.pt")

    # Stream video
    stframe = st.empty()  # Placeholder for the video stream

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame. Is your webcam connected?")
            break

        # Make predictions
        results = model.predict(frame, conf=confidence_threshold)

        # Draw predictions on the frame
        annotated_frame = results[0].plot()

        # Convert frame to RGB for Streamlit display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        # Check if the user pressed the stop button
        if stop_button:
            break

    # Release resources
    cap.release()

if __name__ == "__main__":
    main()
