import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def main():
    # Header Section
    st.markdown(
        """
        <style>
        .header {
            font-size: 40px;
            color: white;
            text-align: center;
        }
        .subheader {
            font-size: 20px;
            color: #6D6D6D;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="header">🐾 Meow Meow Breed Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Detect cat breeds in real-time or via uploaded images</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar for settings
    st.sidebar.header("🔧 Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

    # YOLO model loading
    model = YOLO("cat_breed_11-30.pt")

    # Options for input mode
    st.sidebar.header("📷 Choose Input Mode")
    mode = st.sidebar.radio("Input Mode", ("Webcam", "Upload Image"))

    if mode == "Webcam":
        # Webcam capture initialization
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("🚨 Error: Could not open webcam.")
            return

        st.sidebar.warning("Press **Stop Stream** to end the video feed.")
        stop_stream = st.sidebar.button("Stop Stream")

        # Video stream placeholder
        stframe = st.empty()

        # Stream video
        try:
            while not stop_stream:
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
                with st.container():
                    stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Release resources
            cap.release()
            st.sidebar.success("Video stream ended. You can restart the app to begin again.")

    elif mode == "Upload Image":
        # Upload image section
        uploaded_image = st.file_uploader("Upload an image of a cat", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            # Load and display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert to OpenCV format
            image = np.array(image)
            if image.shape[2] == 4:  # Convert RGBA to RGB if necessary
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Make predictions
            results = model.predict(image, conf=confidence_threshold)

            # Draw predictions on the image
            annotated_image = results[0].plot()

            # Convert to RGB for Streamlit display
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Display the annotated image
            st.image(annotated_image, caption="Predicted Image", use_column_width=True)

if __name__ == "__main__":
    main()
