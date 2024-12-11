# 🐾 Meow Meow Breed Detection

**Meow Meow Breed Detection** is a real-time cat breed detection application built using Streamlit and YOLOv8. The app can identify and classify various cat breeds from live webcam feeds or uploaded images, providing an interactive and fun way to explore feline diversity.

## 🚀 Features

- **Real-Time Detection**: Use your webcam to identify cat breeds live.
- **Upload & Detect**: Upload an image to classify cat breeds instantly.
- **Background Music**: Enjoy the soothing *"Meow Meow Meow"* tune while detecting.
- **Interactive Interface**: Adjust detection thresholds and explore supported breeds.
- **Breed Filtering**: Exclude specific breeds (e.g., Sphynx) from detection results.

## 🖥️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Object Detection**: [YOLOv8](https://github.com/ultralytics/ultralytics)
- **Multimedia**: [Pygame](https://www.pygame.org/)

## 🐱 Detectable Cat Breeds

- Abyssinian  
- Bengal  
- Birman  
- Bombay  
- British Shorthair  
- Egyptian Mau  
- Maine Coon  
- Persian  
- Puspin  
- Ragdoll  
- Russian Blue  
- Siamese  
- American Shorthair  
- Scottish Fold  

## 🔧 Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/markanthony14/cat_breeds_detection.git
   cd cat_breeds_detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add YOLO Model**:
   Download the `cat_breed_15epochs.pt` model and place it in the project root.

4. **Prepare Background Music**:
   Ensure the `Meow meow meow meow.mp3` file is available in the root directory.

## 🎮 Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Select Input Mode**:
   - **Webcam**: View live breed detection through your webcam.
   - **Upload Image**: Upload a cat image to identify its breed.

3. **Control Music**:
   Use the "Pause/Resume Music" button in the sidebar to toggle background music.

4. **Adjust Confidence**:
   Fine-tune the detection confidence threshold using the sidebar slider.

## 🛠️ Customization

- **Modify Breed List**: Update detectable breeds by editing the `filter_results` function in the `app.py` script.
- **Adjust YOLO Weights**: Replace `cat_breed_15epochs.pt` with your custom-trained YOLO weights.

## 🌟 Demo

![App Interface](demo_screenshot.png)

## 🐾 Contributing

Contributions are welcome! Feel free to submit issues or pull requests for enhancements or bug fixes.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

