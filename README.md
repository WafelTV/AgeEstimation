# Age Estimation with YOLO and CNN

This project uses **YOLOv8** (pretrained) for face detection and a **custom CNN model** for age estimation from webcam footage.

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run the following command to start the real-time age estimation:

```bash
python YOLO_age_detection.py
```

## Project Structure
- **`best_model.pth`**: Pre-trained CNN model for age prediction.
- **YOLOv8**: Used for detecting faces in real time.
- **OpenCV**: Captures the webcam stream.
- **PyTorch**: Runs the CNN model on detected faces.

## Controls
- Press **'q'** to exit the webcam stream.

## Notes
- Make sure your webcam is properly connected.
- Adjust thresholds if detection results aren't optimal.
- The CNN model should be trained separately and saved as `best_model.pth`.

## Author
Jakub Romanowski  
Kacper Trznadel
