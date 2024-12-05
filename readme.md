# 2 Camera Epipolar Line 3D Tracker 🎥

## 📝 Description
The 2 Camera Epipolar Line Tracker is a computer vision project designed to track objects using two camera feeds. It leverages YOLO for object detection and epipolar geometry for accurate tracking across different camera views. This project is useful for applications requiring object tracking in multi-camera setups, such as surveillance and sports analysis.

## ✨ Features
- 📹 Multi-camera object tracking
- 📐 Epipolar geometry for accurate cross-camera tracking
- 📊 Real-time visualization of 3D and 2D plots
- ⚙️ Customizable tracking parameters

## 🚀 Installation
### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- YOLO model weights

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/bielaltoe/two_camera_epipolar_line_tracker.git
    ```
2. Navigate to the project directory:
    ```bash
    cd two_camera_epipolar_line_tracker
    ```

## 💻 Usage
1. Enter the source directory:
    ```bash
    cd two_view_tracker
    pip install .
    ```
2. Place your camera calibration files in the `config_camera` directory.
3. Update the `config.py` file with your settings.
4. Run the main script:
    ```bash
    two-view-tracker.exe your_video_path/cam1.mp4 your_video_path/cam2.mp4
    ```

## 📸 Demo Results
### Tracking Few People
<div align="center">
  <img src="./gifs/result_two_cameras_few_people.gif" alt="Test with few people" width="700"/>
  <p><i>Tracking demonstration with a few people in the scene</i></p>
</div>

### Multiple People Tracking
<div align="center">
  <img src="./gifs/multiple_people_vid.gif" alt="Test with multiple people" width="700"/>
  <p><i>System performance with multiple people</i></p>
</div>

### Umbrella Tracking Test
<div align="center">
  <img src="./gifs/umbrella_vid.gif" alt="Test with umbrella" width="700"/>
  <p><i>Demonstration of tracking an umbrella</i></p>
</div>

## ⚙️ Configuration
| Parameter            | Description                                    |
|----------------------|------------------------------------------------|
| `YOLO_MODEL`         | Path to the YOLO model file                    |
| `NUM_CAM`            | Number of cameras                              |
| `CONFIDENCE`         | Confidence threshold for YOLO                  |
| `DISTANCE_THRESHOLD` | Distance threshold for epipolar line matching  |

## 🤝 Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 💬 Support
For issues, please open a GitHub issue or contact the maintainer.

## 📧 Contact
For questions, contact [gabriel.altoe@edu.ufes.br](mailto:gabriel.altoe@edu.ufes.br)

## 📚 References
- [YOLO](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
