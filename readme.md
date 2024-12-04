# 2 Camera Norm Tracker

## Description
2 Camera Norm Tracker is a computer vision project designed to track objects using multiple camera feeds. It leverages YOLO for object detection and epipolar geometry for accurate tracking across different camera views. This project is useful for applications requiring precise object tracking in multi-camera setups, such as surveillance and sports analysis.

## Features
- Multi-camera object tracking
- Epipolar geometry for accurate cross-camera tracking
- Real-time visualization of 3D and 2D plots
- Customizable tracking parameters

## Installation
### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- YOLO model weights

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/2_camera_norm_tracker.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Place your camera calibration files in the `config_camera` directory.
2. Update the `config.py` file with your settings.
3. Run the main script:
    ```bash
    python main.py
    ```

## Screenshots/Demos
![3D Plot](screenshots/3d_plot.png)
![2D Plot](screenshots/2d_plot.png)

## Configuration
- `YOLO_MODEL`: Path to the YOLO model file.
- `NUM_CAM`: Number of cameras.
- `CONFIDENCE`: Confidence threshold for YOLO.
- `DISTANCE_THRESHOLD`: Distance threshold for epipolar line matching.

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- YOLO model weights

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Testing
Run the test cases using:
```bash
python -m unittest discover tests
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support
For issues, please open a GitHub issue or contact the maintainer.

## Contact
For questions, contact [yourname@example.com](mailto:yourname@example.com).

## References
- [YOLO](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)