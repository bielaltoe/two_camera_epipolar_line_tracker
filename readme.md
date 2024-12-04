# 2 Camera Epipolar Line Tracker

## Description
2 Camera Epipolar Line Tracker is a computer vision project designed to track objects using 2 camera feeds. It leverages YOLO11 for object detection and epipolar geometry for accurate tracking across different camera views. This project is useful for applications requiring object tracking in multi-camera setups, such as surveillance and sports analysis.

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
    git clone https://github.com/bielaltoe/two_camera_epipolar_line_tracker.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Enter in source directory:
    ```bash
    cd source
    ```
2. Place your camera calibration files in the `config_camera` directory.
3. Update the `config.py` file with your settings.
4. Run the main script:
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

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support
For issues, please open a GitHub issue or contact the maintainer.

## Contact
For questions, contact [gabriel.altoe@edu.ufes.br](mailto:gabriel.altoe@edu.ufes.br).

## References
- [YOLO](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)