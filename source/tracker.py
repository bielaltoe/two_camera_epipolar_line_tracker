from ultralytics import YOLO
import numpy as np
import config
from Detection import ObjectDetection
import cv2


class Tracker:
    def __init__(self, yolo_trackers_list, cam_list):
        self.cam_list = cam_list
        self.trackers = [YOLO(tracker) for tracker in yolo_trackers_list]
        self.results = []
        self.frames = None
        self.classes = [config.CLASSES]
        print("created")

    def detect_and_track(self, cameras_frame: list):
        self.frames = cameras_frame
        self.results = []

        for tracker, frame in zip(self.trackers, self.frames):
            result = tracker.track(
                frame,
                persist=True,
                classes=self.classes,
                device="cuda:0",
                conf=config.CONFIDENCE,
                verbose=False,
                show=False,
                cls=True,
            )
            self.results.append(result[0])

    def divide_bbox(bbox):
        x_min, y_min, x_max, y_max = bbox
        subdivision_height = (y_max - y_min) / 5
        # subdivision_width = (x_max - x_min)/5

        sub_centroids = []

        for i in range(6):
            sub_centroids.append([(x_min + x_max) / 2, y_min + subdivision_height * i])

        return sub_centroids

    def single_centroid(self, bbox):
        return np.array([[(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2] for bb in bbox])

    def get_detections(self, show=False):
        detections = []
        # print(self.results)

        for i in range(len(self.results)):
            # print(result[i].boxes.xyxy.cpu().numpy())
            if self.results[i] is not None and self.results[i].boxes.id is not None:
                ids = self.results[i].boxes.id.cpu().numpy()
                bbox = self.results[i].boxes.xyxy.cpu().numpy()
                centroid = self.single_centroid(bbox)
                name = self.results[i].boxes.cls.cpu().numpy()
                for j, b, n, centroid in zip(ids, bbox, name, centroid):
                    detections.append(
                        ObjectDetection(
                            self.cam_list[i], j, b, self.frames[i], None, centroid, n
                        )
                    )

        return detections
