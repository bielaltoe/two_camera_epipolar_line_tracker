import colorsys
import random

import cv2
import numpy as np

from .config import DISTANCE_THRESHOLD
from .load_fundamental_matrices import FundamentalMatrices


class EpipolarLine:

    def __init__(self):
        self.F_all = FundamentalMatrices().fundamental_matrices_all(
            [
                "config_camera/0.json",
                "config_camera/1.json",
                "config_camera/2.json",
                "config_camera/3.json",
            ]
        )
        self.P_all = FundamentalMatrices().projection_matrices_all(
            [
                "config_camera/0.json",
                "config_camera/1.json",
                "config_camera/2.json",
                "config_camera/3.json",
            ]
        )
        self.lines_1_2 = None
        self.lines_2_1 = None
        self.colors = {}

    def _calculate_lines(self, F, points):
        lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, F)
        return lines.reshape(-1, 3)

    def id_to_rgb_color(self, id: int) -> tuple[int, int, int]:
        """
        Gera uma cor RGB única para um ID fornecido. Se o ID já estiver registrado,
        retorna a cor associada.

        Args:
            id (int): Um identificador único.

        Returns:
            tuple[int, int, int]: Cor RGB normalizada no intervalo (0-255).
        """
        if id not in self.colors:
            # Garante uma geração consistente baseada no ID
            random.seed(int(id))
            hue = random.uniform(0, 1)  # Matiz único entre 0 e 1
            saturation = 0.8  # Saturação fixa
            luminance = 0.6  # Luminosidade fixa

            # Converte HLS para RGB e escala para intervalo 0-255
            r, g, b = [
                int(x * 255) for x in colorsys.hls_to_rgb(hue, luminance, saturation)
            ]
            self.colors[id] = (r, g, b)  # Salva a cor como uma tupla
        return self.colors[id]

    def dist_p_l(self, line, centroid):
        a, b, c = line
        x, y = centroid
        return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    def _cross_distance(self, bbox1, bbox2, line1, line2):
        x1, y1, x2, y2 = bbox2
        width_1 = x2 - x1
        height_1 = y2 - y1

        x3, y3, x4, y4 = bbox1
        width_2 = x4 - x3
        height_2 = y4 - y3

        a1, b1, c1 = line1
        a2, b2, c2 = line2
        d_cross = self.dist_p_l([a1, b1, c1], [(x1 + x2) / 2, (y1 + y2) / 2]) / abs(
            width_1 + height_1
        ) + self.dist_p_l([a2, b2, c2], [(x3 + x4) / 2, (y3 + y4) / 2]) / abs(
            width_2 + height_2
        )
        return d_cross

    def plot_lines(self, img1, img2):
        """
        Plots the epipolar lines on an image.

        Args:
            lines (np.ndarray): An array of epipolar lines.
            img (np.ndarray): The image on which to draw the lines.

        Returns:
            np.ndarray: The image with the epipolar lines drawn on it.
        """
        i = 0
        j = 0
        for r in self.lines_1_2:
            color1 = self.id_to_rgb_color(i)
            print(f"r: {r}")
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
            img2 = cv2.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 2)
            i += 1

        for r2 in self.lines_2_1:
            color2 = self.id_to_rgb_color(j)
            print(f"r2: {r2}")
            x0_2, y0_2 = map(int, [0, -r2[2] / r2[1]])
            x1_2, y1_2 = map(
                int, [img1.shape[1], -(r2[2] + r2[0] * img1.shape[1]) / r2[1]]
            )
            img1 = cv2.line(img1, (x0_2, y0_2), (x1_2, y1_2), (0, 255, 0), 2)
            j += 1

        return img1, img2

    def match(self, detetions, cams):
        self.lines_1_2 = []
        self.lines_2_1 = []

        F_1_2 = self.F_all[1][2]  # from camera 1 to camera 2
        F_2_1 = self.F_all[2][1]  # from camera 2 to camera 1

        detections_1 = [det for det in detetions if det.cam == cams[0]]
        detections_2 = [det for det in detetions if det.cam == cams[1]]
        if len(detections_1) < 1 or len(detections_2) < 1:
            return []

        centroids_1 = np.array([det.single_centroid for det in detections_1])
        centroids_2 = np.array([det.single_centroid for det in detections_2])

        # print(f"Centroids 1: {centroids_1}")

        lines_1_2 = self._calculate_lines(
            F_1_2, centroids_1
        )  # Lines from centroids in camera 1 to plot in camera 2
        lines_2_1 = self._calculate_lines(
            F_2_1, centroids_2
        )  # Lines from centroids in camera 2 to plot in camera 1

        self.lines_1_2 = lines_1_2
        self.lines_2_1 = lines_2_1

        cross_distances = []

        for i, det in enumerate(detections_1):
            for j, det2 in enumerate(detections_2):
                d_cross = self._cross_distance(
                    det.bbox, det2.bbox, lines_1_2[i], lines_2_1[j]
                )
                # simple_distance = self.dist_p_l(
                #     lines_1_2[i], det2.single_centroid[0]
                # ) + self.dist_p_l(lines_2_1[j], det.single_centroid[0])

                cross_distances.append((det, det2, d_cross))
                # print(
                #     "===================================================================================================="
                # )
                # print(
                #     f"CROSS Distance between line that came from centroid in {det.cam} - ID{det.id} and centroid ID{det2.id} from {det2.cam}: {d_cross}"
                # )
                # # print(
                # #     f"Distance between line that came from centroid in {det.cam} - ID{det.id} and  centroid ID{det2.id} from {det2.id}: {simple_distance}"
                # # )
                # print(
                #     "===================================================================================================="
                # )

        sorted_cross_distances = sorted(cross_distances, key=lambda x: x[2])

        # print("\nSorted cross distances: ", sorted_cross_distances)

        # CODIGO GABRIEL
        # real_matches = []
        # matches_ids = []
        # for possible_match in sorted_cross_distances:
        #     det1, det2, d_cross = possible_match
        #     if d_cross < config.DISTANCE_THRESHOLD and det.name == det2.name:
        #         if (det1.id, det2.id) not in matches_ids:
        #             real_matches.append((det1, det2))
        #             matches_ids.append((det1.id, det2.id))
        #             # print(
        #             #     f"Match found between ID{det1.id} from CAM{det1.cam} and ID{det2.id} from CAM{det2.cam}"
        #             # )

        # CODIGO CHATGPT
        used_ids_cam1 = set()  # IDs já utilizados da câmera 1
        used_ids_cam2 = set()  # IDs já utilizados da câmera 2
        real_matches = []
        real_matches_ids = []
        for possible_match in sorted_cross_distances:
            det1, det2, d_cross = possible_match
            if (
                d_cross < DISTANCE_THRESHOLD
                and det1.name == det2.name
                and det1.id not in used_ids_cam1
                and det2.id not in used_ids_cam2
            ):
                real_matches.append((det1, det2))
                real_matches_ids.append((det1.id, det2.id))
                used_ids_cam1.add(det1.id)  # Marca o ID da câmera 1 como utilizado
                used_ids_cam2.add(det2.id)  # Marca o ID da câmera 2 como utilizado

        print(
            f"Number of matches => {len(real_matches_ids)}\nMATCHES => {real_matches_ids}\n\n\n"
        )

        #################################################################################################################
        # OS VIDEOS ESTAO FICANDO DESSINCRONIZADOS FINAL DO VIDEO                                                       #
        #################################################################################################################

        # print("\nReal matches: ", real_matches)

        # print(
        #     f"Number of matches => {len(matches_ids)}\nMATCHES => {matches_ids}\n\n\n"
        # )
        # print("Detections 1: ", detections_1)
        # print("Detections 2: ", detections_2)

        return real_matches
