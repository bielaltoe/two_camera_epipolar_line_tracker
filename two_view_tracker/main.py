import colorsys
import random
import sys

import cv2
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np

from .epipolar_lines import EpipolarLine
from .tracker import Tracker
from .triangulation import triangulate_point_from_multiple_views_linear
from .video_loader import VideoLoader
import os
COLORS = {}


def id_to_rgb_color(id: int) -> tuple[int, int, int]:
    """
    Gera uma cor RGB única para um ID fornecido. Se o ID já estiver registrado,
    retorna a cor associada.

    Args:
        id (int): Um identificador único.

    Returns:
        tuple[int, int, int]: Cor RGB normalizada no intervalo (0-255).
    """
    if id not in COLORS:
        # Garante uma geração consistente baseada no ID
        random.seed(int(id))
        hue = random.uniform(0, 1)  # Matiz único entre 0 e 1
        saturation = 0.8  # Saturação fixa
        luminance = 0.6  # Luminosidade fixa

        # Converte HLS para RGB e escala para intervalo 0-255
        r, g, b = [
            int(x * 255) for x in colorsys.hls_to_rgb(hue, luminance, saturation)
        ]
        COLORS[id] = (r, g, b)  # Salva a cor como uma tupla
    return COLORS[id]


# Função para converter o gráfico do matplotlib em imagem para o OpenCV
def fig_to_image(fig):
    """
    Converte uma figura do Matplotlib em uma imagem para o OpenCV.
    """
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    return buf.reshape(h, w, 3)


def normalize_rgb_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Converte uma cor RGB de (0-255) para o formato (0-1) necessário pelo Matplotlib.

    Args:
        color (tuple[int, int, int]): Cor no formato (R, G, B) no intervalo 0-255.

    Returns:
        tuple[float, float, float]: Cor normalizada no formato (R, G, B) no intervalo 0-1.
    """
    return tuple(channel / 255.0 for channel in color)


def update_3d_plot(keypoint_list, ids, ax_3d):
    ax_3d.clear()
    for keypoints_3d, obj_id in zip(keypoint_list, ids):
        color_rgb = normalize_rgb_color(id_to_rgb_color(obj_id))
        ax_3d.scatter(
            keypoints_3d[0],
            keypoints_3d[1],
            keypoints_3d[2],
            c=[color_rgb],
            s=50,
            label=f"ID: {obj_id}",
        )

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_xlim([-4, 4])
    ax_3d.set_ylim([-4, 4])
    ax_3d.set_zlim([0, 4])
    ax_3d.legend(loc="upper right")


def rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Converte uma cor RGB para o formato BGR necessário pelo OpenCV.

    Args:
        color (tuple[int, int, int]): Cor RGB no formato (R, G, B) no intervalo 0-255.

    Returns:
        tuple[int, int, int]: Cor BGR no formato (B, G, R) no intervalo 0-255.
    """
    return color[2], color[1], color[0]


def update_2d_plot(keypoint_list, ids, ax_2d):
    ax_2d.clear()
    for keypoints_3d, obj_id in zip(keypoint_list, ids):
        color_rgb = normalize_rgb_color(id_to_rgb_color(obj_id))
        ax_2d.scatter(
            keypoints_3d[0],
            keypoints_3d[1],
            c=[color_rgb],
            label=f"ID: {obj_id}",
        )

    ax_2d.set_xlabel("X")
    ax_2d.set_ylabel("Y")
    ax_2d.set_xlim([-4, 4])
    ax_2d.set_ylim([-4, 4])
    ax_2d.legend(loc="upper right")


def main():

    if len(sys.argv) > 1:
        videos = sys.argv[1:3]
    else:
        print("Please, provide video files.")
        sys.exit(1)
    # Criando a figura do matplotlib
    fig_3d = plt.figure(figsize=(10, 5))
    ax_3d = fig_3d.add_subplot(121, projection="3d")
    ax_2d = fig_3d.add_subplot(122)
    plt.ion()
    
    cam_numbers = []
    for video in videos:
        cam = video
        cam_numbers.append(int(os.path.basename(cam).replace(".mp4", "").replace("cam", "")))
        
    print("Câmeras: ", cam_numbers)
    # Inicializando o tracker e o loader de vídeo
    tracker = Tracker(["yolo11x.pt", "yolo11x.pt"], cam_numbers)
    video_loader = VideoLoader(videos)
    print("Número de frames: ", video_loader.get_number_of_frames())

    epipolar_matcher = EpipolarLine()

    # Loop para processar os frames e exibir as visualizações
    for i in range(video_loader.get_number_of_frames()):
        frames = video_loader.get_frames()
        tracker.detect_and_track(frames)
        det = tracker.get_detections()

        matches = epipolar_matcher.match(det, cams=[1, 2])
        keypoints_3d_list = []
        ids = []
        for match in matches:
            match_cam_0, match_cam_1 = match

            print("Matches: ", match_cam_0.cam, match_cam_1.cam)
            proj_matrix_0 = epipolar_matcher.P_all[match_cam_0.cam]
            proj_matrix_1 = epipolar_matcher.P_all[match_cam_1.cam]

            feet_0 = (match_cam_0.bbox[2], match_cam_0.bbox[3])
            feet_1 = (match_cam_1.bbox[2], match_cam_1.bbox[3])

            points_2d_centroids = [
                match_cam_0.single_centroid,
                match_cam_1.single_centroid,
            ]

            points_2d_feets = [feet_0, feet_1]

            keypoints_3d = triangulate_point_from_multiple_views_linear(
                [proj_matrix_0, proj_matrix_1], points_2d_feets
            )

            keypoints_3d_list.append(keypoints_3d)
            ids.append(match_cam_0.id)

        update_3d_plot(keypoints_3d_list, ids, ax_3d)
        update_2d_plot(keypoints_3d_list, ids, ax_2d)

        frames[0], frames[1] = epipolar_matcher.plot_lines(
            frames[0],
            frames[1],
        )
        # Converte os gráficos para imagens do OpenCV
        image_3d = fig_to_image(fig_3d)

        # Exibe as imagens dos gráficos junto com o vídeo
        for d in det:
            bbox = d.bbox
            id = d.id
            cam = d.cam
            frame = d.frame
            centroid = d.single_centroid
            name = d.name

            cv2.putText(
                frame,
                "CLASS: " + str(int(name)),
                (int(bbox[0]), int(bbox[3])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 245, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                id_to_rgb_color(id),
                2,
            )

            cv2.putText(
                frame,
                str(id),
                (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame, (int(centroid[0]), int(centroid[1])), 5, id_to_rgb_color(id), -1
            )

        temp1 = cv2.resize(frames[0], (540, 360))
        temp2 = cv2.resize(frames[1], (540, 360))

        # Juntando o vídeo e o gráfico para exibir
        final_frame = np.hstack([temp1, temp2])

        # Concatena a imagem 3D ao lado do vídeo
        final_frame = np.vstack(
            [final_frame, cv2.resize(image_3d, (final_frame.shape[1], 360))]
        )
        if i == 0:

            result = cv2.VideoWriter(
                "result_two_cameras_few_people.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (final_frame.shape[1], final_frame.shape[0]),
            )

        result.write(final_frame)

        # Exibe o resultado
        cv2.imshow("Frame e Graficos 3D", final_frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    plt.ioff()


if __name__ == "__main__":
    main()
