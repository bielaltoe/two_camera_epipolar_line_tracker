import cv2


class VideoLoader:
    def __init__(self, sources_list: list):
        self.sources_list = sources_list
        self.video_captures = []
        for source in self.sources_list:
            self.video_captures.append(cv2.VideoCapture(source))
            print("=> Video captured Successfully")

    def get_frames(self):
        frames = []
        for idx, video_capture in enumerate(self.video_captures):
            pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            # print(f"Video {idx} - Frame Position: {pos}")
            frames.append(video_capture.read()[1])
        return frames

    def get_number_of_frames(self):
        num_frames = []
        for video_capture in self.video_captures:
            print("=> Getting number of frames")
            print(
                f"Number of frames: {int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))}\n--------------------------------------------------------------------\n\n"
            )
            num_frames.append(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        return min(num_frames)
