import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_histogram(frame, title: str, x_label: str, y_label: str):
    plt.plot(cv2.calcHist([frame], [0], None, [256], [0, 256]))
    if frame.shape[-1] == 3:
        plt.plot(cv2.calcHist([frame], [1], None, [256], [0, 256]))
        plt.plot(cv2.calcHist([frame], [2], None, [256], [0, 256]))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show(block=True)


def video_to_frames(vid_path: str, start_second, end_second):
    """
    Load a video and return its frames from the wanted time range.
    :param vid_path: video file path.
    :param start_second: time of first frame to be taken from the
    video in seconds.
    :param end_second: time of last frame to be taken from the
    video in seconds.
    :return:
    frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C]
    containing the wanted video frames.
    """
    # ====== YOUR CODE: ======
    # ========================
    cap = cv2.VideoCapture(vid_path)
    frame_set = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    i = 0  # iterator to iterate one frame each loop
    while cap.isOpened():
        ret, frame = cap.read()
        if start_second * fps > i:  # continue until you get to the required start second
            i += 1
            continue
        if end_second * fps < i:  # stop when you get to the required end second
            if end_second == start_second and end_second > 0:
                frame_set.append(cv2.cvtColor(frame, None))
            break
        if not ret or video_time < end_second:
            print('wrong input')
            return
        frame_set.append(cv2.cvtColor(frame, None))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frame_set)
