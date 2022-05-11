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


def match_corr(corr_obj, img):
    """
    return the center coordinates of the location of 'corr_obj' in 'img'.
    :param corr_obj: 2D numpy array of size [H_obj x W_obj]
    containing an image of a component.
    :param img: 2D numpy array of size [H_img x W_img]
    where H_img >= H_obj and W_img>=W_obj,
    containing an image with the 'corr_obj' component in it.
    :return:
    match_coord: the two center coordinates in 'img'
    of the 'corr_obj' component.
    """
    # ====== YOUR CODE: ======
    object_conv = cv2.filter2D(corr_obj.astype('int64'), -1, corr_obj.astype('int64'), borderType=cv2.BORDER_CONSTANT)
    object_max = object_conv.max()
    image_corr = cv2.filter2D(img.astype('int64'), -1, corr_obj.astype('int64'), borderType=cv2.BORDER_CONSTANT)
    match_coord = np.unravel_index((np.abs(image_corr - object_max)).argmin(), image_corr.shape)
    # ========================

    return match_coord
