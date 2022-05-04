import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy


def show_histogram(frame, title: str, x_label: str, y_label: str):
    plt.plot(cv2.calcHist([frame], [0], None, [256], [0, 256]))
    if frame.shape[-1] == 3:
        plt.plot(cv2.calcHist([frame], [1], None, [256], [0, 256]))
        plt.plot(cv2.calcHist([frame], [2], None, [256], [0, 256]))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show(block=True)

