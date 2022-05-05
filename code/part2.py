import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy


def main():
    # section 2.a
    keyboard = cv2.imread('../given_data/keyboard.jpg')
    cv2.imshow('Original keyboard', keyboard)
    kernel_a = np.ones((8, 1), np.uint8)
    kernel_b = np.ones((1, 8), np.uint8)
    erosion_a = cv2.erode(keyboard, kernel_a, iterations=1)
    erosion_b = cv2.erode(keyboard, kernel_b, iterations=1)
    cv2.imshow('Vertical erosion keyboard', erosion_a)
    cv2.imshow('Horizon erosion keyboard', erosion_b)
    threshold = 0.2 * 255
    bw_keyboard = cv2.threshold(erosion_a + erosion_b, threshold, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('threshold keyboard', bw_keyboard)
    # section 2.b
    inv_bw_keyboard = 255 - bw_keyboard
    inv_bw_keyboard_median_filtered = cv2.medianBlur(inv_bw_keyboard, 9)
    cv2.imshow('Median filtered keyboard', inv_bw_keyboard_median_filtered)
    # section 2.c
    kernel_c = np.ones((8, 8), np.uint8)
    square_erosion = cv2.erode(inv_bw_keyboard_median_filtered, kernel_c, iterations=1)
    cv2.imshow('Square erosion keyboard', square_erosion)
    # section 2.d
    binary_processed_image = square_erosion / 255
    intersection_image = binary_processed_image * keyboard
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen_image = cv2.filter2D(intersection_image, -1, sharpen_kernel)
    cv2.imshow('sharpen intersect keyboard', sharpen_image)
    threshold_b = 0.9 * 255
    keyboard_values = cv2.threshold(sharpen_image, threshold_b, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('Filtered keys of the keyboard', keyboard_values)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
