import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy


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


def poisson_noisy_image(orig_img, a):
    """
    Creates a Poisson noisy image.
    :param orig_img: The Original image. np array of size [H x W] and of type uint8.
    :param a: number of photons scalar factor
    :return:
    noisy_img: The noisy image. np array of size [H x W] and of type uint8.
    """
    # ====== YOUR CODE: ======
    # ========================
    photons_in_img = copy.deepcopy(orig_img).astype('float') * a
    poisson_photons_img = np.random.poisson(photons_in_img)
    poisson_img = poisson_photons_img / a
    noisy_img = np.clip(poisson_img, 0, 255).astype('uint8')
    return noisy_img


def denoise_by_l2(Y, X, num_iter, lambda_reg, epsilon0=0):
    """
    L2 image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    # ====== YOUR CODE: ======
    # ========================
    d_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    x_shape = X.shape
    Y = copy.deepcopy(Y).flatten('F')
    X = copy.deepcopy(X).flatten('F')
    Xout = Y
    for k in range(num_iter):
        double_con_X_k = double_conv(Xout, d_kernel, x_shape)
        G_k = Xout + lambda_reg * double_con_X_k - Y
        double_con_G_k = double_conv(G_k, d_kernel, x_shape)
        mu = (G_k.transpose() @ G_k) / (G_k.transpose() @ G_k + lambda_reg * G_k.transpose() @ double_con_G_k)
        Xout = (Xout - mu * G_k)
    conv_X_out = (cv2.filter2D(np.reshape(Xout, x_shape, order='F'), -1, d_kernel)).flatten('F')
    Err1 = (Xout - Y).transpose() @ (Xout - Y) + lambda_reg * (conv_X_out.transpose() @ conv_X_out)
    Err2 = (Xout - X).transpose() @ (Xout - X)
    Xout = np.reshape(Xout, x_shape, order='F').astype('uint8')
    return Xout, Err1, Err2


def double_conv(X, d_kernel, x_shape):
    first_conv = (cv2.filter2D(np.reshape(X, x_shape, order='F'), -1, d_kernel)).flatten('F')
    double_conv = (cv2.filter2D(np.reshape(first_conv, x_shape, order='F'), -1, d_kernel)).flatten('F')
    return double_conv


def denoise_by_tv(Y, X, num_iter, lambda_reg, epsilon0):
    """
    TV image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :param: epsilon0: small scalar for numerical stability
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    # ====== YOUR CODE: ======
    # ========================
    Y = copy.deepcopy(Y)
    X = copy.deepcopy(X)
    Xout = Y
    mu = 150 * epsilon0
    for k in range(num_iter):
        Xout_grad = np.gradient(np.gradient(Xout) / (np.sqrt(np.power(np.gradient(Xout), 2) + np.power(epsilon0, 2))))
        U_k = 2 * (Y - Xout) + lambda_reg * Xout_grad
        Xout = (Xout + mu * U_k / 2)
    Xout_grad = np.power(np.gradient(Xout, axis=0), 2) + np.power(np.gradient(Xout, axis=1), 2)
    TV = np.sum(np.sum(Xout_grad, axis=0), axis=0)
    Err1 = (Xout - Y).transpose() @ (Xout - Y) + lambda_reg * TV
    Err2 = (Xout - X).transpose() @ (Xout - X)
    Xout = Xout.astype('uint8')
    return Xout, Err1, Err2


def make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_func):
    Err1 = np.zeros(50)
    Err2 = np.zeros(50)
    Xout = None
    noisy_img = copy.deepcopy(noisy_img)
    resized_red_image = copy.deepcopy(resized_red_image)
    for n in range(50):
        Xout, Err1[n], Err2[n] = denoise_func(resized_red_image, noisy_img, n, 0.5, 1e-4)
    n = np.linspace(0, 50, 50)
    plt.plot(n, np.log2(Err1), 'r')
    plt.plot(n, np.log2(Err2), 'b')
    cv2.imshow("restoration image", Xout)
    plt.show()


def main():
    #   section 3.a
    frames = video_to_frames('../given_data/Flash Gordon Trailer.mp4', 20, 21)
    bgr = frames[0]
    cv2.imshow("20 second's frame", bgr)
    red_channel = bgr[:, :, 2]
    cv2.imshow("red channel", red_channel)
    dim = (int(red_channel.shape[1] / 2), int(red_channel.shape[0] / 2))
    resized_red_image = cv2.resize(red_channel, dim)
    cv2.imshow("resized_red_image", resized_red_image)
    noisy_img = poisson_noisy_image(resized_red_image, 0.5)
    cv2.imshow("noisy_img", noisy_img)
    cv2.destroyAllWindows()

    #   section 3.b
    make_restoration_and_error_graph(resized_red_image, noisy_img, denoise_by_l2)
    #   section 3.c
    make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_by_tv)
    cv2.destroyAllWindows()

    #   section 3.d
    #   section 3.e

    frames = video_to_frames('../given_data/Flash Gordon Trailer.mp4', 38, 39)
    bgr = frames[3]
    red_channel = bgr[:, :, 2]
    dim = (int(red_channel.shape[1] / 2), int(red_channel.shape[0] / 2))
    resized_red_image = cv2.resize(red_channel, dim)
    cv2.imshow("resized_red_image", resized_red_image)
    noisy_img = poisson_noisy_image(resized_red_image, 3)
    cv2.imshow("noisy_img", noisy_img)
    make_restoration_and_error_graph(resized_red_image, noisy_img, denoise_by_l2)
    make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_by_tv)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)

