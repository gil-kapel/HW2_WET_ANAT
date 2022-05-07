import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from utils import video_to_frames


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


def double_conv(X, d_kernel, x_shape):
    first_conv = cv2.filter2D(np.reshape(X, x_shape, order='F'), -1, d_kernel)
    second_conv = cv2.filter2D(first_conv, -1, d_kernel)
    return second_conv.flatten('F')


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
    Y = Y.flatten('F')
    X = X.flatten('F')
    Xout = Y
    Err1 = np.zeros(num_iter)
    Err2 = np.zeros(num_iter)
    for k in range(num_iter):
        double_conv_x_k = double_conv(Xout, d_kernel, x_shape)
        G_k = Xout + lambda_reg * double_conv_x_k - Y
        double_conv_g_k = double_conv(G_k, d_kernel, x_shape)
        mu = (G_k.transpose() @ G_k) / (G_k.transpose() @ G_k + lambda_reg * G_k.transpose() @ double_conv_g_k)
        Xout = Xout - mu * G_k
        conv_X_out = (cv2.filter2D(np.reshape(Xout, x_shape, order='F'), -1, d_kernel)).flatten('F')
        Err1[k] = (Xout - Y).transpose() @ (Xout - Y) + lambda_reg * (conv_X_out.transpose() @ conv_X_out)
        Err2[k] = (Xout - X).transpose() @ (Xout - X)
    Xout = np.reshape(Xout, x_shape, order='F').astype('uint8')
    return Xout, Err1, Err2


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
    Xout = Y
    mu = 150 * epsilon0
    Err1 = np.zeros(num_iter)
    Err2 = np.zeros(num_iter)
    for k in range(num_iter):
        x_grad = np.gradient(Xout)
        normal_x_grad = x_grad / np.sqrt(np.power(x_grad[0], 2) + np.power(x_grad[1], 2) + np.power(epsilon0, 2))
        x_divergence = np.ufunc.reduce(np.add, [np.gradient(normal_x_grad[i], axis=i) for i in range(2)])
        U_k = 2 * (Y - Xout) + lambda_reg * x_divergence
        Xout = Xout + mu * U_k / 2
        Xout_grad = np.sqrt(np.power(np.gradient(Xout)[0], 2) + np.power(np.gradient(Xout)[1], 2))
        TV = np.sum(Xout_grad)
        Err1[k] = (Xout.flatten('F') - Y.flatten('F')).transpose() @ (Xout.flatten('F') - Y.flatten('F')) + lambda_reg * TV
        Err2[k] = (Xout.flatten('F') - X.flatten('F')).transpose() @ (Xout.flatten('F') - X.flatten('F'))
    Xout = Xout.astype('uint8')
    return Xout, Err1, Err2


def make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_func, num_iter, lambda_reg, epsilon0):
    noisy_img = copy.deepcopy(noisy_img)
    resized_red_image = copy.deepcopy(resized_red_image)
    Xout, Err1, Err2 = denoise_func(noisy_img, resized_red_image, num_iter,
                                    lambda_reg, epsilon0)
    n = np.linspace(0, num_iter, num_iter)
    plt.plot(n, np.log2(1 + Err1), 'r', label='Err1')
    plt.plot(n, np.log2(1 + Err2), 'b', label='Err2')
    plt.legend()
    plt.xlabel('number of epochs')
    plt.ylabel('error rate - log scale')
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
    noisy_img = poisson_noisy_image(resized_red_image, 1/3)
    cv2.destroyAllWindows()
    cv2.imshow("noisy_img", noisy_img)

    # section 3.b
    make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_by_l2, 50, 0.5, 0)

    # section 3.c
    make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_by_tv, 200, 20, 2e-4)

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
    make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_by_l2, 50, 0.5, 0)
    make_restoration_and_error_graph(noisy_img, resized_red_image, denoise_by_tv, 200, 20, 2e-4)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)

