import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from utils import video_to_frames, match_corr


def main():
    # ---------------------------- section 1.a -------------------------------
    # implemented match_corr

    # ---------------------------- section 1.b -------------------------------
    corisca_frames = video_to_frames('../given_data/Corsica.mp4', 250, 260)
    temp_corisca = np.asarray([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in corisca_frames])
    lower_part = int(temp_corisca.shape[1] // 3)
    gray_corisca = temp_corisca[:, lower_part:, 7:627]

    # ----------------------------- section 1.c -------------------------------
    len_panorama_img = int(2.5 * gray_corisca.shape[2])
    panorama_img = np.zeros((gray_corisca.shape[1], len_panorama_img))
    selected_frame = gray_corisca[125, :, :]
    panorama_img[:, int(0.75 * gray_corisca.shape[2]):(int(0.75 * gray_corisca.shape[2]) + gray_corisca.shape[2])] \
        = selected_frame

    figure_selected_frame = plt.figure(figsize=(8, 8))
    plot_frame = figure_selected_frame.add_subplot(2, 1, 1)
    plot_frame.imshow(selected_frame, cmap="gray")
    plot_frame.set_title("selected reference frame")

    plot_frame = figure_selected_frame.add_subplot(2, 1, 2)
    plot_frame.imshow(panorama_img, cmap="gray")
    plot_frame.set_title("panorama selected frame")

    early_frm = gray_corisca[50, :, :]
    later_frm = gray_corisca[200, :, :]

    figure_early_late = plt.figure(figsize=(8, 8))
    plot_frame = figure_early_late.add_subplot(2, 1, 2)
    plot_frame.imshow(early_frm, cmap="gray")
    plot_frame.set_title("earlier selected frame")

    plot_frame = figure_early_late.add_subplot(2, 1, 1)
    plot_frame.imshow(later_frm, cmap="gray")
    plot_frame.set_title("later selected frame")

    # ----------------------------- section 1.d -------------------------------
    rec_early_frm = early_frm[:, :int(0.25 * early_frm.shape[1])]
    high_rec_early_cen_cord = int(early_frm.shape[0] / 2)
    len_rec_early_cen_cord = int(0.5 * 0.25 * early_frm.shape[1])
    rec_late_frm = later_frm[:, int(0.75 * later_frm.shape[1]):]
    high_rec_late_cen_cord = int(later_frm.shape[0] / 2)
    len_rec_late_cen_cord = int(0.75 * later_frm.shape[1] + 0.5 * 0.25 * later_frm.shape[1])

    early_x, early_y = match_corr(rec_early_frm, selected_frame)
    later_x, later_y = match_corr(rec_late_frm, selected_frame)

    rec_early_late_fig = plt.figure(figsize=(8, 8))
    plot_frame = rec_early_late_fig.add_subplot(2, 1, 1)
    plot_frame.imshow(rec_early_frm, cmap="gray")
    plot_frame.set_title(f"Earlier rectangle frames, coordinates: %s %s" % (early_x, early_y))

    plot_frame = rec_early_late_fig.add_subplot(2, 1, 2)
    plot_frame.imshow(rec_late_frm, cmap="gray")
    plot_frame.set_title(f"Later rectangle frames, coordinates: %s %s" % (later_x, later_y))

    # ----------------------------- section 1.e -------------------------------
     earl_cor_high_diff = early_x - high_rec_early_cen_cord
    earl_cor_len_diff = early_y - len_rec_early_cen_cord
    late_cor_high_diff = later_x - high_rec_late_cen_cord
    late_cor_len_diff = later_y - len_rec_late_cen_cord
    vid_s_1 = gray_corisca.shape[1]
    vid_s_2 = gray_corisca.shape[2]

    early_pan = np.zeros(panorama_img.shape)

    first_frm_earl = 0
    last_frm_earl = gray_corisca.shape[2]

    early_pan[max(0, earl_cor_high_diff):min(vid_s_1, vid_s_1 + earl_cor_high_diff),
    max(int(0.75 * vid_s_2) + earl_cor_len_diff, 0):min((int(0.75 * vid_s_2) + vid_s_2) + earl_cor_len_diff,
                                                        panorama_img.shape[1])] = early_frm[
                                                                                  max(0, earl_cor_high_diff):min(
                                                                                      vid_s_1,
                                                                                      vid_s_1 + earl_cor_high_diff),
                                                                                  first_frm_earl:last_frm_earl]
    late_pan = np.zeros(panorama_img.shape)

    first_frm_late = 0
    last_frm_late = gray_corisca.shape[2]

    late_pan[max(0, late_cor_high_diff):min(vid_s_1, vid_s_1 + late_cor_high_diff),
    max(int(0.75 * vid_s_2) + late_cor_len_diff, 0):min((int(0.75 * vid_s_2) + vid_s_2) + late_cor_len_diff,
                                                        panorama_img.shape[1])] = later_frm[
                                                                                  max(0, late_cor_high_diff):min(
                                                                                      vid_s_1,
                                                                                      vid_s_1 + late_cor_high_diff),
                                                                                  first_frm_late:last_frm_late]

    panorama_img = np.where(np.logical_and(panorama_img != 0, early_pan != 0), (early_pan + panorama_img)/2, panorama_img)
    panorama_img = np.where(panorama_img == 0, early_pan, panorama_img)
    panorama_img = np.where(np.logical_and(panorama_img != 0,late_pan != 0), (late_pan + panorama_img)/2, panorama_img)
    panorama_img = np.where(panorama_img == 0, late_pan, panorama_img)

    panorama_fig = plt.figure(figsize=(16, 5))
    plot_frame = panorama_fig.add_subplot(1, 1, 1)
    plot_frame.imshow(panorama_img, cmap="gray")
    plot_frame.set_title("panorama image")
    plt.show()


if __name__ == '__main__':
    main()
