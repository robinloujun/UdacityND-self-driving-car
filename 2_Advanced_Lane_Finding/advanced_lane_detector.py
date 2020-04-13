import os.path
import glob
import codecs
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from numpy_encoder import NumpyEncoder
from line import Line, fit_polynomial_sliding_windows, fit_polynomial_from_prior

debug = False
left_line = Line()
right_line = Line()
frame_count = 0
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


def calibrate_camera(calib_dir):
    """
    Perform the camera calibation with the calibration images in given path
    :param calib_dir: the directory where calibration images are saved
    :return: calibration parameters
    """

    # Pre-check: if calibration done before, there is no need to do it again
    calib_file = calib_dir + 'calib.json'
    if os.path.exists(calib_file):
        json_load = json.loads(codecs.open(
            calib_file, 'r', encoding='utf-8').read())
        mtx = np.array(json_load["camera_matrix"])
        dist = np.array(json_load["distortion_coefficients"])
        rvecs = np.array(json_load["rotation_vectors"])
        tvecs = np.array(json_load["translation_vectors"])
        print('Calibration once done. Load the calib file from ' + calib_dir)
    else:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        image_list = glob.glob(calib_dir + 'calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in image_list:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(50)

        cv2.destroyAllWindows()

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        # Write the calibration matrix in a txt for further use
        with open(calib_dir+'/calib.json', 'w') as json_file:
            json.dump({'camera_matrix': mtx, 'distortion_coefficients': dist,
                       'rotation_vectors': rvecs, 'translation_vectors': tvecs},
                      json_file, cls=NumpyEncoder)
        print('Calibration finished. The calib.json is stored in ' + calib_dir)

    return mtx, dist, rvecs, tvecs


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Apply the threshold on the absolute value of sobel results
    :param img: Grayscale
    :param orient: direction - x or y axis
    :param sobel_kernel: size of sobel kernel
    :param thresh: apply threshold on pixel intensity of gradient image
    return: binary image
    """
    # Calculate the derivative based on the orientation
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Absolute derivative to accentuate lines away from horizontal
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(sobel)

    # Apply the threshold
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Apply the threshold on the magnitude of gradient
    :param img: Grayscale
    :param sobel_kernel: size of sobel kernel
    :param thresh: apply threshold on pixel intensity of gradient image
    return: binary image
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)

    # Calculate gradient direction
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    # Apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) &
                  (scaled_sobel < thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Apply the threshold on the computed direction of gradient
    :param img: Grayscale
    :param sobel_kernel: size of sobel kernel
    :param thresh: apply threshold on pixel intensity of gradient image
    return: binary image
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)

    # Calculate gradient direction
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

    # Apply threshold
    binary_output = np.zeros_like(gradient_direction)
    binary_output[(gradient_direction > thresh[0]) &
                  (gradient_direction < thresh[1])] = 1

    return binary_output


def hsv_threshold(img, thres_lower, thres_upper):
    """
    Apply the threshold on the HSV image based on the channel value
    :param img: HSV image
    :param thres_lower: the lower threshold on one channel
    :param thres_upper: the upper threshold on one channel
    return: binary image
    """
    hsv_mask = cv2.inRange(img, thres_lower, thres_upper)
    # binary_output = cv2.bitwise_and(img, img, mask=hsv_mask)

    h, w = img.shape[:2]
    binary_output = np.zeros(shape=(h, w), dtype=np.uint8)
    binary_output = np.logical_or(binary_output, hsv_mask)
    return binary_output


def binarize_threshold(raw_img):
    """
    Apply all binariyation methods on the raw image
    :param raw_img: the raw image for binarization
    return: combined binary image
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)

    # Convert to grayscale image
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    combined = np.zeros_like(gray, dtype=np.uint8)

    # Extract the edges with the sobel operator and threshold
    sobelx_binary = abs_sobel_thresh(gray, 'x', 3, (20, 100))

    # Combine all binary image
    combined = np.logical_or(combined, sobelx_binary)

    # Apply the threshold on the magnitude of gradient
    mag_binary = mag_threshold(gray, thresh=(50, 255))
    combined = np.logical_or(combined, mag_binary)

    # Extract the white part in the image
    ret_white, mask = cv2.threshold(
        gray, thresh=205, maxval=255, type=cv2.THRESH_BINARY)
    white_binary = cv2.bitwise_and(gray, mask)
    combined = np.logical_or(combined, white_binary)

    # Extract the yellow part in the image
    yellow_lower = np.array([0, 70, 70], dtype="uint8")
    yellow_upper = np.array([45, 255, 255], dtype="uint8")
    yellow_binary = hsv_threshold(hsv, yellow_lower, yellow_upper)
    combined = np.logical_or(combined, yellow_binary)

    # Binarization based on Direction of the Gradient
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
    # No need to apply this binarization
    # combined = np.logical_or(combined, dir_binary)

    # Convert the dtype from bool to uint8 for next process
    combined = combined.astype(np.uint8)

    if debug:
        # Display raw images, all binarized images and the final output for parameter tuning
        fig, axs = plt.subplots(2, 3, figsize=(10, 3))
        axs[0, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Raw Image')

        axs[0, 1].imshow(white_binary, cmap='gray')
        axs[0, 1].set_title('White Mask')

        axs[0, 2].imshow(yellow_binary, cmap='gray')
        axs[0, 2].set_title('Yellow Mask')

        axs[1, 0].imshow(mag_binary, cmap='gray')
        axs[1, 0].set_title('Magnitude Mask')

        axs[1, 1].imshow(sobelx_binary, cmap='gray')
        axs[1, 1].set_title('Sobel Mask')

        axs[1, 2].imshow(combined, cmap='gray')
        axs[1, 2].set_title('Final Image')

        plt.show()

    return combined


def perspective_transform(img_in):
    """
    Apply perspective transform on input image
    :param img_in: the raw image
    :return: warped image and (inverse) perspective transformation matrix
    """
    img_size = img_in.shape

    # src = np.float32(
    #     [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    #      [((img_size[0] / 6) - 10), img_size[1]],
    #      [(img_size[0] * 5 / 6) + 60, img_size[1]],
    #      [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    # dst = np.float32(
    #     [[(img_size[0] / 4), 0],
    #      [(img_size[0] / 4), img_size[1]],
    #      [(img_size[0] * 3 / 4), img_size[1]],
    #      [(img_size[0] * 3 / 4), 0]])

    src = np.float32(
        [[(img_size[1] / 2) - 100, (img_size[0] / 2) + 100],
         [(img_size[1] / 2) + 100, (img_size[0] / 2) + 100],
         [img_size[1], img_size[0]],
         [0, img_size[0]]])

    dst = np.float32(
        [[0, 0],
         [img_size[1], 0],
         [img_size[1], img_size[0]],
         [0, img_size[0]]])

    # Get the perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(
        img_in, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)

    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(img_in, cmap='gray')
        axs[0].set_title('Raw Image')

        axs[1].imshow(warped, cmap='gray')
        axs[1].set_title('Warped Image')

        plt.show()

    return warped, M, Minv


def draw(undistort, warped, Minv, left_line, right_line):
    """
    Draw the generated information back onto the output frame
    :param undistort: undistorted image
    :param warped: warped image
    :param Minv: inverse perspective transform matrix
    :param left_line/right_line: detected lane lines
    :return result as output frame
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fit = left_line.current_fit_pixels
    right_fit = right_line.current_fit_pixels
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    lane_line_warp = np.zeros_like(undistort, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (undistort.shape[1], undistort.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)

    # Calculate and show the radius of curvature
    radius_of_curvature = np.mean(
        [left_line.radius_of_curvature_meters, right_line.radius_of_curvature_meters])
    cv2.putText(result, 'Radius of Curvature: {:.02f}m'.format(
        radius_of_curvature), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Calculate and show the offset to the lane center
    if left_line.detected and right_line.detected:  # Otherwise the base is missing
        left_base_pose = np.mean(
            left_line.allx[left_line.ally > 0.9 * left_line.ally.max()])
        right_base_pose = np.mean(
            right_line.allx[right_line.ally > 0.9 * right_line.ally.max()])
        offset_center = np.abs((right_base_pose - left_base_pose)/2 +
                               left_base_pose - (undistort.shape[1] - left_base_pose)) * xm_per_pix
        cv2.putText(result, 'Offset to Center: {:.02f}m'.format(
            offset_center), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result, 'Offset to Center: ?m', (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    if debug:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()

    return result


def process_image(img_in):
    """
    the image process of advanced lane finding
    :param img_in: the input image / video frame
    :return: img_out: the output image / video frame
    """
    global left_line, right_line, frame_count

    # First perform the image undistortion
    undist = cv2.undistort(img_in, mtx, dist, None, mtx)
    binarized = binarize_threshold(undist)
    warped, M, Minv = perspective_transform(binarized)

    if frame_count > 0 and left_line.detected and right_line.detected:
        left_line, right_line, fit_out = fit_polynomial_from_prior(
            warped, left_line, right_line)
    else:
        left_line, right_line, fit_out = fit_polynomial_sliding_windows(
            warped, left_line, right_line)

    if debug:
        plt.imshow(fit_out, cmap='gray')
        plt.show()

    img_out = draw(undist, warped, Minv, left_line, right_line)

    frame_count += 1

    return img_out


if __name__ == "__main__":
    camera_cal_dir = 'camera_cal/'
    mtx, dist, rvecs, tvecs = calibrate_camera(camera_cal_dir)

    images = glob.glob(camera_cal_dir + 'calibration*.jpg')

    img_out_dir = 'output_images/'

    # Save all undistorted chessboard images
    # Step through the list and search for chessboard corners
    # for fname in images:
    #     img = cv2.imread(fname)
    #     undist = cv2.undistort(img, mtx, dist, None, mtx)
    #     cv2.imshow('img', img)
    #     cv2.imshow('img', undist)
    #     cv2.waitKey(5)
    #     out_path = img_out_dir + 'undist_' + os.path.split(fname)[-1]
    #     print(out_path)
    #     cv2.imwrite(out_path, undist)
    # cv2.destroyAllWindows()

    # test_img = cv2.imread('test_images/test1.jpg')
    # img_out = process_image(test_img)

    # plt.imshow(img_out, cmap='gray')
    # plt.show()

    # undist = cv2.undistort(test_img, mtx, dist, None, mtx)
    # binary = binarize_threshold(undist)
    # warp, M, Minv = perspective_transform(undist)

    # cv2.imshow('img', binary)
    # cv2.waitKey(1000)

    # cv2.imwrite(img_out_dir+'undist_test1.jpg', undist)

    test_type = 'image1'

    if test_type == 'image':
        test_img = cv2.imread('test_images/test1.jpg')
        output_image = process_image(test_img)
    else:
        challenge_output = 'out.mp4'
        clip = VideoFileClip("project_video.mp4")
        # clip = VideoFileClip("challenge_video.mp4").subclip(0, 10)
        # clip = VideoFileClip("harder_challenge_video.mp4").subclip(0, 10)
        white_clip = clip.fl_image(process_image)
        white_clip.write_videofile(challenge_output, audio=False)
