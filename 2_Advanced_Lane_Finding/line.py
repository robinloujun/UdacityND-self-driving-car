import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


class Line():
    """
    Containing all characteristics of each line detection
    """

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # length of the sequences that will be saved (n iterations)
        self.deque_length = 10
        # # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.deque_length)
        # # average x values of the fitted line over the last n iterations
        self.bestx = None
        # # polynomial coefficients averaged over the last n iterations
        # self.best_fit = None
        # polynomial coefficients for the most recent fit in pixels
        self.current_fit_pixels = None
        # polynomial coefficients for the most recent fit in meters
        self.current_fit_meters = None
        # polynomial coefficients for the last n fits in pixels
        self.recent_fit_pixels = deque(maxlen=self.deque_length)
        # polynomial coefficients for the last n fits in meters
        self.recent_fit_meters = deque(maxlen=self.deque_length)
        # radius of curvature of the line in pixels
        self.radius_of_curvature_pixels = None
        # radius of curvature of the line in meters
        self.radius_of_curvature_meters = None
        # base position of the line in pixels
        # self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def best_fit(self):
        return np.mean(self.recent_fit_pixels, axis=0)

    def measure_curvature_pixels(self):
        y_eval = 0
        fit = np.mean(self.recent_fit_pixels, axis=0)
        return (1 + (2*fit[0]*y_eval + fit[1])**2)**1.5 / np.abs(2*fit[0])

    def measure_curvature_meters(self):
        y_eval = 0
        fit = np.mean(self.recent_fit_meters, axis=0)
        return (1 + (2*fit[0]*y_eval*ym_per_pix + fit[1])**2)**1.5 / np.abs(2*fit[0])

    # def calculate_base_pose(self):
    #     line_bottom_pose = np.mean(self.allx[self.ally > 0.9 * self.ally.max()])
    #     return line_bottom_pose

    def reset_line(self):
        self.detected = False
        self.recent_xfitted = deque(maxlen=self.deque_length)
        self.bestx = deque(maxlen=self.deque_length)
        # self.best_fit = None
        self.current_fit_pixels = None
        self.current_fit_meters = None
        self.recent_fit_pixels = deque(maxlen=self.deque_length)
        self.recent_fit_meters = deque(maxlen=self.deque_length)
        self.radius_of_curvature_pixels = None
        self.radius_of_curvature_meters = None
        # self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None


def fit_polynomial_sliding_windows(binary_warped, left_line, right_line):
    """
    Fit a polynomial for both lane lines with sliding windows in given warped image
    :param binary_warped: the input warped binary image
    :param left_line/right line: lane lines to be updated
    :return img_out: the output image
    :return left_line/right_line: updated lane lines
    """
    # Reset the two lane lines
    # left_line.reset_line
    # right_line.reset_line

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    img_out = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(len(histogram)//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # Find the four below boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        # cv2.rectangle(img_out, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(img_out, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                          (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                           (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window
        # (`right` or `leftx_current`) on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    left_line.allx = nonzerox[left_lane_inds]
    left_line.ally = nonzeroy[left_lane_inds]
    right_line.allx = nonzerox[right_lane_inds]
    right_line.ally = nonzeroy[right_lane_inds]

    # Fit new polynomials
    if not left_line.allx.any() or not left_line.ally.any():
        left_fit_pixels = left_line.current_fit_pixels
        left_fit_meters = left_line.current_fit_meters
        left_line.detected = False
    else:
        # Fit a second order polynomial to each using `np.polyfit` in pixels and meters
        left_fit_pixels = np.polyfit(left_line.ally, left_line.allx, 2)
        left_fit_meters = np.polyfit(
            left_line.ally*ym_per_pix, left_line.allx*xm_per_pix, 2)
        left_line.detected = True

    if not right_line.allx.any() or not right_line.ally.any():
        right_fit_pixels = right_line.current_fit_pixels
        right_fit_meters = right_line.current_fit_meters
        right_line.detected = False
    else:
        # Fit a second order polynomial to each using `np.polyfit` in pixels and meters
        right_fit_pixels = np.polyfit(right_line.ally, right_line.allx, 2)
        right_fit_meters = np.polyfit(
            right_line.ally*ym_per_pix, right_line.allx*xm_per_pix, 2)
        right_line.detected = True

    # Update the current fits in class Line
    left_line.current_fit_pixels = left_fit_pixels
    left_line.current_fit_meters = left_fit_meters
    right_line.current_fit_pixels = right_fit_pixels
    right_line.current_fit_meters = right_fit_meters

    # Append the current fit to the deque
    left_line.recent_fit_pixels.append(left_line.current_fit_pixels)
    left_line.recent_fit_meters.append(left_line.current_fit_meters)
    right_line.recent_fit_pixels.append(right_line.current_fit_pixels)
    right_line.recent_fit_meters.append(right_line.current_fit_meters)

    # Update the radius of curvature
    left_line.radius_of_curvature_pixels = left_line.measure_curvature_pixels()
    left_line.radius_of_curvature_meters = left_line.measure_curvature_meters()
    right_line.radius_of_curvature_pixels = right_line.measure_curvature_pixels()
    right_line.radius_of_curvature_meters = right_line.measure_curvature_meters()

    # Sanity check with radius of curvature
    if left_line.radius_of_curvature_meters - right_line.radius_of_curvature_meters > 100:
        left_line.detected = False
        right_line.detected = False

    # # Update the base pose
    # left_line.line_base_pos = left_line.calculate_base_pose()
    # right_line.line_base_pos = right_line.calculate_base_pose()

    # # Sanity check with base pose
    # if np.abs(left_line.line_base_pos * xm_per_pix - right_line.line_base_pos *xm_per_pix) < 3:
    #     left_line.detected = False
    #     right_line.detected = False

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit_pixels[0]*ploty**2 + \
            left_fit_pixels[1]*ploty + left_fit_pixels[2]
        right_fitx = right_fit_pixels[0]*ploty**2 + \
            right_fit_pixels[1]*ploty + right_fit_pixels[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Update the fitted x values
    left_line.recent_xfitted = deque(maxlen=left_line.deque_length)
    left_line.bestx = left_fitx
    left_line.recent_xfitted.append(left_fitx)
    right_line.recent_xfitted = deque(maxlen=right_line.deque_length)
    right_line.bestx = right_fitx
    right_line.recent_xfitted.append(right_fitx)

    ## Visualization ##
    # Colors in the left and right lane regions
    img_out[left_line.ally, left_line.allx] = [255, 0, 0]
    img_out[right_line.ally, right_line.allx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return left_line, right_line, img_out


def fit_polynomial_from_prior(binary_warped, left_line, right_line):
    """
    Fit a polynomial for both lane lines with previous info in given warped image
    :param binary_warped: the input warped binary image
    :param left_line/right line: lane lines to be updated
    :return img_out: the output image
    :return left_line/right_line: updated lane lines
    """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    img_shape = binary_warped.shape

    # Get the polynomial coefficients from the last fit
    left_fit_pixels = left_line.current_fit_pixels
    right_fit_pixels = right_line.current_fit_pixels

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    # Hint: consider the window areas for the similarly named variables
    # in the previous quiz, but change the windows to our new search area
    left_lane_inds = ((nonzerox > (left_fit_pixels[0]*nonzeroy**2 +
                                   left_fit_pixels[1]*nonzeroy + left_fit_pixels[2] - margin)) &
                      (nonzerox < (left_fit_pixels[0]*nonzeroy**2 +
                                   left_fit_pixels[1]*nonzeroy + left_fit_pixels[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_pixels[0]*nonzeroy**2 +
                                    right_fit_pixels[1]*nonzeroy + right_fit_pixels[2] - margin)) &
                       (nonzerox < (right_fit_pixels[0]*nonzeroy**2 +
                                    right_fit_pixels[1]*nonzeroy + right_fit_pixels[2] + margin)))

    # Again, extract left and right line pixel positions
    left_line.allx = nonzerox[left_lane_inds]
    left_line.ally = nonzeroy[left_lane_inds]
    right_line.allx = nonzerox[right_lane_inds]
    right_line.ally = nonzeroy[right_lane_inds]

    # Fit new polynomials
    if not left_line.allx.any() or not left_line.ally.any():
        left_fit_pixels = left_line.current_fit_pixels
        left_fit_meters = left_line.current_fit_meters
        left_line.detected = False
    else:
        # Fit a second order polynomial to each using `np.polyfit` in pixels and meters
        left_fit_pixels = np.polyfit(left_line.ally, left_line.allx, 2)
        left_fit_meters = np.polyfit(
            left_line.ally*ym_per_pix, left_line.allx*xm_per_pix, 2)
        left_line.detected = True

    if not right_line.allx.any() or not right_line.ally.any():
        right_fit_pixels = right_line.current_fit_pixels
        right_fit_meters = right_line.current_fit_meters
        right_line.detected = False
    else:
        # Fit a second order polynomial to each using `np.polyfit` in pixels and meters
        right_fit_pixels = np.polyfit(right_line.ally, right_line.allx, 2)
        right_fit_meters = np.polyfit(
            right_line.ally*ym_per_pix, right_line.allx*xm_per_pix, 2)
        right_line.detected = True

    # Update the current fits in class Line
    left_line.current_fit_pixels = left_fit_pixels
    left_line.current_fit_meters = left_fit_meters
    right_line.current_fit_pixels = right_fit_pixels
    right_line.current_fit_meters = right_fit_meters

    # Append the current fit to the deque
    left_line.recent_fit_pixels.append(left_line.current_fit_pixels)
    left_line.recent_fit_meters.append(left_line.current_fit_meters)
    right_line.recent_fit_pixels.append(right_line.current_fit_pixels)
    right_line.recent_fit_meters.append(right_line.current_fit_meters)

    # Update the radius of curvature
    left_line.radius_of_curvature_pixels = left_line.measure_curvature_pixels()
    left_line.radius_of_curvature_meters = left_line.measure_curvature_meters()
    right_line.radius_of_curvature_pixels = right_line.measure_curvature_pixels()
    right_line.radius_of_curvature_meters = right_line.measure_curvature_meters()

    # Sanity check with radius of curvature
    if left_line.radius_of_curvature_meters - right_line.radius_of_curvature_meters > 100:
        left_line.detected = False
        right_line.detected = False

    # # Update the base pose
    # left_line.line_base_pos = left_line.calculate_base_pose()
    # right_line.line_base_pos = right_line.calculate_base_pose()

    # # Sanity check with base pose
    # if np.abs(left_line.line_base_pos * xm_per_pix - right_line.line_base_pos *xm_per_pix) < 3:
    #     left_line.detected = False
    #     right_line.detected = False

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit_pixels[0]*ploty**2 + \
        left_fit_pixels[1]*ploty + left_fit_pixels[2]
    right_fitx = right_fit_pixels[0]*ploty**2 + \
        right_fit_pixels[1]*ploty + right_fit_pixels[2]

    # Update the fitted x values
    left_line.recent_xfitted.append(left_fitx)
    left_line.bestx = np.mean(left_line.recent_xfitted, axis=0)
    right_line.recent_xfitted.append(right_fitx)
    right_line.bestx = np.mean(right_line.recent_xfitted, axis=0)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    img_out = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(img_out)
    # Color in left and right line pixels
    img_out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    img_out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_line.bestx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_line.bestx+margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_line.bestx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx+margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    # img_out = cv2.addWeighted(img_out, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return left_line, right_line, img_out
