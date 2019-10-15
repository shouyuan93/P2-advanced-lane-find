## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
1. use glob.glob()to get all the folder of Chessboard imgs, transform to grayleval and use command cv2.findChessboardCorners() to get the imgpoints, define the objp
2. via command cv2.calibrateCamera(objp,imgp,img_size,None,None) get camera parameters
3. extract parameter 'mtx' and 'dist' and use command cv2.unditort to correct the distortion.

example_Img stored at folder'output_images/undistorted.png' 


### Pipeline (single images)

#### 1.  correct the image distortion. 
1, get the parameters 'cameraMatrix'and'distorsionCoeff' from step Calibration and use command cv2.undistort(img,mtx,dist,None,mtx) to correct img

example_Img stored at folder'output-images/test1_corrected.jpg'

#### 2. transform the img to the overlook pespective
1, def 4 point(src) from undistorted_img, and def 4 point(dst) in overlook img,
get the matrix of the transform and of the inverse transform.
2, via command cv2.warpPerspective(img,M,img_size,flags) change the img perspectiv

example_Img stored at folder'output_images/test2_warped.jpg'



#### 3. use hls_s channel and sobel x direction to generate binary_map

1, because of the lane line mostly in vertical direction extend, so we take the use sobel matrix in x direction to detect the gradient of graylevel. via command:
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
the output should be ein binarymap, wtih the same size(h,w) of grayimg. the pixel will be detected as line when, the treshold_min < gradient <=threshold_max,
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
2, hls_s_channel it robust for the light condition, so we take one more binarymap in s_channel, first convert the img from RGB to HlS and the compare with treshold.
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    binary_output = np.zeros_like(hls[:,:,2])
    binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    
3, combine the binarymap of sobel x direction and of s channel
    out=np.zeros_like(binary_x)
    out[(binary_x==1)|(binary_s==1)]=1
    
example_img stored at folder'output_images/final_binary.jpg'


#### 4. sliding window to detect lane point and fit a 2 order polynomial

1, get one histogram of the bottom half of the image, where the peak are, is the lane,
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
2, via command np.argmax, get the x position of the peak, we set this is the midpoint of first two windows
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
3, we set there are 9 row windows pro image,so the height of window should be np.int(img.shape[0]/9), we set the margin of window be 80, so we get the left and right limit

4, command np.nonzero() return 2 array, the fist array is the column daten and second array is the row daten. set a for-loop, to iterativ update window and detect the points.just the point in the window will be detected, and in base of new points, update the midpoints for the next window. 
    for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 3)
            rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
5, left_lane_inds, and right_lane_inds are the list of point indexes in array nonzero[0] and nonzero[1], which point are belong to the lane, we get the point indexes of binary_map via nonzero[0][_lane_inds] and nonzero[1][_lane_inds], then fit a 2 order polynomial.

Demo_Img stored at folder'output_images/demo_windows.jpg'



#### 5. calculate the radius of curvature of the lane and the position of the vehicle with respect to center and draw back onto the road.
 
1, in order to get the radius of curvature, need measuring scale, meter per pixel in x and y dimension, the polynomialCoeff,  so, in basis of formular, we can get the radius of left and right lane line

2, the vehicle should located in the mid of img, and the difference between the img mid and lane mid, is the distance between vehicle and lane mid

3, via array stack get the left and right lane line and the lane area, because of the sequence of 4 point of rectangle, pts_right should be reverse order. draw all this of ein copy of undistorted img
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
4, via command cv2.warpPerspectiv() and inverse Transform Matrix, transform the img to original perspectiv, and through command cv2.addWeighted() to take weight add with original undistorted img

Example_img stored at folder'output_images/test*_out.jpg'


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

output video are stored, with name'project_video_out.mp4'(./project_video.mp4)
 
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1, maybe should take a shorter detect area, because for big curvature 2 order cannot fit polynomial good magnitude
2, maybe one other way to generate binary map, because im challenge video detect false lane
3, didt use the fit paras of previous frame detection to tracking lane,
