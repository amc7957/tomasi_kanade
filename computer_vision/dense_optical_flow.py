import cv2
import numpy as np
import os

if not os.path.exists('output'):
    os.makedirs('output')

# load video from file
cap = cv2.VideoCapture("./input/drone_fpv.mp4")

# create output video with same height and double width to allow for side-by-side images (unprocessed/processed)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('./output/drone_dense_flow.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width * 2,
                                                                                                        frame_height))

# get the first frame, grayscale, and store as the previous for optical flow calculation
ret, frame1 = cap.read()
img_previous = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# output will use an HSV image, but the S channel will be constant
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# loop over all video frames
while 1:

    # get the current frame and greyscale
    ret, frame2 = cap.read()
    img_next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(img_previous, img_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # transform the optical flow vectors to magnitude and direction
    # set H channel to the direction and V channel to the magnitude
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(1) & 0xff

    # place the original and processed video frames side-by-side and write to video
    display = np.concatenate((frame2, rgb), axis=1)
    out.write(display)

    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('./output/opticalfb.png', frame2)
        cv2.imwrite('./output/opticalhsv.png', rgb)
    img_previous = img_next

out.release()
cap.release()
cv2.destroyAllWindows()
