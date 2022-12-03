import numpy as np
import cv2
import os

if not os.path.exists('output'):
    os.makedirs('output')

# load video from file
cap = cv2.VideoCapture('./input/drone_fpv.mp4')

# create output video with same height and width as input
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('./output/drone_sparse_flow.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                      (frame_width, frame_height))

# set algorithm parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=1000,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# set track parameters
track_len = 15
detect_interval = 5
tracks = []
frame_idx = 0

# loop of all video frames
while True:

    # read next frame in video
    _ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    # if there are existing feature tracks try to extend them into the current frame
    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)

        # perform a forward and backward pass to ensure the features are of high quality
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        tracks = new_tracks
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

    # add new features at specified frame interval
    if frame_idx % detect_interval == 0:
        # find features where there aren't existing features by masking the image
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray
    cv2.imshow('lk_track', vis)

    out.write(vis)

    ch = cv2.waitKey(1)
    if ch == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
