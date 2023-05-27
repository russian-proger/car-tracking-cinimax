import random
import time

import numpy as np
import cv2 as cv

cap = cv.VideoCapture("2305231807.ts")

# object_detector = cv.createBackgroundSubtractorMOG2(detectShadows=True, history=30, varThreshold=50)
object_detector = cv.createBackgroundSubtractorKNN(history=30000, dist2Threshold=500, detectShadows=True)

ok, frame = cap.read()

parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.5, minDistance=150)

# convert to grayscale
frame_gray_init = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Use Shi-Tomasi to detect object corners / edges from initial frame
edges = cv.goodFeaturesToTrack(frame_gray_init, mask = None, **parameters_shitomasi)

# create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (100, 3))

# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    # get next frame
    ok, frame = cap.read()
    if not ok:
        print("[INFO] end of file reached")
        break
    # prepare grayscale image
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # update object corners by comparing with found edges in initial frame
    update_edges, status, errors = cv.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None,
                                                         **parameter_lucas_kanade)
    # only update edges if algorithm successfully tracked
    new_edges = update_edges[status == 1]
    # to calculate directional flow we need to compare with previous position
    old_edges = edges[status == 1]

    for i, (new, old) in enumerate(zip(new_edges, old_edges)):
        a, b = new.ravel()
        c, d = old.ravel()

        # draw line between old and new corner point with random colour
        mask = cv.line(canvas, (int(a), int(b)), (int(c), int(d)), colours[i].tolist(), 2)
        # draw circle around new position
        frame = cv.circle(frame, (int(a), int(b)), 5, colours[i].tolist(), -1)

    result = cv.add(frame, mask)
    cv.imshow('Optical Flow (sparse)', result)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # overwrite initial frame with current before restarting the loop
    frame_gray_init = frame_gray.copy()
    # update to new edges before restarting the loop
    edges = new_edges.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()