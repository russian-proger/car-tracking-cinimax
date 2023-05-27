import argparse
import os
import random
import time

import numpy as np
import cv2 as cv

from tracker import MedianTracker
from vector import Vector

parser = argparse.ArgumentParser()
parser.add_argument('filename', metavar='FILE', type=str)
parser.add_argument('-i', '--interactive', action='store_true')
parser.add_argument('-o', required=False, dest='destination')
args = parser.parse_args()

if not os.path.exists(args.filename):
    print("Error while reading file", args.filename)
    exit(-1)

interactive = args.interactive
destination = (args.filename if args.filename.count('.') == 0 else ".".join(args.filename.split('.')[:-1])) + ".out"
if args.destination is not None:
    destination = args.destination


cap = cv.VideoCapture(args.filename)

subtractors = [
    cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=150),
    cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=900),
    cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=200)
]

min_area = [700, 3500, 700]

framesBounds = [
    [[147,150], [273,275], [400,139], [250,80]],
    [[651,675], [755,720], [959,720], [759,570]],
    [[810,263], [904,191], [984,218], [911,294]]
]

trackers = [
    MedianTracker(Vector(0,0), 30, 20),
    MedianTracker(Vector(0,0), 70, 10),
    MedianTracker(Vector(0,0), 30, 5)
]

def randomColor():
    c = [random.randint(0,255) for i in range(3)]
    return (c[0], c[1], c[2])

frame_rate = 25
ticks = 0

while True:
    ok, frame = cap.read()

    ticks += 1
    if ticks % (frame_rate * 60) == 0:
        print(ticks // (frame_rate * 60), "min")

    if not ok:
        break


    for i in range(3):
        window_name = "Out " + str(i+1)
        frameBounds = framesBounds[i]

        x_min = min(map(lambda l : l[0], frameBounds))
        x_max = max(map(lambda l : l[0], frameBounds))
        y_min = min(map(lambda l : l[1], frameBounds))
        y_max = max(map(lambda l : l[1], frameBounds))

        w = x_max - x_min
        h = y_max - y_min

        polygons = map(lambda l : np.array(l), [
            [[bound[0] - x_min, bound[1] - y_min] for bound in frameBounds],
            [[0, 0], [w,0], [w,h], [0,h]],
        ])

        framePart = frame[y_min:y_max, x_min:x_max]

        img = cv.fillPoly(np.copy(framePart), list(polygons), (255,255,255))
        smooth = img
        diff = subtractors[i].apply(smooth)
        _, diff = cv.threshold(diff, 254, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(diff, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        median_points = []

        for c in contours:
            area = cv.contourArea(c)
            if area > min_area[i]:
                if interactive:
                    hull = cv.convexHull(c)
                    cv.polylines(framePart, [hull], True, randomColor(), 2)

                    box = cv.boxPoints(cv.minAreaRect(c))
                    box = np.intp(box)
                    box = cv.boundingRect(c)

                    cv.rectangle(framePart, [box[0], box[1]], [box[0] + box[2], box[1] + box[3]], (0,255,0), 2)


                box = cv.boundingRect(c)
                point = Vector((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                median_points.append(point)


        tracker = trackers[i]

        tracker.update(framePart, median_points)
        if interactive:
            cv.imshow(window_name, framePart)


    if interactive:
        cv.imshow("Frame", frame)

    key = 0
    if interactive:
        key = cv.waitKey(1)

    # Quit
    if key == ord('q'):
        break

    # Pause
    elif key == ord(' '):
        while cv.waitKey(30) != ord(' '):
            continue

cap.release()
cv.destroyAllWindows()


# Global
timepoints = []
for tracker in trackers:
    for obj in tracker.archive:
        timepoints.append(obj.end / frame_rate)

timepoints.sort()
fout = open(destination, "w")
fout.write(" ".join([str(i) for i in timepoints]) + '\n')

# Local
for i in range(3):
    tracker = trackers[i]
    timepoints = list(map(lambda x: x.end / frame_rate, tracker.archive))

    fout.write(" ".join([str(i) for i in timepoints]) + '\n')

fout.close()
