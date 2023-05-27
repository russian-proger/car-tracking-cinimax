import math

import numpy as np
import cv2 as cv

from vector import Vector


gid: int = 0

def gen_id() -> int:
    global gid
    gid += 1
    return gid

class DetObject:
    pts: list[Vector]
    pos: Vector
    penalty: int
    end: int

    def __init__(self, pos: Vector):
        self.penalty = 0
        self.pos = pos
        self.pts = [pos]
        self.end = 0

    def can(self, point: Vector, max_dist: float, direction: Vector, ) -> bool:
        delta = self.pos - point
        dist: float = math.hypot(delta.x, delta.y)
        if dist <= max_dist:
            return True

        return False

    def update(self, point: Vector):
        self.penalty = 0
        self.pts.append(point)
        pos = point

    def finish(self, direction: Vector, min_obj_ticks: int) -> bool:
        if len(self.pts) >= min_obj_ticks:
            return True
        return False


class MedianTracker:
    tick: int
    objects: list[DetObject]
    direction: Vector
    max_dist: float
    archive: list[DetObject]

    def __init__(self, direction: Vector = Vector(0,0), max_dist: float=30, min_obj_ticks: int=20):
        self.direction = direction
        self.objects = []
        self.archive = []
        self.tick = 0
        self.max_dist = max_dist
        self.min_obj_ticks = min_obj_ticks

    def update(self, frame, upd_points: list[Vector]) -> None:
        self.tick += 1
        new_objects: list[DetObject] = []

        n = len(upd_points)
        used = [False for i in range(n)]

        for obj in self.objects:
            for i in range(len(upd_points)):
                point = upd_points[i]
                if obj.can(point, self.max_dist, self.direction):
                    used[i] = True

        for obj in self.objects:
            for i in range(len(upd_points)):
                point = upd_points[i]
                if obj.can(point, self.max_dist, self.direction):
                    new_objects.append(obj)
                    obj.update(point)
                    used[i] = True
                    break

            else:
                obj.penalty += 1
                if obj.penalty >= 10:
                    obj.end = self.tick
                    if obj.finish(self.direction, self.min_obj_ticks):
                        self.archive.append(obj)
                else:
                    new_objects.append(obj)

        for i in range(n):
            if not used[i]:
                new_objects.append(DetObject(upd_points[i]))

        self.objects = new_objects
        # print(list(map(lambda x: [x.x, x.y], upd_points)))
        # print(list(map(lambda x: [x.pos.x, x.pos.y], self.objects)))
        # print(len(self.archive))
