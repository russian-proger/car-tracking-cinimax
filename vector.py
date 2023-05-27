import math


class Vector:
    x: float
    y: float

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def norm(self):
        d = abs(self)
        self.x /= d
        self.y /= d
        return self

    def cross(self, other) -> float:
        return self.x * other.y - self.y * other.x

    def scalar(self, other) -> float:
        return self.x * other.x + self.y * other.y

    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other)