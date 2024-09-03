from enum import Enum


class Plane(Enum):
    IN_PLANE = 0
    OO_PLANE = 1


def str2plane(s):
    if s.lower() in ("ip", "in", "i", "in plane"):
        return Plane.IN_PLANE
    if s.lower() in ("op", "oop", "out", "o", "out of plane"):
        return Plane.OO_PLANE
    raise Exception(f"Could not parse plane descriptor '{s}'")


def plane2str(p):
    if p == Plane.IN_PLANE:
        return "ip"
    if p == Plane.OO_PLANE:
        return "op"