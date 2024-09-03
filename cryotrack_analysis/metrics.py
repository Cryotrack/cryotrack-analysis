import numpy as np


def euclidean_error(tip_point, target_point):
    return np.linalg.norm(tip_point - target_point)


def lateral_error(target_point, entry_point, tip_point):
    a = target_point
    b = entry_point
    c = tip_point
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.linalg.norm(ba) * np.sin(angle)
