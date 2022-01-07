import numpy as np


def radians2degrees(list_of_angles: np.array or list) -> np.array:
    radians2degrees = lambda angle: 180 * angle / np.pi
    list_of_angles = radians2degrees(np.array(list_of_angles))
    list_of_angles = np.round(list_of_angles, 1)
    return list_of_angles


def degrees2radians(list_of_angles: np.array or list) -> np.array:
    degrees2radians = lambda angle: np.pi * angle / 180
    list_of_angles = degrees2radians(np.array(list_of_angles))
    list_of_angles = np.round(list_of_angles, 1)
    return list_of_angles
