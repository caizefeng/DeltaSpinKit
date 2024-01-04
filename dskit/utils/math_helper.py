#!/usr/bin/env python3
# @File    : math_helper.py
# @Time    : 9/5/2021 9:52 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import math

import numpy as np


def jacobi(r, theta, phi):
    return np.array([[math.sin(theta) * math.cos(phi),
                      r * math.cos(theta) * math.cos(phi), - r * math.sin(theta) * math.sin(phi)],
                     [math.sin(theta) * math.sin(phi), r * math.cos(theta) * math.sin(phi),
                      r * math.sin(theta) * math.cos(phi)],
                     [math.cos(theta), -r * math.sin(theta), 0]])


def sphe2cart(r, theta, phi):
    return np.array([r * math.sin(theta) * math.cos(phi),
                     r * math.sin(theta) * math.sin(phi),
                     r * math.cos(theta)])


# change the range of atan2 to (0, 2pi]
def atan2_2pi(y, x):
    convention = math.atan2(y, x)
    result = convention if convention > 0 else convention + 2 * math.pi
    return result


def cart2sphe(x, y, z):  # return r, theta, phi
    return np.array([math.sqrt(x ** 2 + y ** 2 + z ** 2), math.atan2(math.sqrt(x ** 2 + y ** 2), z), atan2_2pi(y, x)])


def general_spiral_parametric_equation(q_dot_r, theta, phi):
    return [math.sin(theta) * math.cos(q_dot_r + phi), math.sin(theta) * math.sin(q_dot_r + phi), math.cos(theta)]


# def cycloid_parametric_equation(q_dot_r, theta, phi):
#     theta_old = q_dot_r + theta
#     theta_reduce = theta_old - theta_old // (2 * math.pi) * (2 * math.pi)
#     # theta larger than pi, transformation needed
#     if theta_reduce > math.pi:
#         theta_new = 2 * math.pi - theta_reduce
#         phi_new = math.pi + phi
#     else:
#         theta_new = theta_reduce
#         phi_new = phi
#     return [math.cos(phi_new) * math.sin(theta_new), math.sin(phi_new) * math.sin(theta_new), math.cos(theta_new)]


def almost_zero(v, atol=1e-08):
    """
    Test if v is almost the zero vector.
    """
    return np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=atol)


def almost_collinear(v1, v2, atol=1e-08):
    """
    Test if `v1` and `v2` are almost collinear.
    This will return true if either `v1` or `v2` is the zero vector, because
    mathematically speaking, the zero vector is collinear to everything.
    Geometrically that doesn't necessarily make sense, so if you want to handle
    zero vectors specially, you can test your inputs with `vg.almost_zero()`.
    """
    if almost_zero(v1) or almost_zero(v2):
        raise ValueError("inputs of `almost_collinear` cannot be zero vectors")
    cross = np.cross(normalize(v1), normalize(v2))
    norm = np.linalg.norm(cross)
    return np.isclose(norm, 0.0, rtol=0, atol=atol)


def normalize(vector):
    """
    Return the vector, normalized.
    If vector is 2d, treats it as stacked vectors, and normalizes each one.
    """
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector)
    elif vector.ndim == 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        raise ValueError(str(vector))


def normalize_to_norm(vector, norm):
    """
    Return the vector, normalized.
    If vector is 2d, treats it as stacked vectors, and normalizes each one.
    """
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector) * norm
    elif vector.ndim == 2:
        for idx, arr in enumerate(vector):
            if not almost_zero(arr):
                vector[idx] = arr / np.linalg.norm(arr) * norm
        return vector
    else:
        raise ValueError(str(vector))


def included_angle(vector_1, vector_2):
    return np.arccos(vector_1 @ vector_2 / np.linalg.norm(vector_1) / np.linalg.norm(vector_2))
