from OpenGL.GL import *
from time import time
import math

import numpy as np
from numpy import linalg as LA

# Provide a terse way to get a uniform location from its name
def U(name):
    p = glGetIntegerv(GL_CURRENT_PROGRAM)
    return glGetUniformLocation(p, name)

# Provide a terse way to create a f32 numpy 3-tuple
def V3(x, y, z):
    return np.array([x, y, z], 'f')

def translation(direction):
    M = np.identity(4, 'f')
    M[:3, 3] = direction[:3]
    return M

def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation(angle, direction):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4, 'f')
    M[:3, :3] = R
    return M

def look_at(eye, target, up):
    F = target[:3] - eye[:3]
    f = F / LA.norm(F)
    U = up / LA.norm(up)
    s = np.cross(f, U)
    u = np.cross(s, f)
    M = np.matrix(np.identity(4))
    M[:3,:3] = [s,u,-f]
    T = translation(-eye)
    return np.matrix(M * T, 'f')

def perspective(fovy, aspect, f, n):
    s = 1.0/math.tan(math.radians(fovy)/2.0)
    sx, sy = s / aspect, s
    zz = (f+n)/(n-f)
    zw = 2*f*n/(n-f)
    m = np.matrix([[sx,0,0,0],
                   [0,sy,0,0],
                   [0,0,zz,zw],
                   [0,0,-1,0]], 'f')
    return m
