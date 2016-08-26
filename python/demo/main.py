#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt4 import QtGui, QtCore
from window import Window
from demo import Demo
import sys
import numpy as np
import osd

def main():

    app = QtGui.QApplication(sys.argv)
    demo = Demo()
    win = Window(demo)
    win.raise_()

    verts = [ 0.0, -1.414214, 1.0,  # 0
              1.414214, 0.0, 1.0,   # 1
              -1.414214, 0.0, 1.0,  # 2
              0.0, 1.414214, 1.0,   # 3
              -1.414214, 0.0, -1.0, # 4
              0.0, 1.414214, -1.0,  # 5
              0.0, -1.414214, -1.0, # 6
              1.414214, 0.0, -1.0 ] # 7

    verts = np.array(verts, np.float32).reshape((-1, 3))

    faces = [ (0,1,3,2),  # 0
              (2,3,5,4),  # 1
              (4,5,7,6),  # 2
              (6,7,1,0),  # 3
              (1,7,5,3),  # 4
              (6,0,2,4) ] # 5

    dtype = [('Px', np.float32),
             ('Py', np.float32),
             ('Pz', np.float32)]

    topo = osd.Topology(faces)
    topo.boundaryInterpolation = osd.InterpolateBoundary.EDGE_ONLY
    topo.finalize()

    for v in (2, 3, 4, 5):
        topo.vertices[v].sharpness = 2.0

    for e in xrange(4):
        topo.faces[3].edges[e].sharpness = 3

    subdivider = osd.Subdivider(
        topo,
        vertexLayout = dtype,
        indexType = np.uint32,
        levels = 5)

    inds = subdivider.getRefinedTopology()
    demo.updateIndicesVbo(inds)

    def animateVerts():
        from time import time
        import math
        t = 4 * time()
        t = 0.5 + 0.5 * math.sin(t)
        t = 0.25 + t * 0.75
        a = np.array([ 0.0, -1.414214, 1.0])
        b = np.array([ 1.414214, 0.0, 1.0])
        c = np.array([ 0.0, -1.414214, -1.0])
        d = np.array([1.414214, 0.0, -1.0 ])
        center = (a + b + c + d) / 4
        center = np.multiply(center, 1-t)
        verts[0] = center + np.multiply(a, t)
        verts[1] = center + np.multiply(b, t)
        verts[6] = center + np.multiply(c, t)
        verts[7] = center + np.multiply(d, t)

    def updateAnimation():
        animateVerts()
        subdivider.setCage(verts)
        subdivider.refine()
        pts = subdivider.getRefinedVertices()
        demo.updatePointsVbo(pts)
    
    updateAnimation()
    demo.drawHook = updateAnimation

    retcode = app.exec_()
    sys.exit(retcode)

if __name__ == '__main__':
    main()
