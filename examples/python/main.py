#!/usr/bin/env python
#
#     Copyright 2013 Pixar
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License
#     and the following modification to it: Section 6 Trademarks.
#     deleted and replaced with:
#
#     6. Trademarks. This License does not grant permission to use the
#     trade names, trademarks, service marks, or product names of the
#     Licensor and its affiliates, except as required for reproducing
#     the content of the NOTICE file.
#
#     You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing,
#     software distributed under the License is distributed on an
#     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
#     either express or implied.  See the License for the specific
#     language governing permissions and limitations under the
#     License.
#
from PyQt4 import QtGui, QtCore
from window import Window
from renderer import Renderer
import sys
import numpy as np
import osd

def main():

    app = QtGui.QApplication(sys.argv)
    renderer = Renderer()
    win = Window(renderer)
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

    topo = osd.Topology(faces)
    topo.boundaryMode = osd.BoundaryMode.EDGE_ONLY
    for v in (2, 3, 4, 5):
        topo.vertices[v].sharpness = 2.0
    for e in xrange(4):
        topo.faces[3].edges[e].sharpness = 3
    topo.finalize()

    subdivider = osd.Subdivider(
        topo,
        vertexLayout = 'f4, f4, f4',
        indexType = np.uint32,
        levels = 4)
    subdivider.setCoarseVertices(verts)
    subdivider.refine()
    inds = subdivider.getRefinedQuads()
    renderer.updateIndicesVbo(inds)

    def animateVerts():
        from time import time
        import math
        t = 4 * time()
        t = 0
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
        subdivider.setCoarseVertices(verts)
        subdivider.refine()
        pts = subdivider.getRefinedVertices()
        renderer.updatePointsVbo(pts)
    
    updateAnimation()
    renderer.drawHook = updateAnimation
    retcode = app.exec_()
    sys.exit(retcode)

if __name__ == '__main__':
    main()
