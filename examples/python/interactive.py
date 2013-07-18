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
import utility

from time import time
import math

def interactive():

    app = QtGui.QApplication(sys.argv)
    renderer = Renderer()
    win = Window(renderer)
    win.raise_()

    cage = [ 0.0, -1.414214, 1.0,  # 0
             1.414214, 0.0, 1.0,   # 1
             -1.414214, 0.0, 1.0,  # 2
             0.0, 1.414214, 1.0,   # 3
             -1.414214, 0.0, -1.0, # 4
             0.0, 1.414214, -1.0,  # 5
             0.0, -1.414214, -1.0, # 6
             1.414214, 0.0, -1.0 ] # 7
    
    cage = np.array(cage, np.float32).reshape((-1, 3))

    faces = [ (0,1,3,2),  # 0
              (2,3,5,4),  # 1
              (4,5,7,6),  # 2
              (6,7,1,0),  # 3
              (1,7,5,3),  # 4
              (6,0,2,4) ] # 5

    topo = osd.Topology(faces)

    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32)]

    def updateTopo(numLevels = 4):
        global subdivider
        topo.reset()
        topo.finalize()
        subdivider = osd.Subdivider(
            topo,
            vertexLayout = dtype,
            indexType = np.uint32,
            levels = numLevels)
        quads = subdivider.getRefinedQuads()
        renderer.updateIndicesVbo(quads)

    def updateCoarseVertices():
        global subdivider
        subdivider.setCoarseVertices(cage)
        subdivider.refine()
        pts = subdivider.getRefinedVertices()
        renderer.updatePointsVbo(pts)

    updateTopo()
    updateCoarseVertices()

    renderer.drawHook = updateCoarseVertices

    # Start an interactive session
    import code
    from time import time
    timer = QtCore.QTimer()
    code.interact(local=dict(globals(), **locals()))

    sys.exit(0)

if __name__ == '__main__':
    interactive()
