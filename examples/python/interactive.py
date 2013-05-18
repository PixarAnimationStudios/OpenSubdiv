#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
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
