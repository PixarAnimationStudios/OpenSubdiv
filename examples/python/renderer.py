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

import math
import numpy as np

from OpenGL.GL import *
from shaders import *
from utility import *
from canvas import *

class Renderer:
    def __init__(self):
        self.indexCount = 0

    def draw(self):
        
        if hasattr(self, "drawHook"):
            self.drawHook()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self.indexCount:
            return

        theta = time() * math.pi / 2
        eye = V3(0, 0, 10)
        target = V3(0, 0, 0)
        up = V3(0, 1, 0)
        view = look_at(eye, target, up)

        model = np.identity(4, 'f')
        model = rotation(-1.1 + theta, [0,1,0])

        objToEye = view * model
        eyeToClip = perspective(15, self.aspect, 5, 200)
        normalM = np.identity(3, 'f')
        normalM[:3,:3] = objToEye[:3,:3]

        glUseProgram(self.programs['BareBones'])
        glUniformMatrix4fv(U("Projection"), 1, True, eyeToClip)
        glUniformMatrix4fv(U("Modelview"), 1, True, objToEye)
        glUniformMatrix3fv(U("NormalMatrix"), 1, True, normalM)
        glBindVertexArray(self.vao)

        # Since quads are deprecated we use LINES_ADJACENCY
        # simply because they're 4 verts per prim, and we can
        # expand them to triangles in the GS.
        glDrawElements(
            GL_LINES_ADJACENCY,
            self.indexCount,
            GL_UNSIGNED_INT,
            None)

    def resize(self, w, h):
        self.aspect = float(w) / float(h)
        glViewport(0, 0, w, h)

    def init(self):
        print glGetString(GL_VERSION)
        glClearColor(0.0, 0.25, 0.5, 1.0)
        self.programs = load_shaders()

        try:
            self.vao = glGenVertexArrays(1)
        except:
            import sys
            print "glGenVertexArrays isn't available, so you might need a newer version of PyOpenGL."
            sys.exit(1)

        self.pointsVbo = glGenBuffers(1)
        self.normalsVbo = None
        self.indicesVbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indicesVbo)

        glBindBuffer(GL_ARRAY_BUFFER, self.pointsVbo)
        glVertexAttribPointer(Attribs.POSITION, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(Attribs.POSITION)

        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)

    def updatePointsVbo(self, points):
        glBindBuffer(GL_ARRAY_BUFFER, self.pointsVbo)
        glBufferData(GL_ARRAY_BUFFER, points, GL_STATIC_DRAW)

    def updateNormalsVbo(self, normals):
        if not self.normalsVbo:
            self.normalsVbo = glGenBuffers(1)
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.normalsVbo)
            glVertexAttribPointer(Attribs.NORMAL, 3, GL_FLOAT, GL_FALSE, 12, None)
            glEnableVertexAttribArray(Attribs.NORMAL)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalsVbo)
        glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)

    def updateIndicesVbo(self, indices):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indicesVbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)
        self.indexCount = len(indices)
