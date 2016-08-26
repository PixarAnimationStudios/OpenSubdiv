import math
import numpy as np

from OpenGL.GL import *
from shaders import *
from utility import *
from canvas import *

class Demo:
    def __init__(self):
        self.indexCount = 0

    def draw(self):
        
        if hasattr(self, "drawHook"):
            self.drawHook()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self.indexCount:
            return

        theta = time() * math.pi
        eye = V3(.1, -0.1, 10)
        target = V3(.1, -0.1, 0)
        up = V3(0, 1, 0)
        view = look_at(eye, target, up)

        model = np.identity(4, 'f')
        model = rotation(-1.1, [0,1,0])

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
