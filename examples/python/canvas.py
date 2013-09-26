#
#   Copyright 2013 Pixar
#
#   Licensed under the Apache License, Version 2.0 (the "Apache License")
#   with the following modification; you may not use this file except in
#   compliance with the Apache License and the following modification to it:
#   Section 6. Trademarks. is deleted and replaced with:
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor
#      and its affiliates, except as required to comply with Section 4(c) of
#      the License and to reproduce the content of the NOTICE file.
#
#   You may obtain a copy of the Apache License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the Apache License with the above modification is
#   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied. See the Apache License for the specific
#   language governing permissions and limitations under the Apache License.
#

from PyQt4 import QtCore
from PyQt4.QtOpenGL import *
from OpenGL.GL import *
from time import time

# Set this to 'None' to refresh as rapidly as possible
# (assuming that vsync is disabled.)
ThrottleFps = 60

class Canvas(QGLWidget):
    def __init__(self, parent, client):
        self.client = client
        f = QGLFormat(QGL.SampleBuffers)
        if hasattr(QGLFormat, 'setVersion'):
            f.setVersion(3, 2)
            f.setProfile(QGLFormat.CoreProfile)
        else: 
            pass
        if f.sampleBuffers():
            f.setSamples(16)
        c = QGLContext(f, None)
        QGLWidget.__init__(self, c, parent) 
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateGL)
        interval = 1000.0 / ThrottleFps if ThrottleFps else 0
        self.timer.start(interval)
        self.setMinimumSize(500, 500)

    def paintGL(self):
        self.client.draw()

    def updateGL(self):
        self.client.draw()
        self.update()

    def resizeGL(self, w, h):
        self.client.resize(w, h)

    def initializeGL(self):
        self.client.init()
