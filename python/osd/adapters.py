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

# Helper classes that present mesh data as a Pythonic list-like
# objects with properties. In general we prefer clients to use numpy
# arrays for data. However, for OSD-specific data such as sharpness,
# we provide these adapter classes.

class VertexListAdapter(object):
    def __init__(self, shim):
        self.shim = shim
        self.length = shim.getNumVertices()
    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError("Index out of bounds")
        return _VertexAdapter(self.shim, index)
    def __len__(self):
        return self.length

class _VertexAdapter(object):
    def __init__(self, shim, index):
        self.shim = shim
        self.index = index
    @property
    def sharpness(self): 
        return self.shim.getVertexSharpness(
            self.index)
    @sharpness.setter
    def sharpness(self, value):
        self.shim.setVertexSharpness(
            self.index,
            value)

class FaceListAdapter(object):
    def __init__(self, shim):
        self.shim = shim
        self.length = shim.getNumFaces()
    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError("Index out of bounds")
        return _FaceAdapter(self.shim, index)
    def __len__(self):
        return self.length

class _FaceAdapter(object):
    def __init__(self, shim, faceIndex):
        self.shim = shim
        self.faceIndex = faceIndex
        self._edgeListAdapter = _EdgeListAdapter(self.shim, faceIndex)
    @property
    def hole(self):
        return self.shim.getFaceHole(self.faceIndex)
    @hole.setter
    def hole(self, value):
        self.shim.setFaceHole(self.faceIndex, value)
    @property
    def edges(self):
        return self._edgeListAdapter

class _EdgeListAdapter(object):
    def __init__(self, shim, faceIndex):
        self.shim = shim
        self.faceIndex = faceIndex
        self.length = shim.getNumEdges(faceIndex)
    def __getitem__(self, edgeIndex):
        if edgeIndex >= self.length:
            raise IndexError("Index out of bounds")
        return _EdgeAdapter(self.shim, self.faceIndex, edgeIndex)
    def __len__(self):
        return self.length

class _EdgeAdapter(object):
    def __init__(self, shim, faceIndex, edgeIndex):
        self.shim = shim
        self.faceIndex = faceIndex
        self.edgeIndex = edgeIndex
    @property
    def sharpness(self):
        return self.shim.getEdgeSharpness(
            self.faceIndex,
            self.edgeIndex)
    @sharpness.setter
    def sharpness(self, value):
        self.shim.setEdgeSharpness(
            self.faceIndex,
            self.edgeIndex,
            value)
