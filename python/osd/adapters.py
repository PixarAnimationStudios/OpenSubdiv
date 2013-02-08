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
