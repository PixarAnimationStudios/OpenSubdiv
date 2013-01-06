# Helper classes that present mesh data as a Pythonic list-like
# objects with properties. In general we prefer clients to use numpy
# arrays for data. However, for OSD-specific data such as sharpness,
# we provide these adapter classes.

import shim

class VertexListAdapter(object):
    def __init__(self, topo):
        self.topo = topo
    def __getitem__(self, index):
        return _VertexAdapter(self.topo, index)
    def __len__(self):
        return self.topo._maxIndex + 1

class _VertexAdapter(object):
    def __init__(self, topo, index):
        self.mesh = topo._hbr_mesh
        self.index = index
    @property
    def sharpness(self): 
        return shim.hbr_get_vertex_sharpness(
            self.mesh,
            self.index)
    @sharpness.setter
    def sharpness(self, value):
        shim.hbr_set_vertex_sharpness(
            self.mesh,
            self.index,
            value)

class FaceListAdapter(object):
    def __init__(self, topo):
        self.mesh = topo._hbr_mesh
        shim.hbr_update_faces(self.mesh)
        self.length = shim.hbr_get_num_faces(
            self.mesh)
    def __getitem__(self, index):
        return _FaceAdapter(self.mesh, index)
    def __len__(self):
        return self.length

class _FaceAdapter(object):
    def __init__(self, mesh, faceIndex):
        self.mesh = mesh
        self.faceIndex = faceIndex
        self._edgeListAdapter = _EdgeListAdapter(mesh, faceIndex)
    @property
    def hole(self):
        return shim.hbr_get_face_hole(
            self.mesh,
            self.faceIndex)
    @hole.setter
    def hole(self, value):
        shim.hbr_set_face_hole(
            self.mesh,
            self.faceIndex,
            value)
    @property
    def edges(self):
        return self._edgeListAdapter

class _EdgeListAdapter(object):
    def __init__(self, mesh, faceIndex):
        self.mesh = mesh
        self.faceIndex = faceIndex
        self.length = shim.hbr_get_num_edges(
            mesh,
            faceIndex)
    def __getitem__(self, edgeIndex):
        return _EdgeAdapter(self.mesh, self.faceIndex, edgeIndex)
    def __len__(self):
        return self.length

class _EdgeAdapter(object):
    def __init__(self, mesh, faceIndex, edgeIndex):
        self.mesh = mesh
        self.faceIndex = faceIndex
        self.edgeIndex = edgeIndex
    @property
    def sharpness(self):
        return shim.hbr_get_edge_sharpness(
            self.mesh,
            self.faceIndex,
            self.edgeIndex)
    @sharpness.setter
    def sharpness(self, value):
        shim.hbr_set_edge_sharpness(
            self.mesh,
            self.faceIndex,
            self.edgeIndex,
            value)
