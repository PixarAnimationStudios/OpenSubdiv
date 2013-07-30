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

from osd import *
import shim, numpy

class Subdivider(object):
    '''Wraps a frozen :class:`osd.Topology` object and efficiently
    subdivides it into triangles.

    On creation, the :class:`Subdivider` is locked to an immutable,
    finalized :class:`osd.Topology` object.  However, actual
    subdivision computation can be requested repeatedly using dynamic
    per-vertex data.

    :param topo: Finalized mesh topology.
    :type topo: :class:`osd.Topology`
    :param vertexLayout: Describes the data structure composing each vertex.
    :type vertexLayout: numpy dtype_ object or short-hand string
    :param indexType: Integer type for the indices returned from `getRefinedTopology`.
    :type indexType: single numpy type_
    :param levels: Number of subdivisions.
    :type levels: integer

    .. note:: In the current implementation, ``vertexLayout`` must be
       composed of ``numpy.float32``, and ``indexType`` must be
       ``numpy.uint32``.

    .. _type: http://docs.scipy.org/doc/numpy/user/basics.types.html
    .. _dtype: http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    '''

    def __init__(self, topology, vertexLayout, indexType, levels):
        if levels < 2:
            raise TopoError("Subdivision levels must be 2 or greater")
        if type(vertexLayout) != numpy.dtype:
            vertexLayout = numpy.dtype(vertexLayout)
        self.vertexLayout = vertexLayout
        self.indexType = indexType
        self.levels = levels
        self.shim = shim.Subdivider(topology.shim, vertexLayout, indexType, levels)

    # Calls UpdateData on the vertexBuffer.
    def setCoarseVertices(self, coarseVerts, listType = None):
        '''Pushes a new set of coarse verts to the mesh without
        changing topology.

        If a numpy array is supplied for ``coarseVerts``, its
        ``dtype`` must be castable (via view_) to the ``vertexLayout``
        of the :class:`Subdivider`.

        If a Python list is supplied for ``coarseVerts``, the client
        must also supply the ``listType`` argument to specify the
        numpy type of the incoming array before it gets cast to
        ``vertexLayout``.

        .. _view: http://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.ndarray.view.html
        '''
        if type(coarseVerts) is not numpy.ndarray:
            if not listType:
                raise TopoError("listType must be supplied")
            coarseVerts = numpy.array(coarseVerts, listType)
        coarseVerts = coarseVerts.view(self.vertexLayout)
        self.shim.setCoarseVertices(coarseVerts)

    # Calls Refine on the compute controller, passing it the compute
    # context and vertexBuffer.
    def refine(self):
        '''Performs the actual subdivision work.'''
        self.shim.refine()

    # Calls the strangely-named "BindCpuBuffer" on the
    # OsdCpuVertexBuffer to get back a float*
    def getRefinedVertices(self):
        '''Returns a numpy array representing the vertex data in the
        subdivided mesh.

        The data is returned in the format specified by the client
        when instancing the subdivider (``vertexLayout``).
        '''
        empty = numpy.empty(0, self.vertexLayout)
        return self.shim.getRefinedVertices(empty)

    def getRefinedQuads(self):
        '''Returns a numpy array representing the vertex indices of each quad in
        subdivided mesh.

        The data is returned in the format specified by the client
        when instancing the subdivider (``indexType``).
        '''
        empty = numpy.empty(0, self.indexType)
        return self.shim.getRefinedQuads(empty)

    def __del__(self):
        pass
