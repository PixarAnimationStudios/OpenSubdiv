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
