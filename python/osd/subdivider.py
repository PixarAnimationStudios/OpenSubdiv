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
    :type vertexLayout: list of numpy types_
    :param indexType: Integer type for the indices returned from `getRefinedTopology`.
    :type indexType: single numpy type_
    :param levels: Number of subdivisions.
    :type levels: integer

    .. note:: In the current implementation, ``vertexLayout`` must be
       composed of ``numpy.float32``, and ``indexType`` must be
       ``numpy.uint32``.

    .. _types: http://docs.scipy.org/doc/numpy/user/basics.types.html
    .. _type: http://docs.scipy.org/doc/numpy/user/basics.types.html
    '''

    def __init__(self, topo, vertexLayout, indexType, levels):
        if levels < 2:
            raise TopoError("Subdivision levels must be 2 or greater")
        self.vertexLayout = vertexLayout
        self.indexType = indexType
        self.level = levels
        self._csubd = shim.csubd_new(self, topo)

    # Calls UpdateData on the vertexBuffer.
    def setCage(self, coarseVerts, listType = None):
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
            coarseVerts = numpy.array(coarseVerts, listType)
        coarseVerts = coarseVerts.view(self.vertexLayout)
        shim.csubd_update(self._csubd, coarseVerts)
        pass

    # Calls Refine on the compute controller, passing it the compute
    # context and vertexBuffer.
    def refine(self):
        '''Performs the actual subdivision work.'''
        shim.csubd_refine(self._csubd)

    # Calls the strangely-named "BindCpuBuffer" on the
    # OsdCpuVertexBuffer to get back a float*
    def getRefinedVertices(self):
        '''Returns a numpy array representing the vertex data in the
        subdivided mesh.

        The data is returned in the format specified by the client
        when instancing the subdivider (``vertexLayout``).
        '''
        return shim.csubd_getverts(self._csubd)

    def getRefinedTopology(self):
        '''Returns a numpy array representing the vertex indices of each quad in
        subdivided mesh.

        The data is returned in the format specified by the client
        when instancing the subdivider (``indexType``).
        '''
        if not hasattr(self, "_quads"):
            self._quads = shim.csubd_getquads(self._csubd, self.indexType)
        return self._quads

    def __del__(self):
        if hasattr(self, "_far_mesh"):
            shim.csubd_delete(self._far_mesh)
