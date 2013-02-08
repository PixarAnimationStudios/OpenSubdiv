.. OpenSubdiv documentation master file, created by
   sphinx-quickstart on Wed Nov 21 15:45:14 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview and Design Philosophy
=======

The Python module for OpenSubdiv does not provide one-to-one wrapping of the native C++ APIs: Osd, Hbr, and Far.  Rather, the ``osd`` module is a higher-level utility layer composed of a small handful of classes and free functions.

We do not yet support rendering or GPU-accelerated subdivision from Python, but a demo is provided that renders a subdivided surface using ``PyOpenGL`` and ``QGLWidget``.  The demo uses a "modern" OpenGL context (Core Profile).

These bindings leverage numpy_ arrays for passing data.  The numpy library is the de facto standard for encapsulating large swaths of typed data in Python.  However, for special attributes (such as sharpness), the OpenSubdiv wrapper exposes Pythonic interfaces, using properties and list accessors.  For example::

    import osd
    import numpy as np
    faceList = np.array([[0,1,2,3],[0,1,6,7]])
    topo = osd.Topology(faceList)
    topo.vertices[2].sharpness = 1.3
    topo.faces[0].edges[3].sharpness = 0.6

After constructing a :class:`osd.Topology` object, simply finalize it and pass it into a :class:`osd.Subdivider` instance::

    topo.finalize()
    subdivider = osd.Subdivider(
        topology = topo,
        vertexLayout = np.dtype('f4, f4, f4'),
        indexType = np.uint32,
        levels = 4)

The final step is to perform actual refinement.  This often occurs inside an animation loop or callback function::

    subdivider.setCoarseVertices(positions)
    subdivider.refine()
    pts = subdivider.getRefinedVertices()
    
Only uniform subdivision is supported from Python, which means the topology of the subdivided mesh will never change::

    indices = subdivider.getRefinedQuads()

This returns a flat list of indices (four per quad) using the integer type that was specified as the ``indexType`` argument in the constructor.

.. _numpy: http://www.numpy.org

Topology Class
==============

.. autoclass:: osd.Topology
   :members:

Subdivider Class
==============

.. autoclass:: osd.Subdivider
   :members:

Enumerations
==============

.. autoclass:: osd.BoundaryMode
   :members:

.. autoclass:: osd.Sharpness
   :members:

Caveats
=======

- Hierarchical edits are not yet supported
- Face varying data is not yet supported
- Feature adaptive subdivision is not yet supported
- CUDA and GLSL tessellation shaders are not yet supported
- Only the Catmull-Clark scheme is supported (no loop or linear schemes)
- Vertex data must be 32-bit floats although API is in place to accept heterogenous interpolation data
- Index data must be 32-bit unsigned integers although API is in place to accept other types
