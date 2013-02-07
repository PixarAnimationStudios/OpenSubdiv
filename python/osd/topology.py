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

from common import *
from adapters import *

import numpy as np
import shim
import itertools

class Topology(object):
    '''Represents an abstract graph of connected polygons.

    A :class:`Topology` object contains only connectivity information;
    it does not contain any coordinate data.  Instances are simply
    populated, finalized, and submitted to an instance of
    :class:`osd.Subdivider`.

    The constructor can take a single two-dimensional numpy array
    (``indices``), or two one-dimensional numpy arrays
    (``indices`` and ``valences``).  If every face has the same
    valence, clients can pass in a single integer for
    ``valences``.

    If desired, simple Python lists can be passed in rather than numpy
    arrays.  However they will get converted into numpy arrays
    internally.

    .. note:: Input data is always copied to internal storage,
       rather than referenced.

    :param indices: Defines each face as a list of vertex indices.
    :type indices: list or numpy array
    :param valences: If ``indices`` is 2D, this should be :const:`None`.  If every face has the same valence, this can be a single integer. Otherwise this is a list of integers that specify the valence of each face.
    :type valences: list, number, or numpy array
    '''

    def __init__(self, indices, valences = None):
        indices, valences = _flatten_args(indices, valences)
        valences = _process_valences(indices, valences)
        _check_topology(indices, valences)

        self.indices = np.array(indices, 'int32')
        self.valences = np.array(valences, 'uint8')
        self.shim = shim.Topology(self.indices, self.valences)
        self.boundaryMode = BoundaryMode.EDGE_ONLY
        self._vertexListAdapter = VertexListAdapter(self.shim)
        self._faceListAdapter = FaceListAdapter(self.shim)

    def reset(self):
        '''Un-finalizes the mesh to allow adjustment of sharpness and
        custom data.

        This is a costly operation since it effectively recreates the
        HBR mesh.
        '''
        topo = shim.Topology(self.indices, self.valences)
        topo.copyAnnotationsFrom(self.shim)
        self._vertexListAdapter = VertexListAdapter(topo)
        self._faceListAdapter = FaceListAdapter(topo)
        self.shim = topo

    @property
    def boundaryMode(self):
        '''Gets or sets the boundary interpolation method for this
        mesh to one of the values defined in
        :class:`osd.BoundaryMode`.
        '''
        return self.shim.getBoundaryMode()

    @boundaryMode.setter
    def boundaryMode(self, value):
        self.shim.setBoundaryMode(value)

    @property
    def vertices(self):
        '''Pythonic read/write access to special attributes (e.g., sharpness) in
        the HBR mesh.

        This property is not an actual list-of-things; it's a short-lived
        adapter that allows clients to use Pythonic properties::
 
            topo = osd.Topology(faces)
            topo.vertices[2].sharpness = 1.3

        '''
        return self._vertexListAdapter

    @property
    def faces(self):
        '''Pythonic read/write access to face data and edge data in the HBR mesh.

        This property is not an actual list-of-things; it's a short-lived
        adapter that allows clients to use Pythonic properties::

            topo = osd.Topology(faces)
            topo.faces[1].edges[0].sharpness = 0.6

        '''
        return self._faceListAdapter

    def finalize(self):
        '''Calls finish on the HBR mesh, thus preparing it for subdivision.'''
        self.shim.finalize()

    def __del__(self):
        pass

# Checks that no valence is less than 3, and that the sum of all valences
# equal to the # of indices.
def _check_topology(indices, valences):
    acc = 0
    for v in valences:
        if v < 3:
            raise TopoError("all valences must be 3 or greater")
        acc = acc + v
    if len(indices) != acc:
        msg = "sum of valences ({0}) isn't equal to the number of indices ({1})"
        raise TopoError(msg.format(acc, len(indices)))

# Given a list-of-lists, returns a list pair where the first list has
# values and the second list has the original counts.
def _flatten(faces):
    flattened = list(itertools.chain(*faces))
    lengths = [len(face) for face in faces]
    return flattened, lengths

# If indices is two-dimensional, splits it into two lists.
# Otherwise returns the lists unchanged.
def _flatten_args(indices, valences):
    try:
        flattened, lengths = _flatten(indices)
        if valences is not None:
            raise OsdTypeError(
               "valences must be None if indices is two-dimensional")
        return (flattened, lengths)
    except TypeError:
        if valences is None:
            raise OsdTypeError(
                "valences must be provided if indices is one-dimensional")
        return (indices, valences)

# If valences is a scalar, returns a list of valences.
# Otherwise returns the original valence list.
def _process_valences(indices, valences):
    try:
        v = int(valences)
        faceCount = len(indices) / v
        if len(indices) % v is not 0:
            msg = "Scalar provided for valences argument ({0}) that " \
                "does evenly divide the number of indices ({1})"
            raise OsdTypeError(msg.format(len(indices), v))
        valences = [v] * faceCount
    except TypeError:
        pass
    return valences
