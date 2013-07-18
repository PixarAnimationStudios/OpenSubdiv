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
