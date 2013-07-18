#!/usr/bin/env python
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

import numpy as np
import unittest, sys
import osd

# Topology of a cube.
faces = [ (0,1,3,2),
          (2,3,5,4),
          (4,5,7,6),
          (6,7,1,0),
          (1,7,5,3),
          (6,0,2,4) ]

# Vertex positions and "temperature" as an example of a custom
# attribute.
verts = [ 0.000000, -1.414214,  1.000000, 71,
          1.414214,  0.000000,  1.000000, 82,
         -1.414214,  0.000000,  1.000000, 95,
          0.000000,  1.414214,  1.000000, 100,
         -1.414214,  0.000000, -1.000000, 63,
          0.000000,  1.414214, -1.000000, 77,
          0.000000, -1.414214, -1.000000, 82,
          1.414214,  0.000000, -1.000000, 32 ]

dtype = [('Px', np.float32),
         ('Py', np.float32),
         ('Pz', np.float32),
         ('temperature', np.float32)]

class SimpleTest(unittest.TestCase):

    def test_usage(self):
        mesh = osd.Topology(faces)
        mesh.boundaryMode = osd.BoundaryMode.EDGE_ONLY

        mesh.vertices[0].sharpness = 2.7
        self.assertAlmostEqual(mesh.vertices[0].sharpness, 2.7)

        self.assertEqual(len(mesh.vertices), len(verts) / len(dtype))
        self.assertEqual(len(mesh.faces), len(faces))
        self.assertEqual(len(mesh.faces[0].edges), len(faces[0]))

        mesh.finalize()

        subdivider = osd.Subdivider(
            mesh,
            vertexLayout = dtype,
            indexType = np.uint32,
            levels = 4)

        subdivider.setCoarseVertices(verts, np.float32)
        subdivider.refine()

        numQuads = len(subdivider.getRefinedQuads()) / 4
        numVerts = len(subdivider.getRefinedVertices()) / len(dtype)

        self.assertEqual(numQuads, 1536, "Unexpected number of refined quads")
        self.assertEqual(numVerts, 2056, "Unexpected number of refined verts")

    # For now, disable the leak test by prepending "do_not_".
    def do_not_test_leaks(self):
        self.test_usage()
        start = _get_heap_usage()
        history = []
        for i in xrange(1024):
            self.test_usage()
            if ((i+1) % 256) == 0:
                history.append(_get_heap_usage() - start)
                print str(history[-1]) + "...",
                sys.stdout.flush()
        print
        total = 0
        for i in xrange(1, len(history)):
            delta = history[i] - history[i - 1]
            if delta <= 0:
                return
            total = total + delta
        avg = total / (len(history) - 1)
        self.fail("Memory usage is strictly increasing ({0}).".format(avg))

    def test_Topology_creation(self):

        # Input data
        indices, valences = _flatten(faces)

        # Native list-of-lists, constant valence:
        mesh = osd.Topology(faces)
        self.assert_(mesh,
                     "Unable to construct Topology object from a list-of-lists")

        # Native list, constant valence:
        mesh = osd.Topology(indices, 4)
        self.assert_(mesh,
                     "Unable to construct Topology object from a list")

        # Native list-of-lists, variable valence:
        faces2 = faces + [(8,9,10)]
        mesh = osd.Topology(faces2)
        self.assert_(mesh,
                     "Unable to construct Topology object from a list of "
                     "variable-sized lists")

        # Two-dimensional numpy array:
        numpyFaces = np.array(indices, 'uint16').reshape(-1, 4)
        mesh = osd.Topology(numpyFaces)
        self.assert_(mesh,
                     "Unable to construct Topology object from a "
                     "two-dimensional numpy array")

        # Native index list and valence list:
        mesh = osd.Topology(indices, valences)
        self.assert_(mesh)
 
        # Numpy index list and valence list:
        indices = np.array(indices, 'uint16')
        valences = np.array(valences, 'uint8')
        mesh = osd.Topology(indices, valences)
        self.assert_(mesh)

        # Ensure various topology checks
        self.assertRaises(osd.OsdTypeError, osd.Topology, indices, None)
        self.assertRaises(osd.OsdTypeError, osd.Topology, faces, valences)
        faces2 = faces + [(8,9)]
        self.assertRaises(osd.TopoError, osd.Topology, faces2)
        valences2 = valences + [3]
        self.assertRaises(osd.TopoError, osd.Topology, indices, valences2)

def _get_heap_usage():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def _flatten(faces):
    import itertools
    flattened = list(itertools.chain(*faces))
    lengths = [len(face) for face in faces]
    return flattened, lengths

if __name__ == "__main__":
    unittest.main()
