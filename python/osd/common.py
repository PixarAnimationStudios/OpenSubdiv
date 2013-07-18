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

class TopoError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class OsdTypeError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

# From hbr/mesh.h
class BoundaryMode:
    "Subdivision boundary rules for :class:`osd.Topology` construction."

    NONE = 0
    "Boundary edges are corner verts are not processed specially."

    EDGE_ONLY = 1
    "Boundary edges are marked infinitely sharp."

    EDGE_AND_CORNER = 2
    '''Boundary edges are marked infinitely sharp, and vertices with
    valence=2 are marked infinitely sharp.'''

    ALWAYS_SHARP = 3
    "Unused."

class Sharpness:
    '''Provides some floating-point constants that clients can
    optionally use to set sharpness.'''

    SMOOTH = 0
    "As smooth as possible."

    SHARP = 1
    "Moderately sharp."

    INFINITELY_SHARP = 2
    "As sharp as possible."
