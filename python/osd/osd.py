class TopoError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class OsdTypeError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

# From hbr/mesh.h
class InterpolateBoundary:
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
