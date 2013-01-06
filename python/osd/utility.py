import numpy as np
import numpy.linalg as linalg
from itertools import izip

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return izip(*[iter(iterable)]*n)

def repl():
    "For debugging, breaks to an interactive Python session."
    import code, inspect
    frame = inspect.currentframe()
    myLocals = locals()
    callersLocals = frame.f_back.f_locals
    code.interact(local=dict(globals(), **callersLocals))

def dumpMesh(name, subd):
    "Dumps refined mesh data to a pair of raw binary files"

    quads = subd.getRefinedTopology()
    if quads.max() >= (2 ** 16):
        indexType = 'uint32'
    else:
        indexType = 'uint16'
    quads.astype(indexType).tofile(name + ".quads")    

    positions = subd.getRefinedVertices()
    positions.astype('float32').tofile(name + ".positions")    

def computeSmoothNormals(coords, quads):
    "Returns a list of normals, whose length is the same as coords."

    if quads.dtype != np.uint32 or coords.dtype != np.float32:
        raise OsdTypeError("Only uint32 indices and float coords are supported")

    if (len(quads) % 4) or (len(coords) % 3):
        raise OsdTypeError("Only quads and 3D coords are supported")

    print "Computing normals..."

    quads = quads.reshape((-1,4))
    coords = coords.reshape((-1,3))
    vertexToQuads = [[] for c in coords]
    quadNormals = np.empty((len(quads),3), np.float32)
    vertexNormals = np.empty((len(coords),3), np.float32)

    for quadIndex, q in enumerate(quads):
        vertexToQuads[q[0]].append(quadIndex)
        vertexToQuads[q[1]].append(quadIndex)
        vertexToQuads[q[2]].append(quadIndex)
        vertexToQuads[q[3]].append(quadIndex)
        a, b, c = coords[q[0]], coords[q[1]], coords[q[2]]
        ab = np.subtract(b, a)
        ac = np.subtract(c, a)
        n = np.cross(ab, ac)
        n = n / linalg.norm(n)
        quadNormals[quadIndex] = n

    for i, v2q in enumerate(vertexToQuads):
        n = np.zeros(3, np.float32)
        if not v2q:
            vertexNormals[i] = n
            continue
        for q in v2q:
            n = n + quadNormals[q]
        vertexNormals[i] = n / linalg.norm(n)

    vertexNormals.resize(len(vertexNormals) * 3)
    return vertexNormals
