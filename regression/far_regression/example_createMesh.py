#!/usr/bin/env python

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

"""
Example

Description:
    Create a Maya Mesh using OpenMaya

    in Maya's python editor :

    import example_createMesh
    reload(example_createMesh)

    example_createMesh.readPolyFile('/host/devel/rtc3/trees/dev/amber/bin/pbr2/script')

"""

from maya import OpenMaya as om
from itertools import chain
import maya.cmds as cmds

def getDagPath(nodeName):
    """Get an MDagPath for the associated node name
    """
    selList = om.MSelectionList()
    selList.add(nodeName)
    dagPath = om.MDagPath()
    selList.getDagPath(0, dagPath)
    return dagPath

def convertToMIntArray(listOfInts):
    newMIntArray = om.MIntArray(len(listOfInts))
    for i in range(len(listOfInts)):
        val = listOfInts[i]
        newMIntArray.set(val, i)
    return newMIntArray


def convertToMPointArray(listOfVertexTuples):
    newMPointArray = om.MPointArray(len(listOfVertexTuples))
    for i in range(len(listOfVertexTuples)):
        v = listOfVertexTuples[i]
        newMPointArray.set(i, v[0], v[1], v[2], 1.0)
    return newMPointArray


def createMesh(vertices, polygons, parent=None):
    '''Create a mesh with the specified vertices and polygons
    '''
    # The parameters used in MFnMesh.create() can all be derived from the
    # input vertices and polygon lists

    numVertices = len(vertices)
    numPolygons = len(polygons)

    verticesM = convertToMPointArray(vertices)

    polygonCounts = [len(i) for i in polygons]
    polygonCountsM = convertToMIntArray(polygonCounts)

    # Flatten the list of lists
    # Reference: http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    polygonConnects = list(chain.from_iterable(polygons))
    polygonConnectsM = convertToMIntArray(polygonConnects)

    # Determine parent
    if parent == None:
        parentM = om.MObject()
    else:
        parentM = getDagPath(parent).node()

    # Create Mesh
    newMesh = om.MFnMesh()
    newTransformOrShape = newMesh.create(
        numVertices,
        numPolygons,
        verticesM,
        polygonCountsM,
        polygonConnectsM,
        parentM)

    dagpath = om.MDagPath()
    om.MDagPath.getAPathTo( newTransformOrShape, dagpath )

    # Assign the default shader to the mesh.
    cmds.sets(
        dagpath.fullPathName(),
        edit=True,
        forceElement='initialShadingGroup')

    return dagpath.partialPathName()

def readPolyFile(path):
    polys = ''
    try:
        with open(path, 'r') as f:
            polys = ''
            for line in f.readlines():
                polys += line.rstrip()
    except:
        print 'Cannot read '+str(path)

    polys = eval(polys)

    tx = 0.0
    ty = 0.0
    for poly in polys:
        verts = poly['verts']
        faces = poly['faces']
        parent = None
        dagpath = createMesh(verts, faces, parent)
        cmds.move( tx, ty, 0, dagpath, absolute=True )
        tx+=4.0
        if tx > 16.0:
            tx=0.0
            ty+=4.0
