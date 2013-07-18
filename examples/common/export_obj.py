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
import maya.OpenMaya as OpenMaya

selectionList = OpenMaya.MSelectionList()
OpenMaya.MGlobal.getActiveSelectionList(selectionList)

path = OpenMaya.MDagPath()
selectionList.getDagPath(0, path)

meshFn = OpenMaya.MFnMesh(path)

points = OpenMaya.MPointArray()
normals = OpenMaya.MFloatVectorArray()
meshFn.getPoints(points)
meshFn.getVertexNormals(True, normals);

f = open('out.obj', 'w')

for i in range(0,points.length()):
    f.write('v %f %f %f\n' % (points[i].x, points[i].y, points[i].z))

for i in range(0,points.length()):
    f.write('vt 0 0 \n')
    
for i in range(0,normals.length()):
    f.write('vn %f %f %f\n' % (normals[i].x, normals[i].y, normals[i].z))

f.write('s off\n')

vertexCount = OpenMaya.MIntArray()
vertexList = OpenMaya.MIntArray()
meshFn.getVertices(vertexCount, vertexList)
edgeIds = OpenMaya.MUintArray()
edgeCreaseData = OpenMaya.MDoubleArray()
vtxIds = OpenMaya.MUintArray()
vtxCreaseData = OpenMaya.MDoubleArray()
meshFn.getCreaseEdges(edgeIds, edgeCreaseData)
#meshFn.getCreaseVertices(vtxIds, vtxCreaseData)

vindex = 0
for i in range(0,vertexCount.length()):
    f.write('f')
    for j in range(0, vertexCount[i]):
        v = vertexList[vindex] + 1
        f.write(' %d/%d/%d' % (v, v, v))
        vindex = vindex+1
    f.write('\n')

if vtxIds.length() > 0:
    f.write('t corner %d/%d/0' % (vtxIds.length(), vtxIds.length()))
    for i in range(0,vtxIds.length()):
        f.write(' %d' % vtxIds[i])
    for i in range(0,vtxCreaseData.length()):
        f.write(' %f' % vtxCreaseData[i])
    f.write('\n')
    
for i in range(0, edgeIds.length()):
    edgeIt = OpenMaya.MItMeshEdge(path)
    dummy = OpenMaya.MScriptUtil().asIntPtr()
    edgeIt.setIndex(edgeIds[i], dummy)
    faceList = OpenMaya.MIntArray()
    edgeIt.getConnectedFaces(faceList)
    vid0 = edgeIt.index(0)
    vid1 = edgeIt.index(1)

    f.write('t crease 2/1/0 %d %d %f\n' % (vid0, vid1, edgeCreaseData[i]))


f.close()

