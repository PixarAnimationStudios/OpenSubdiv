//
//   Copyright 2013 Autodesk, Inc.
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#ifndef _MayaPolySmooth
#define _MayaPolySmooth

#include <maya/MPxNode.h>
#include <maya/MTypeId.h>


class MayaPolySmooth : public MPxNode {

public:

    MayaPolySmooth();

    virtual  ~MayaPolySmooth();

    virtual MStatus compute( const MPlug& plug, MDataBlock& data );

    static void * creator();
    
    static MStatus initialize();

public:

    // MAYA_NODE_BUILDER:BEG [ATTRIBUTE DECLARATION] ==========
    static MObject a_inputPolymesh;        // This is a description for this attribute
    static MObject a_output;               // This is a description for this attribute
    static MObject a_subdivisionLevels;    // The number of recursive quad subdivisions to perform on each face.
    static MObject a_recommendedIsolation; // The recommended number of levels of isolation / subdivision based on topology
    static MObject a_vertBoundaryMethod;   // Controls how boundary edges and vertices are interpolated. <ul> <li>Smooth, Edges: Renderman: InterpolateBoundaryEdgeOnly</li> <li>Smooth, Edges and Corners: Renderman: InterpolateBoundaryEdgeAndCorner</li> </ul>
    static MObject a_fvarBoundaryMethod;   // Controls how boundaries are treated for face-varying data (UVs and Vertex Colors). <ul> <li>Bi-linear (None): Renderman: InterpolateBoundaryNone</li> <li>Smooth, (Edge Only): Renderman: InterpolateBoundaryEdgeOnly</li> <li>Smooth, (Edges and Corners: Renderman: InterpolateBoundaryEdgeAndCorner</li> <li>Smooth, (ZBrush and Maya "Smooth Internal Only"): Renderman: InterpolateBoundaryAlwaysSharp</li> </ul>
    static MObject a_fvarPropagateCorners; //
    static MObject a_smoothTriangles;      // Apply a special subdivision rule be applied to all triangular faces that was empirically determined to make triangles subdivide more smoothly.
    static MObject a_creaseMethod;         // Controls how boundary edges and vertices are interpolated. <ul> <li>Normal</li> <li>Chaikin: Improves the appearance of multiedge creases with varying weight</li> </ul>
    // MAYA_NODE_BUILDER:END [ATTRIBUTE DECLARATION] ==========

    static const MTypeId id;
    static const MString typeNameStr;
};

#endif // _MayaPolySmooth
