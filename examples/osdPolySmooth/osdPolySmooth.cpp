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

#include "osdPolySmooth.h"

#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnGenericAttribute.h>
#include <maya/MFnLightDataAttribute.h>

#include <maya/MDataHandle.h>
#include <maya/MDataBlock.h>
#include <maya/MPlug.h>
#include <maya/MIOStream.h>

#include <maya/MGlobal.h>
#include <maya/MFnMesh.h>
#include <maya/MFnMeshData.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MFloatArray.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MPointArray.h>
#include <maya/MFloatPointArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MUintArray.h>

#include <maya/MFnPlugin.h>

#include <stdexcept>
#include <map>

#if defined(_MSV_VER) and (not defined(__INTEL_COMPILER))
    #pragma warning( disable : 174 593 )
#endif

// OpenSubdiv includes
#include <osd/vertex.h>
#include <osd/vertex.h>
#include <osd/mesh.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
#include <osd/cpuVertexBuffer.h>


// ====================================
// Static Initialization
// ====================================

// MAYA_NODE_BUILDER:BEG [ATTRIBUTE INITIALIZATION] ==========
const MTypeId OsdPolySmooth::id( 0x0010a25b );
const MString OsdPolySmooth::typeNameStr("osdPolySmooth");
MObject OsdPolySmooth::a_inputPolymesh;
MObject OsdPolySmooth::a_output;
MObject OsdPolySmooth::a_subdivisionLevels;
MObject OsdPolySmooth::a_vertBoundaryMethod;
MObject OsdPolySmooth::a_fvarBoundaryMethod;
MObject OsdPolySmooth::a_fvarPropagateCorners;
MObject OsdPolySmooth::a_smoothTriangles;
MObject OsdPolySmooth::a_creaseMethod;
// MAYA_NODE_BUILDER:END [ATTRIBUTE INITIALIZATION] ==========

// ATTR ENUMS
// Note: Do not change these values as these are serialized numerically in the Maya scenes)
//
enum BoundaryMethod {
    k_BoundaryMethod_InterpolateBoundaryNone = 0,
    k_BoundaryMethod_InterpolateBoundaryEdgeOnly = 1,
    k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner  = 2,
    k_BoundaryMethod_InterpolateBoundaryAlwaysSharp  = 3
};

enum CreaseMethod {
    k_creaseMethod_normal  = 0,
    k_creaseMethod_chaikin = 1
};

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>               HMesh;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>               HFace;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>             HVertex;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex>           HHalfedge;
typedef OpenSubdiv::HbrFVarData<OpenSubdiv::OsdVertex>           HFvarData;
typedef OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex> HCatmark;

typedef OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>               FMesh;
typedef OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex>        FMeshFactory;

HMesh::InterpolateBoundaryMethod
ConvertMayaBoundaryMethodShortToOsdInterpolateBoundaryMethod(short boundaryMethod) {
    switch (boundaryMethod) {

        case k_BoundaryMethod_InterpolateBoundaryNone: return HMesh::k_InterpolateBoundaryNone;

        case k_BoundaryMethod_InterpolateBoundaryEdgeOnly: return HMesh::k_InterpolateBoundaryEdgeOnly;

        case k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner: return HMesh::k_InterpolateBoundaryEdgeAndCorner;

        case k_BoundaryMethod_InterpolateBoundaryAlwaysSharp: return HMesh::k_InterpolateBoundaryAlwaysSharp;

        default: ;
    }
    MGlobal::displayError("InterpolateBoundaryMethod value out of range. Using \"none\"");
    return OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>::k_InterpolateBoundaryNone;
}

// ====================================
// Macros
// ====================================
#define MCHECKERR(status,message)       \
    if( MStatus::kSuccess != status ) {   \
        cerr << "ERROR: " << message << "[" << status << "]" << endl;        \
        return status;                    \
    }

#define MWARNERR(status,message)       \
    if( MStatus::kSuccess != status ) {   \
        cerr << "ERROR: " << message << "[" << status << "]" << endl;        \
    }


// ====================================
// Constructors/Destructors
// ====================================
OsdPolySmooth::OsdPolySmooth() {}

OsdPolySmooth::~OsdPolySmooth() {}


// ====================================
// Helper Functions
// ====================================
// Create component groups
void
createComp(MFnMeshData &dataCreator, MFn::Type compType, unsigned compId, MIntArray &compList) {

  MStatus returnStatus;

  MFnSingleIndexedComponent comp;
  MObject compObj = comp.create(compType,&returnStatus);
  MWARNERR(returnStatus, "cannot create MFnSingleIndexedComponent");

  returnStatus = comp.addElements(compList);
  MWARNERR(returnStatus, "Error in addElements() for MFnSingleIndexedComponent");

  returnStatus = dataCreator.addObjectGroup(compId);
  MWARNERR(returnStatus, "Error in addObjectGroup()");

  returnStatus = dataCreator.setObjectGroupComponent(compId, compObj);
  MWARNERR(returnStatus, "Error in setObjectGroupComponent()");
}


// ====================================
// OpenSubdiv Functions
// ====================================


// Reference: OSD shape_utils.h:: applyTags() "crease"
float
applyCreaseEdges(MFnMesh const & inMeshFn, HMesh * hbrMesh) {

    MStatus returnStatus;
    MUintArray tEdgeIds;
    MDoubleArray tCreaseData;
    float maxCreaseValue = 0.0f;

    if (inMeshFn.getCreaseEdges(tEdgeIds, tCreaseData)) {

        assert( tEdgeIds.length() == tCreaseData.length() );

        // Has crease edges
        int2 edgeVerts;
        for (unsigned int j=0; j < tEdgeIds.length(); j++) {

            // Get vert ids from maya edge
            int edgeId = tEdgeIds[j];
            returnStatus = inMeshFn.getEdgeVertices(edgeId, edgeVerts);

            // Assumption: The OSD vert ids are identical to those of the Maya mesh
            HVertex const * v = hbrMesh->GetVertex( edgeVerts[0] ),
                          * w = hbrMesh->GetVertex( edgeVerts[1] );

            HHalfedge * e = 0;
            if( v and w ) {

                if( (e = v->GetEdge(w)) == 0) {
                    e = w->GetEdge(v);
                }

                if(e) {
                    assert( tCreaseData[j] >= 0.0 );
                    e->SetSharpness( (float)tCreaseData[j] );

                    maxCreaseValue = std::max(float(tCreaseData[j]), maxCreaseValue);
                } else {
                    fprintf(stderr,
                        "warning: cannot find edge for crease tag (%d,%d)\n",
                            edgeVerts[0], edgeVerts[1] );
                }
            }
        }
    }
    return maxCreaseValue;
}


// Reference: OSD shape_utils.h:: applyTags() "corner"
float
applyCreaseVertices( MFnMesh const & inMeshFn, HMesh * hbrMesh ) {

    MUintArray tVertexIds;
    MDoubleArray tCreaseData;
    float maxCreaseValue = 0.0f;

    if ( inMeshFn.getCreaseVertices(tVertexIds, tCreaseData) ) {

        assert( tVertexIds.length() == tCreaseData.length() );

        // Has crease vertices
        for (unsigned int j=0; j < tVertexIds.length(); j++) {

            // Assumption: The OSD vert ids are identical to those of the Maya mesh

            HVertex * v = hbrMesh->GetVertex( tVertexIds[j] );
            if(v) {

                assert( tCreaseData[j] >= 0.0 );

                v->SetSharpness( (float)tCreaseData[j] );

                maxCreaseValue = std::max(float(tCreaseData[j]), maxCreaseValue);
            } else {
                fprintf(stderr,
                    "warning: cannot find vertex for corner tag (%d)\n",
                        tVertexIds[j] );
           }
        }
    }
    return maxCreaseValue;
}

// XXX -- Future Data Optimization: Implement varying data instead of forcing face-varying for ColorSets.

// Collect UVs and ColorSet info to represent them as face-varying in OpenSubdiv
MStatus
getMayaFvarFieldParams(
    MFnMesh const & inMeshFn,
    MStringArray & uvSetNames,
    MStringArray & colorSetNames,
    std::vector<int> & colorSetChannels,
    std::vector<MFnMesh::MColorRepresentation> &colorSetReps,
    int & totalColorSetChannels) {

    MStatus returnStatus;

    returnStatus = inMeshFn.getUVSetNames(uvSetNames);
    MCHECKERR(returnStatus, "Cannot get uvSet names");

    returnStatus = inMeshFn.getColorSetNames(colorSetNames);
    MCHECKERR(returnStatus, "Cannot get colorSet names");

    colorSetChannels.resize(colorSetNames.length());
    colorSetReps.resize(colorSetNames.length());
    totalColorSetChannels = 0;

    for (unsigned int i=0; i < colorSetNames.length(); i++) {

        colorSetReps[i] = inMeshFn.getColorRepresentation(colorSetNames[i], &returnStatus);
        MCHECKERR(returnStatus, "Cannot get colorSet representation");

               if (colorSetReps[i] == MFnMesh::kAlpha) {
            colorSetChannels[i] = 1;
        } else if (colorSetReps[i] == MFnMesh::kRGB) {
            colorSetChannels[i] = 3;
        } else {
            colorSetChannels[i] = 4; // kRGBA
        }
        totalColorSetChannels += colorSetChannels[i];
    }
    return MS::kSuccess;
}


//! Create OSD HBR mesh.
//! Caller is expected to delete the resulting hbrMesh returned
HMesh *
createOsdHbrFromPoly( MFnMesh const & inMeshFn,
                      MItMeshPolygon & inMeshItPolygon,
                      std::vector<int> & fvarIndices,
                      std::vector<int> & fvarWidths)
{
    MStatus returnStatus;

    // == Mesh Properties

    // Gather FVarData
    MStringArray uvSetNames;
    MStringArray colorSetNames;
    std::vector<int> colorSetChannels;
    std::vector<MFnMesh::MColorRepresentation> colorSetReps;
    int totalColorSetChannels = 0;
    returnStatus = getMayaFvarFieldParams(inMeshFn, uvSetNames, colorSetNames, colorSetChannels, colorSetReps, totalColorSetChannels);
    MWARNERR(returnStatus, "Failed to retrieve Maya face-varying parameters");

    // Create face-varying data with independent float channels of dimension 1
    // Note: This FVarData needs to be kept around for the duration of the HBR mesh
    int totalFvarWidth = 2*uvSetNames.length() + totalColorSetChannels;
    fvarIndices.resize(totalFvarWidth);
    fvarWidths.resize(totalFvarWidth);
    for (int i=0; i < totalFvarWidth; ++i) {
        fvarIndices[i] = i;
        fvarWidths[i] = 1;
    }

    // temp storage for UVs and ColorSets for a face
    MIntArray fvArray; // face vertex array
    std::vector<MFloatArray> uvSet_uCoords(uvSetNames.length());
    std::vector<MFloatArray> uvSet_vCoords(uvSetNames.length());
    std::vector<MColorArray> colorSet_colors(colorSetNames.length());

    // =====================================
    // Init HBR
    // =====================================

    // Determine HBR Subdiv Method
    static HCatmark _catmark;

    // Create HBR mesh
    assert(fvarIndices.size() == fvarWidths.size());
    HMesh * hbrMesh = new HMesh( &_catmark,
                                 (int)fvarIndices.size(),
                                 (fvarIndices.size() > 0) ? &fvarIndices[0] : NULL,
                                 (fvarWidths.size()  > 0) ? &fvarWidths[0] : NULL,
                                 totalFvarWidth );

    // Create Stub HBR Vertices
    int numVerts = inMeshFn.numVertices();
    OpenSubdiv::OsdVertex v;
    for(int i=0; i<numVerts; i++ ) {
        hbrMesh->NewVertex( i, v );
    }

    // ===================================================
    // Create HBR Topology
    // ===================================================

    assert(totalFvarWidth == hbrMesh->GetTotalFVarWidth());
    unsigned int ptxidx = 0;

    for( inMeshItPolygon.reset(); !inMeshItPolygon.isDone(); inMeshItPolygon.next() ) {

        // Verts for this face
        inMeshItPolygon.getVertices(fvArray);
        unsigned int nv = fvArray.length();

        bool valid = true;

        for(unsigned int j=0;j<fvArray.length(); j++) {

            HVertex const * origin      = hbrMesh->GetVertex( fvArray[j] ),
                          * destination = hbrMesh->GetVertex( fvArray[(j+1)%nv] );
            HHalfedge const * opposite = destination->GetEdge(origin);

            if(origin==NULL || destination==NULL) {
                fprintf(stderr, "Skipping face: An edge was specified that connected a nonexistent vertex\n");
                valid = false;
                break;
            }

            if(origin == destination) {
                fprintf(stderr, "Skipping face: An edge was specified that connected a vertex to itself\n");
                valid = false;
                break;
            }

            if(opposite && opposite->GetOpposite() ) {
                fprintf(stderr, "Skipping face: A non-manifold edge incident to more than 2 faces was found\n");
                valid = false;
                break;
            }

            if(origin->GetEdge(destination)) {
                fprintf(stderr, "Skipping face: An edge connecting two vertices was specified more than once."
                       " It's likely that an incident face was flipped\n");
                valid = false;
                break;
            }
        }

        // Update faces
        const int * fvArrayPtr = &(fvArray[0]); // get the raw int* array from std::vector<int>
        OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex> * face = hbrMesh->NewFace(nv, fvArrayPtr, 0);

        // Increment ptex index
        face->SetPtexIndex(ptxidx);

        // Add FaceVaryingData (UVSets, ...)
        if (totalFvarWidth > 0) {

            // Retrieve all UV and ColorSet data
            MIntArray faceCounts;
            for (unsigned int i=0; i < uvSetNames.length(); ++i) {
                returnStatus = inMeshItPolygon.getUVs(uvSet_uCoords[i], uvSet_vCoords[i], &uvSetNames[i] );
                MWARNERR(returnStatus, "Cannot get UVs");
            }
            for (unsigned int i=0; i < colorSetNames.length(); ++i) {
                returnStatus = inMeshItPolygon.getColors (colorSet_colors[i], &colorSetNames[i]);
                MWARNERR(returnStatus, "Cannot get face vertex colors");
            }

            std::vector<float> fvarItem(totalFvarWidth); // storage for all the face-varying channels for this face-vertex

            // loop over each uvSet and the uvs within
            for (unsigned int fvid=0; fvid < fvArray.length(); ++fvid) {

                int fvarItemIndex = 0;
                // Handle uvSets
                for( unsigned int uvSetIt=0; uvSetIt < uvSetNames.length(); ++uvSetIt ) {
                    fvarItem[fvarItemIndex]   = uvSet_uCoords[uvSetIt][fvid];
                    fvarItem[fvarItemIndex+1] = uvSet_vCoords[uvSetIt][fvid];
                    fvarItemIndex +=2;
                }
                // Handle colorSets
                for( unsigned int colorSetIt=0; colorSetIt < colorSetNames.length(); ++colorSetIt ) {
                    if (colorSetChannels[colorSetIt] == 1) {
                        fvarItem[fvarItemIndex]   = colorSet_colors[colorSetIt][fvid].a;
                    }
                    else if (colorSetChannels[colorSetIt] == 3) {
                        fvarItem[fvarItemIndex]   = colorSet_colors[colorSetIt][fvid].r;
                        fvarItem[fvarItemIndex+1] = colorSet_colors[colorSetIt][fvid].g;
                        fvarItem[fvarItemIndex+2] = colorSet_colors[colorSetIt][fvid].b;
                    }
                    else { // colorSetChannels[colorSetIt] == 4
                        fvarItem[fvarItemIndex]   = colorSet_colors[colorSetIt][fvid].r;
                        fvarItem[fvarItemIndex+1] = colorSet_colors[colorSetIt][fvid].g;
                        fvarItem[fvarItemIndex+2] = colorSet_colors[colorSetIt][fvid].b;
                        fvarItem[fvarItemIndex+3] = colorSet_colors[colorSetIt][fvid].a;
                    }
                    fvarItemIndex += colorSetChannels[colorSetIt];
                }
                assert((fvarItemIndex) == totalFvarWidth); // For UVs, sanity check the resulting value

                // Insert into the HBR structure for that face
                HVertex * hbrVertex = hbrMesh->GetVertex( fvArray[fvid] );
                HFvarData & fvarData = hbrVertex->GetFVarData(face);

                if (not fvarData.IsInitialized()) {
                    fvarData.SetAllData(totalFvarWidth, &fvarItem[0]);
                } else if (not fvarData.CompareAll(totalFvarWidth, &fvarItem[0])) {

                    // If data exists for this face vertex, but is different
                    // (e.g. we're on a UV seam) create another fvar datum
                    OpenSubdiv::HbrFVarData<OpenSubdiv::OsdVertex> &fvarData_new = hbrVertex->NewFVarData(face);
                    fvarData_new.SetAllData( totalFvarWidth, &fvarItem[0] );
                }
            }
        }

        // Increment ptxidx and store off ptex index values
        //   The number of ptexIds needed is 1 if a quad.  Otherwise it is the number of
        //   vertices for the face.
        int numPtexIdsForFace;
        if (valid) {
            numPtexIdsForFace = ( nv != 4 ) ? nv : 1 ;
        }
        else {
            numPtexIdsForFace = 0;
        }
        ptxidx += numPtexIdsForFace;
    }

    // Apply Creases
    applyCreaseEdges( inMeshFn, hbrMesh );
    applyCreaseVertices( inMeshFn, hbrMesh );

    // Return the resulting HBR Mesh
    // Note that boundaryMethods and hbrMesh->Finish() still need to be called
    return hbrMesh;
}


MStatus convertOsdFarToMayaMeshData(
     FMesh const * farMesh,
     OpenSubdiv::OsdCpuVertexBuffer * vertexBuffer,
     int subdivisionLevel,
     MFnMesh const & inMeshFn,
     MObject newMeshDataObj ) {

    MStatus returnStatus;

    // Get sizing data from OSD
    const OpenSubdiv::FarPatchTables *farPatchTables = farMesh->GetPatchTables();
    int numPolygons = farPatchTables->GetNumFaces(); // use the highest level stored in the patch tables
    const unsigned int *polygonConnects_orig = farPatchTables->GetFaceVertices(); // use the highest level stored in the patch tables

    const OpenSubdiv::FarSubdivisionTables<OpenSubdiv::OsdVertex> *farSubdivTables = farMesh->GetSubdivisionTables();
    unsigned int numVertices  = farSubdivTables->GetNumVertices(subdivisionLevel);
    unsigned int vertexOffset = farSubdivTables->GetFirstVertexOffset(subdivisionLevel);

    // Init Maya Data
    MFloatPointArray points(numVertices);
    MIntArray faceCounts(numPolygons); // number of edges for each polygon.  Assume quads (4-edges per face)
    MIntArray faceConnects(numPolygons*4); // array of vertex ids for all edges. assuming quads

    // -- Face Counts
    for (int i=0; i < numPolygons; ++i) {
        faceCounts[i] = 4;
    }

    // -- Face Connects
    for (unsigned int i=0; i < faceConnects.length(); i++) {
        faceConnects[i] = polygonConnects_orig[i] - vertexOffset; // adjust vertex indices so that v0 is at index 0
    }

    // -- Points
    // Number of floats in each vertex.  (positions, normals, etc)
    int numFloatsPerVertex = vertexBuffer->GetNumElements();
    assert(numFloatsPerVertex == 3); // assuming only xyz stored for each vertex
    const float *vertexData = vertexBuffer->BindCpuBuffer();
    float *ptrVertexData;

    for (unsigned int i=0; i < numVertices; i++) {

        // make sure to offset to the first osd vertex for that subd level
        unsigned int osdRawVertexIndex = i + vertexOffset;

        // Lookup the data in the vertexData
        ptrVertexData = (float *) vertexData + ((osdRawVertexIndex) * numFloatsPerVertex);
        points.set(i, ptrVertexData[0], ptrVertexData[1], ptrVertexData[2]);
    }

    // Create New Mesh from MFnMesh
    MFnMesh newMeshFn;
    MObject newMeshObj = newMeshFn.create(points.length(), faceCounts.length(),
        points, faceCounts, faceConnects, newMeshDataObj, &returnStatus);
    MCHECKERR(returnStatus, "Cannot create new mesh");

    // Attach UVs (if present)
    // ASSUMPTION: Only tracking UVs as FVar data.  Will need to change this
    // ASSUMPTION: OSD has a unique UV for each face-vertex
    int fvarTotalWidth = farMesh->GetTotalFVarWidth();

    if (fvarTotalWidth > 0) {

        // Get face-varying set names and other info from the inMesh
        MStringArray uvSetNames;
        MStringArray colorSetNames;
        std::vector<int> colorSetChannels;
        std::vector<MFnMesh::MColorRepresentation> colorSetReps;
        int totalColorSetChannels = 0;
        returnStatus = getMayaFvarFieldParams(inMeshFn, uvSetNames, colorSetNames, colorSetChannels, colorSetReps, totalColorSetChannels);

        int numUVSets = uvSetNames.length();
        int expectedFvarTotalWidth = numUVSets*2 + totalColorSetChannels;
        assert(fvarTotalWidth == expectedFvarTotalWidth);

        const OpenSubdiv::FarPatchTables::FVarDataTable &fvarDataTable =  farPatchTables->GetFVarDataTable();
        assert(fvarDataTable.size() == expectedFvarTotalWidth*faceConnects.length());

        // Create an array of indices to map each face-vert to the UV and ColorSet Data
        MIntArray fvarConnects(faceConnects.length());
        for (unsigned int i=0; i < faceConnects.length(); i++) {
            fvarConnects[i] = i;
        }

        MFloatArray uCoord(faceConnects.length());
        MFloatArray vCoord(faceConnects.length());

        for (int uvSetIndex=0; uvSetIndex < numUVSets; uvSetIndex++) {

            for(unsigned int vertid=0; vertid < faceConnects.length(); vertid++) {
                int fvarItem = vertid*fvarTotalWidth + uvSetIndex*2; // stride per vertex is the fvarTotalWidth
                uCoord[vertid] = fvarDataTable[fvarItem];
                vCoord[vertid] = fvarDataTable[fvarItem+1];
            }
            // Assign UV buffer and map the uvids for each face-vertex
            if (uvSetIndex != 0) { // assume uvset index 0 is the default UVset, so do not create
                returnStatus = newMeshFn.createUVSetDataMesh( uvSetNames[uvSetIndex] );
            }
            MCHECKERR(returnStatus, "Cannot create UVSet");
            newMeshFn.setUVs(uCoord,vCoord, &uvSetNames[uvSetIndex]);
            newMeshFn.assignUVs(faceCounts, fvarConnects, &uvSetNames[uvSetIndex]);
        }

        MColorArray colorArray(faceConnects.length());
        int colorSetRelativeStartIndex = numUVSets*2;

        for (unsigned int colorSetIndex=0; colorSetIndex < colorSetNames.length(); colorSetIndex++) {

            for(unsigned int vertid=0; vertid < faceConnects.length(); vertid++) {

                int fvarItem = vertid*fvarTotalWidth + colorSetRelativeStartIndex;
                if (colorSetChannels[colorSetIndex] == 1) {
                    colorArray[vertid].r = fvarDataTable[fvarItem];
                    colorArray[vertid].g = fvarDataTable[fvarItem];
                    colorArray[vertid].b = fvarDataTable[fvarItem];
                    colorArray[vertid].a = 1.0f;
                } else if (colorSetChannels[colorSetIndex] == 3) {
                    colorArray[vertid].r = fvarDataTable[fvarItem];
                    colorArray[vertid].g = fvarDataTable[fvarItem+1];
                    colorArray[vertid].b = fvarDataTable[fvarItem+2];
                    colorArray[vertid].a = 1.0f;
                } else {
                    colorArray[vertid].r = fvarDataTable[fvarItem];
                    colorArray[vertid].g = fvarDataTable[fvarItem+1];
                    colorArray[vertid].b = fvarDataTable[fvarItem+2];
                    colorArray[vertid].a = fvarDataTable[fvarItem+3];
                }
            }

            // Assign UV buffer and map the uvids for each face-vertex
            // API Limitation: Cannot set MColorRepresentation here
            returnStatus = newMeshFn.createColorSetDataMesh(colorSetNames[colorSetIndex]);
            MCHECKERR(returnStatus, "Cannot create ColorSet");

            bool isColorClamped = inMeshFn.isColorClamped(colorSetNames[colorSetIndex], &returnStatus);
            newMeshFn.setIsColorClamped(colorSetNames[colorSetIndex], isColorClamped);
            newMeshFn.setColors(colorArray, &colorSetNames[colorSetIndex], colorSetReps[colorSetIndex]);
            newMeshFn.assignColors(fvarConnects, &colorSetNames[colorSetIndex]);

            // Increment colorSet start location in fvar buffer
            colorSetRelativeStartIndex += colorSetChannels[colorSetIndex];
        }
    }
    return MS::kSuccess;
}


// Propagate objectGroups from inMesh to subdivided outMesh
// Note: Currently only supporting facet groups (for per-facet shading)
MStatus
createSmoothMesh_objectGroups(const MFnMeshData &inMeshDat,
    int subdivisionLevel, MFnMeshData &newMeshDat) {

    MStatus returnStatus;

    int facesPerBaseFace = (int)(pow(4.0f, subdivisionLevel));
    MIntArray newCompElems;

    for(unsigned int gi=0; gi < inMeshDat.objectGroupCount(); gi++) {
        unsigned int compId = inMeshDat.objectGroup(gi, &returnStatus);
        MCHECKERR(returnStatus, "cannot get objectGroup() comp ID.");
        MFn::Type compType = inMeshDat.objectGroupType(compId, &returnStatus);
        MCHECKERR(returnStatus, "cannot get objectGroupType().");

        // get elements from inMesh objectGroupComponent
        MIntArray compElems;
        MFnSingleIndexedComponent compFn(inMeshDat.objectGroupComponent(compId), &returnStatus );
        MCHECKERR(returnStatus, "cannot get MFnSingleIndexedComponent for inMeshDat.objectGroupComponent().");
        compFn.getElements(compElems);

        // Only supporting kMeshPolygonComponent ObjectGroups at this time
        // Skip the other types
        if (compType == MFn::kMeshPolygonComponent) {

            // convert/populate newCompElems from compElems of inMesh
            // (with new face indices) to outMesh
            newCompElems.setLength( compElems.length() * facesPerBaseFace );
            for (unsigned int i=0; i < compElems.length(); i++) {

                int startElemIndex = i * facesPerBaseFace;
                int startElemValue = compElems[i] * facesPerBaseFace;

                for (int j=0; j < facesPerBaseFace; j++) {
                    newCompElems[startElemIndex+j] = startElemValue+j;
                }
            }
            // create comp
            createComp(newMeshDat, compType, compId, newCompElems);
        }
    }
    return MS::kSuccess;
}


// ====================================
// Compute
// ====================================
//
//  Description:
//      This method computes the value of the given output plug based
//      on the values of the input attributes.
//
//  Arguments:
//      plug - the plug to compute
//      data - object that provides access to the attributes for this node
//
MStatus OsdPolySmooth::compute( const MPlug& plug, MDataBlock& data ) {

    MStatus returnStatus;

    // Check which output attribute we have been asked to compute.  If this
    // node doesn't know how to compute it, we must return
    // MS::kUnknownParameter.
    //
    if( plug == a_output ) {
        bool createdSubdMesh = false;

        int subdivisionLevel = data.inputValue(a_subdivisionLevels).asInt();
        short stateH = data.inputValue(state).asShort();

        if ((subdivisionLevel > 0) and (stateH !=1)) {

            // == Retrieve input mesh ====================================
            // Get attr values
            MObject inMeshObj        = data.inputValue(a_inputPolymesh).asMesh();
            short vertBoundaryMethod = data.inputValue(a_vertBoundaryMethod).asShort();
            short fvarBoundaryMethod = data.inputValue(a_fvarBoundaryMethod).asShort();
            bool  fvarPropCorners    = data.inputValue(a_fvarPropagateCorners).asBool();
            bool  smoothTriangles    = data.inputValue(a_smoothTriangles).asBool();
            short creaseMethodVal    = data.inputValue(a_creaseMethod).asShort();

            // Convert attr values to OSD enums
            HMesh::InterpolateBoundaryMethod vertInterpBoundaryMethod =
                ConvertMayaBoundaryMethodShortToOsdInterpolateBoundaryMethod(vertBoundaryMethod);

            HMesh::InterpolateBoundaryMethod fvarInterpBoundaryMethod =
                ConvertMayaBoundaryMethodShortToOsdInterpolateBoundaryMethod(fvarBoundaryMethod);

            HCatmark::CreaseSubdivision creaseMethod =
                (creaseMethodVal == k_creaseMethod_chaikin) ?
                    HCatmark::k_CreaseChaikin : HCatmark::k_CreaseNormal;

            HCatmark::TriangleSubdivision triangleSubdivision =
                smoothTriangles ? HCatmark::k_New : HCatmark::k_Normal;

            // == Get Mesh Functions and Iterators ==========================
            MFnMeshData inMeshDat(inMeshObj);
            MFnMesh inMeshFn(inMeshObj, &returnStatus);
            MCHECKERR(returnStatus, "ERROR getting inMeshFn\n");
            MItMeshPolygon inMeshItPolygon(inMeshObj, &returnStatus);
            MCHECKERR(returnStatus, "ERROR getting inMeshItPolygon\n");

            // == Convert MFnMesh to OpenSubdiv =============================
            // Create the hbrMesh
            // Note: These fvar values only need to be kept alive through the life of the farMesh
            std::vector<int> fvarIndices;
            std::vector<int> fvarWidths;

            HMesh *hbrMesh = createOsdHbrFromPoly(
                inMeshFn, inMeshItPolygon, fvarIndices, fvarWidths);
            assert(hbrMesh);

            // Create the farMesh if successfully created the hbrMesh

            if (hbrMesh) {
                // Set Boundary methods and other hbr paramters
                hbrMesh->SetInterpolateBoundaryMethod( vertInterpBoundaryMethod );
                hbrMesh->SetFVarInterpolateBoundaryMethod( fvarInterpBoundaryMethod );
                hbrMesh->SetFVarPropagateCorners(fvarPropCorners);
                hbrMesh->GetSubdivision()->SetCreaseSubdivisionMethod(creaseMethod);

                // Set HBR Catmark Subdivision parameters
                HCatmark *catmarkSubdivision = dynamic_cast<HCatmark *>(hbrMesh->GetSubdivision());
                if (catmarkSubdivision) {
                    catmarkSubdivision->SetTriangleSubdivisionMethod(triangleSubdivision);
                }

                // Finalize subd calculations -- apply boundary interpolation rules and resolves singular verts, etc.
                // NOTE: This HAS to be called after all HBR parameters are set
                hbrMesh->Finish();

                int ncoarseverts = hbrMesh->GetNumVertices();

                // Create a FarMesh from the HBR mesh and pass into
                // It will be owned by the OsdMesh and deleted in the ~OsdMesh()
                FMeshFactory meshFactory(hbrMesh, subdivisionLevel, false);

                FMesh *farMesh = meshFactory.Create((hbrMesh->GetTotalFVarWidth() > 0));

                // == Setup OSD Data Structures =========================
                int numVertexElements  = 3; // only track vertex positions
                int numVaryingElements = 0; // XXX Future: Revise to include varying ColorSets
                int numVertices = inMeshFn.numVertices();

                int numFarVerts = farMesh->GetNumVertices();

                static OpenSubdiv::OsdCpuComputeController computeController = OpenSubdiv::OsdCpuComputeController();

                OpenSubdiv::OsdCpuComputeController::ComputeContext *computeContext =
                    OpenSubdiv::OsdCpuComputeController::ComputeContext::Create(farMesh);

                OpenSubdiv::OsdCpuVertexBuffer *vertexBuffer =
                    OpenSubdiv::OsdCpuVertexBuffer::Create(numVertexElements, numFarVerts );

                OpenSubdiv::OsdCpuVertexBuffer *varyingBuffer =
                    (numVaryingElements) ? OpenSubdiv::OsdCpuVertexBuffer::Create(numVaryingElements, numFarVerts) : NULL;

                // == UPDATE VERTICES (can be done after farMesh generated from topology) ==
                float const * vertex3fArray = inMeshFn.getRawPoints(&returnStatus);
                vertexBuffer->UpdateData(vertex3fArray, 0, numVertices );

                // Hbr dupes singular vertices during Mesh::Finish() - we need
                // to duplicate their positions in the vertex buffer.
                if (ncoarseverts > numVertices) {

                    MIntArray polyverts;

                    for (int i=numVertices; i<ncoarseverts; ++i) {

                        HVertex const * v = hbrMesh->GetVertex(i);

                        HFace const * f = v->GetIncidentEdge()->GetFace();

                        int vidx = -1;
                        for (int j=0; j<f->GetNumVertices(); ++j) {
                            if (f->GetVertex(j)==v) {
                                vidx = j;
                                break;
                            }
                        }
                        assert(vidx>-1);

                        inMeshFn.getPolygonVertices(f->GetID(), polyverts);

                        int vert = polyverts[vidx];

                        vertexBuffer->UpdateData(&vertex3fArray[0]+vert*numVertexElements, i, 1);
                    }
                }

                // == Delete HBR
                // Can now delete the hbrMesh as we will only be referencing the farMesh from this point on
                delete hbrMesh;
                hbrMesh = NULL;

                // == Subdivide OpenSubdiv mesh ==========================
                computeController.Refine(computeContext, farMesh->GetKernelBatches(), vertexBuffer, varyingBuffer);
                computeController.Synchronize();

                // == Convert subdivided OpenSubdiv mesh to MFnMesh Data outputMesh =============

                // Create New Mesh Data Object
                MFnMeshData newMeshData;
                MObject     newMeshDataObj = newMeshData.create(&returnStatus);
                MCHECKERR(returnStatus, "ERROR creating outputData");

                // Create out mesh
                returnStatus = convertOsdFarToMayaMeshData(farMesh, vertexBuffer, subdivisionLevel, inMeshFn, newMeshDataObj);
                MCHECKERR(returnStatus, "ERROR convertOsdFarToMayaMesh");

                // Propagate objectGroups from inMesh to outMesh (for per-facet shading, etc)
                returnStatus = createSmoothMesh_objectGroups(inMeshDat, subdivisionLevel, newMeshData );

                // Write to output plug
                MDataHandle outMeshH = data.outputValue(a_output, &returnStatus);
                MCHECKERR(returnStatus, "ERROR getting polygon data handle\n");
                outMeshH.set(newMeshDataObj);

                // == Cleanup OSD ============================================
                // REVISIT: Re-add these deletes
                delete(vertexBuffer);
                delete(varyingBuffer);
                delete(computeContext);
                delete(farMesh);

                // note that the subd mesh was created (see the section below if !createdSubdMesh)
                createdSubdMesh = true;
            }
        }

        // Pass-through inMesh to outMesh if not created the subd mesh
        if (!createdSubdMesh) {
            MDataHandle outMeshH = data.outputValue(a_output, &returnStatus);
            returnStatus = outMeshH.copy(data.outputValue(a_inputPolymesh, &returnStatus));
            MCHECKERR(returnStatus, "ERROR getting polygon data handle\n");
        }

        // Clean up Maya Plugs
        data.setClean(plug);
    }
    else {
        // Unhandled parameter in this compute function, so return MS::kUnknownParameter
        // so it is handled in a parent compute() function.
        return MS::kUnknownParameter;
    }
    return MS::kSuccess;
}


// Creator
//
//  Description:
//      this method exists to give Maya a way to create new objects
//      of this type.
//
//  Return Value:
//      a new object of this type
//
void* OsdPolySmooth::creator() {

    return new OsdPolySmooth;
}


// Create and Add Attributes
//
//  Description:
//      This method is called to create and initialize all of the attributes
//      and attribute dependencies for this node type.  This is only called
//      once when the node type is registered with Maya.
//
//  Return Values:
//      MS::kSuccess
//      MS::kFailure
//
MStatus OsdPolySmooth::initialize() {

    MStatus stat;

    MFnCompoundAttribute  cAttr;
    MFnEnumAttribute      eAttr;
    MFnGenericAttribute   gAttr;
    MFnLightDataAttribute lAttr;
    MFnMatrixAttribute    mAttr;
    MFnMessageAttribute   msgAttr;
    MFnNumericAttribute   nAttr;
    MFnTypedAttribute     tAttr;
    MFnUnitAttribute      uAttr;

    // MAYA_NODE_BUILDER:BEG [ATTRIBUTE CREATION] ==========
    // a_inputPolymesh : This is a description for this attribute
    a_inputPolymesh = tAttr.create("inputPolymesh", "ip", MFnData::kMesh, MObject::kNullObj, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::inputPolymesh" );
    stat = tAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::inputPolymesh.setReadable()" );
    stat = tAttr.setWritable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::inputPolymesh.setWritable()" );
    stat = tAttr.setHidden(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::inputPolymesh.setHidden()" );
    stat = addAttribute( a_inputPolymesh );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_inputPolymesh)" );

    // a_output : This is a description for this attribute
    a_output = tAttr.create("output", "out", MFnData::kMesh, MObject::kNullObj, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::output" );
    stat = tAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::output.setReadable()" );
    stat = tAttr.setWritable(false);
    MCHECKERR( stat, "cannot OsdPolySmooth::output.setWritable()" );
    stat = tAttr.setHidden(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::output.setHidden()" );
    stat = addAttribute( a_output );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_output)" );

    // a_subdivisionLevels : The number of recursive quad subdivisions to perform on each face.
    a_subdivisionLevels = nAttr.create("subdivisionLevels", "sl", MFnNumericData::kInt, 0.0, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::subdivisionLevels" );
    stat = nAttr.setDefault(2);
    MCHECKERR( stat, "cannot OsdPolySmooth::subdivisionLevels.setDefault(2)" );
    stat = nAttr.setMin(0);
    MCHECKERR( stat, "cannot OsdPolySmooth::subdivisionLevels.setMin(0)" );
    stat = nAttr.setMax(10);
    MCHECKERR( stat, "cannot OsdPolySmooth::subdivisionLevels.setMax(10)" );
    stat = nAttr.setSoftMax(4);
    MCHECKERR( stat, "cannot OsdPolySmooth::subdivisionLevels.setSoftMax(4)" );
    stat = nAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::subdivisionLevels.setReadable()" );
    stat = nAttr.setWritable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::subdivisionLevels.setWritable()" );
    stat = addAttribute( a_subdivisionLevels );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_subdivisionLevels)" );

    // a_vertBoundaryMethod : Controls how boundary edges and vertices are interpolated. <ul> <li>Smooth, Edges: Renderman: InterpolateBoundaryEdgeOnly</li> <li>Smooth, Edges and Corners: Renderman: InterpolateBoundaryEdgeAndCorner</li> </ul>
    a_vertBoundaryMethod = eAttr.create("vertBoundaryMethod", "vbm", 0, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::vertBoundaryMethod" );
    stat = eAttr.addField("Interpolate Edges", k_BoundaryMethod_InterpolateBoundaryEdgeOnly);
    MCHECKERR( stat, "cannot OsdPolySmooth::vertBoundaryMethod.addField(Interpolate Edges, k_BoundaryMethod_InterpolateBoundaryEdgeOnly)" );
    stat = eAttr.addField("Interpolate Edges And Corners", k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner);
    MCHECKERR( stat, "cannot OsdPolySmooth::vertBoundaryMethod.addField(Interpolate Edges And Corners, k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner)" );
    stat = eAttr.setDefault(k_BoundaryMethod_InterpolateBoundaryEdgeOnly);
    MCHECKERR( stat, "cannot OsdPolySmooth::vertBoundaryMethod.setDefault(k_BoundaryMethod_InterpolateBoundaryEdgeOnly)" );
    stat = eAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::vertBoundaryMethod.setReadable()" );
    stat = eAttr.setWritable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::vertBoundaryMethod.setWritable()" );
    stat = addAttribute( a_vertBoundaryMethod );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_vertBoundaryMethod)" );

    // a_fvarBoundaryMethod : Controls how boundaries are treated for face-varying data (UVs and Vertex Colors). <ul> <li>Bi-linear (None): Renderman: InterpolateBoundaryNone</li> <li>Smooth, (Edge Only): Renderman: InterpolateBoundaryEdgeOnly</li> <li>Smooth, (Edges and Corners: Renderman: InterpolateBoundaryEdgeAndCorner</li> <li>Smooth, (ZBrush and Maya "Smooth Internal Only"): Renderman: InterpolateBoundaryAlwaysSharp</li> </ul>
    a_fvarBoundaryMethod = eAttr.create("fvarBoundaryMethod", "fvbm", 0, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::fvarBoundaryMethod" );
    stat = eAttr.addField("Bi-linear (None)", k_BoundaryMethod_InterpolateBoundaryNone);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarBoundaryMethod.addField(Bi-linear (None), k_BoundaryMethod_InterpolateBoundaryNone)" );
    stat = eAttr.addField("Smooth (Edge Only)", k_BoundaryMethod_InterpolateBoundaryEdgeOnly);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarBoundaryMethod.addField(Smooth (Edge Only), k_BoundaryMethod_InterpolateBoundaryEdgeOnly)" );
    stat = eAttr.addField("Smooth (Edge and Corner)", k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarBoundaryMethod.addField(Smooth (Edge and Corner), k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner)" );
    stat = eAttr.addField("Smooth (Always Sharp)", k_BoundaryMethod_InterpolateBoundaryAlwaysSharp);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarBoundaryMethod.addField(Smooth (Always Sharp), k_BoundaryMethod_InterpolateBoundaryAlwaysSharp)" );
    stat = eAttr.setDefault(k_BoundaryMethod_InterpolateBoundaryNone);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarBoundaryMethod.setDefault(k_BoundaryMethod_InterpolateBoundaryNone)" );
    stat = eAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarBoundaryMethod.setReadable()" );
    stat = eAttr.setWritable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarBoundaryMethod.setWritable()" );
    stat = addAttribute( a_fvarBoundaryMethod );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_fvarBoundaryMethod)" );

    // a_fvarPropagateCorners :
    a_fvarPropagateCorners = nAttr.create("fvarPropagateCorners", "fvpc", MFnNumericData::kBoolean, 0.0, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::fvarPropagateCorners" );
    stat = nAttr.setDefault(false);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarPropagateCorners.setDefault(false)" );
    stat = nAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarPropagateCorners.setReadable()" );
    stat = nAttr.setWritable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::fvarPropagateCorners.setWritable()" );
    stat = addAttribute( a_fvarPropagateCorners );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_fvarPropagateCorners)" );

    // a_smoothTriangles : Apply a special subdivision rule be applied to all triangular faces that was empirically determined to make triangles subdivide more smoothly.
    a_smoothTriangles = nAttr.create("smoothTriangles", "stri", MFnNumericData::kBoolean, 0.0, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::smoothTriangles" );
    stat = nAttr.setDefault(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::smoothTriangles.setDefault(true)" );
    stat = nAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::smoothTriangles.setReadable()" );
    stat = nAttr.setWritable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::smoothTriangles.setWritable()" );
    stat = addAttribute( a_smoothTriangles );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_smoothTriangles)" );

    // a_creaseMethod : Controls how boundary edges and vertices are interpolated. <ul> <li>Normal</li> <li>Chaikin: Improves the appearance of multiedge creases with varying weight</li> </ul>
    a_creaseMethod = eAttr.create("creaseMethod", "crm", 0, &stat);
    MCHECKERR( stat, "cannot create OsdPolySmooth::creaseMethod" );
    stat = eAttr.addField("Normal", k_creaseMethod_normal);
    MCHECKERR( stat, "cannot OsdPolySmooth::creaseMethod.addField(Normal, k_creaseMethod_normal)" );
    stat = eAttr.addField("Chaikin", k_creaseMethod_chaikin);
    MCHECKERR( stat, "cannot OsdPolySmooth::creaseMethod.addField(Chaikin, k_creaseMethod_chaikin)" );
    stat = eAttr.setDefault(0);
    MCHECKERR( stat, "cannot OsdPolySmooth::creaseMethod.setDefault(0)" );
    stat = eAttr.setReadable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::creaseMethod.setReadable()" );
    stat = eAttr.setWritable(true);
    MCHECKERR( stat, "cannot OsdPolySmooth::creaseMethod.setWritable()" );
    stat = addAttribute( a_creaseMethod );
    MCHECKERR( stat, "cannot OsdPolySmooth::addAttribute(a_creaseMethod)" );

    // MAYA_NODE_BUILDER:END [ATTRIBUTE CREATION] ==========


    // Set up a dependency between the input and the output.  This will cause
    // the output to be marked dirty when the input changes.  The output will
    // then be recomputed the next time the value of the output is requested.
    //
    // MAYA_NODE_BUILDER:BEG [ATTRIBUTE DEPENDS] ==========
    stat = attributeAffects( a_creaseMethod, a_output );
    MCHECKERR( stat, "cannot have attribute creaseMethod affect output" );
    stat = attributeAffects( a_inputPolymesh, a_output );
    MCHECKERR( stat, "cannot have attribute inputPolymesh affect output" );
    stat = attributeAffects( a_subdivisionLevels, a_output );
    MCHECKERR( stat, "cannot have attribute subdivisionLevels affect output" );
    stat = attributeAffects( a_smoothTriangles, a_output );
    MCHECKERR( stat, "cannot have attribute smoothTriangles affect output" );
    stat = attributeAffects( a_fvarPropagateCorners, a_output );
    MCHECKERR( stat, "cannot have attribute fvarPropagateCorners affect output" );
    stat = attributeAffects( a_vertBoundaryMethod, a_output );
    MCHECKERR( stat, "cannot have attribute vertBoundaryMethod affect output" );
    stat = attributeAffects( a_fvarBoundaryMethod, a_output );
    MCHECKERR( stat, "cannot have attribute fvarBoundaryMethod affect output" );
    // MAYA_NODE_BUILDER:END [ATTRIBUTE DEPENDS] ==========

    return MS::kSuccess;
}


// ==========================================
// Plugin
// ==========================================

MStatus initializePlugin( MObject obj )
{
    MStatus   status = MS::kSuccess;
    MFnPlugin plugin( obj, "OsdPolySmooth", "1.0", "Any");

    status = plugin.registerNode(
        OsdPolySmooth::typeNameStr,
        OsdPolySmooth::id,
        OsdPolySmooth::creator,
        OsdPolySmooth::initialize);
    MCHECKERR(status, "registerNode");

    // Source UI scripts
    MString path = plugin.loadPath()+"/osdPolySmooth.mel";
    if (not MGlobal::sourceFile(path)) {
        path = "osdPolySmooth.mel";
        if (not MGlobal::sourceFile(path)) {
            MGlobal::displayWarning("Failed to source osdPolySmooth.mel.");
        }
    }

    // RegisterUI
    status = plugin.registerUI("osdPolySmooth_addUI()", "osdPolySmooth_removeUI()");
    MCHECKERR(status, "registerUI");

    return status;
}

MStatus uninitializePlugin( MObject obj)
{
    MStatus   returnStatus = MS::kSuccess;
    MFnPlugin plugin( obj );

    returnStatus = plugin.deregisterNode( OsdPolySmooth::id );
    MCHECKERR(returnStatus, "deregisterNode");

    return returnStatus;
}
