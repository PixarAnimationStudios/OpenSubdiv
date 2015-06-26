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

#include "mayaPolySmooth.h"

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

#include <cassert>
#include <map>

#if defined(_MSV_VER) and (not defined(__INTEL_COMPILER))
    #pragma warning( disable : 174 593 )
#endif

// OpenSubdiv includes
#include <far/topologyDescriptor.h>
#include <far/stencilTableFactory.h>
#include <far/primvarRefiner.h>

#include <osd/mesh.h>
#include <osd/cpuVertexBuffer.h>


// ====================================
// Static Initialization
// ====================================

// MAYA_NODE_BUILDER:BEG [ATTRIBUTE INITIALIZATION] ==========
const MTypeId MayaPolySmooth::id( 0x0010a25b );
const MString MayaPolySmooth::typeNameStr("mayaPolySmooth");
MObject MayaPolySmooth::a_inputPolymesh;
MObject MayaPolySmooth::a_output;
MObject MayaPolySmooth::a_subdivisionLevels;
MObject MayaPolySmooth::a_recommendedIsolation;
MObject MayaPolySmooth::a_vertBoundaryMethod;
MObject MayaPolySmooth::a_fvarBoundaryMethod;
MObject MayaPolySmooth::a_fvarPropagateCorners;
MObject MayaPolySmooth::a_smoothTriangles;
MObject MayaPolySmooth::a_creaseMethod;
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

static OpenSubdiv::Sdc::Options::VtxBoundaryInterpolation
ConvertMayaVtxBoundary(short boundaryMethod) {

    typedef OpenSubdiv::Sdc::Options Sdc;

    switch (boundaryMethod) {
        case k_BoundaryMethod_InterpolateBoundaryNone          : return Sdc::VTX_BOUNDARY_NONE;
        case k_BoundaryMethod_InterpolateBoundaryEdgeOnly      : return Sdc::VTX_BOUNDARY_EDGE_ONLY;
        case k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner : return Sdc::VTX_BOUNDARY_EDGE_AND_CORNER;
        default: ;
    }
    MGlobal::displayError("VTX InterpolateBoundaryMethod value out of range. Using \"none\"");
    return Sdc::VTX_BOUNDARY_NONE;
}

// XXXX note: This function converts the options exposed in Maya's GUI which are
//            based on prman legacy face-varying boundary interpolation rules.
//            As a result, some OpenSubdiv 3.0 FVar interpolation rules are not
//            exposed, and the some of the ones exposed fix incorrect behavior
//            from legacy prman code, so the results are not 100% backward compatible.
static OpenSubdiv::Sdc::Options::FVarLinearInterpolation
ConvertMayaFVarBoundary(short boundaryMethod, bool propagateCorner) {

    typedef OpenSubdiv::Sdc::Options Sdc;

    switch (boundaryMethod) {
        case k_BoundaryMethod_InterpolateBoundaryNone          : return Sdc::FVAR_LINEAR_ALL;
        case k_BoundaryMethod_InterpolateBoundaryEdgeOnly      : return Sdc::FVAR_LINEAR_NONE;
        case k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner : 
            return propagateCorner ? Sdc::FVAR_LINEAR_CORNERS_PLUS2 : Sdc::FVAR_LINEAR_CORNERS_PLUS1;
        case k_BoundaryMethod_InterpolateBoundaryAlwaysSharp   : return Sdc::FVAR_LINEAR_BOUNDARIES;
        default: ;
    }
    MGlobal::displayError("FVar InterpolateMethod value out of range. Using \"none\"");
    return Sdc::FVAR_LINEAR_ALL;
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

#define CHANNELUV 0

#define CHANNELCOLOR 1

// ====================================
// Constructors/Destructors
// ====================================
MayaPolySmooth::MayaPolySmooth() {}

MayaPolySmooth::~MayaPolySmooth() {}


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

typedef OpenSubdiv::Far::TopologyDescriptor Descriptor;

// Reference: OSD shape_utils.h:: applyTags() "crease"
static float
getCreaseEdges(MFnMesh const & inMeshFn, Descriptor & outDesc) {

    MUintArray tEdgeIds;
    MDoubleArray tCreaseData;
    float maxCreaseValue = 0.0f;

    if (inMeshFn.getCreaseEdges(tEdgeIds, tCreaseData)) {

        assert( tEdgeIds.length() == tCreaseData.length() );

        int    ncreases = tEdgeIds.length();
        int * vertPairs = new int[ncreases*2];
        float * weights = new float[ncreases];

        int2 edgeVerts;
        for (unsigned int j=0; j < tEdgeIds.length(); j++) {

            assert( tCreaseData[j] >= 0.0 );
            inMeshFn.getEdgeVertices(tEdgeIds[j], edgeVerts);

            vertPairs[j*2  ] = edgeVerts[0];
            vertPairs[j*2+1] = edgeVerts[1];
            weights[j] = float(tCreaseData[j]);
            maxCreaseValue = std::max(float(tCreaseData[j]), maxCreaseValue);
        }

        outDesc.numCreases = ncreases;
        outDesc.creaseVertexIndexPairs = vertPairs;
        outDesc.creaseWeights = weights;
    }
    return maxCreaseValue;
}


// Reference: OSD shape_utils.h:: applyTags() "corner"
static float
getCreaseVertices( MFnMesh const & inMeshFn, Descriptor & outDesc) {

    MUintArray tVertexIds;
    MDoubleArray tCreaseData;
    float maxCreaseValue = 0.0f;

    if ( inMeshFn.getCreaseVertices(tVertexIds, tCreaseData) ) {

        assert( tVertexIds.length() == tCreaseData.length() );

        int    ncorners = tVertexIds.length();
        int *     verts = new int[ncorners*2];
        float * weights = new float[ncorners];

        // Has crease vertices
        for (unsigned int j=0; j < tVertexIds.length(); j++) {

            assert( tCreaseData[j] >= 0.0 );

            verts[j] = tVertexIds[j];
            weights[j] = float(tCreaseData[j]);

            maxCreaseValue = std::max(float(tCreaseData[j]), maxCreaseValue);
        }

        outDesc.numCorners = ncorners;
        outDesc.cornerVertexIndices = verts;
        outDesc.cornerWeights = weights;
    }
    return maxCreaseValue;
}

// XXX -- Future Data Optimization: Implement varying data instead of forcing face-varying for ColorSets.

// Collect UVs and ColorSet info to represent them as face-varying in OpenSubdiv
static MStatus
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


//! Caller is expected to delete the returned value
static OpenSubdiv::Far::TopologyRefiner *
gatherTopology( MFnMesh & inMeshFn,
                MItMeshPolygon & inMeshItPolygon,
                OpenSubdiv::Sdc::SchemeType type,
                OpenSubdiv::Sdc::Options options,
                bool * hasUVs, bool * hasColors,
                std::vector<MFloatArray> & uvSet_uCoords,
                std::vector<MFloatArray> & uvSet_vCoords,
                std::vector<MColorArray> & colorSet_colors,
                float * maxCreaseSharpness=0 ) {

    MStatus returnStatus;

    // Gather FVarData
    MStringArray uvSetNames;
    MStringArray colorSetNames;
    std::vector<int> colorSetChannels;
    std::vector<MFnMesh::MColorRepresentation> colorSetReps;
    int totalColorSetChannels = 0;
    returnStatus = getMayaFvarFieldParams(inMeshFn, uvSetNames, colorSetNames, 
                        colorSetChannels, colorSetReps, totalColorSetChannels);
    MWARNERR(returnStatus, "Failed to retrieve Maya face-varying parameters");

    // Storage for UVs and ColorSets for face-vertex
    MIntArray fvArray; // face vertex array
    uvSet_uCoords.clear();  uvSet_uCoords.resize(uvSetNames.length());
    uvSet_vCoords.clear();  uvSet_vCoords.resize(uvSetNames.length());
    colorSet_colors.clear();colorSet_colors.resize(colorSetNames.length());

    // Put the data in the format needed for OSD
    Descriptor desc;
    
    int numFaceVertices = inMeshFn.numFaceVertices();

    desc.numVertices = inMeshFn.numVertices();
    desc.numFaces = inMeshItPolygon.count();

    int * vertsPerFace = new int[desc.numFaces],
        * vertIndices = new int[numFaceVertices];

    desc.numVertsPerFace = vertsPerFace;
    desc.vertIndicesPerFace = vertIndices;

    // Create Topology
    for (inMeshItPolygon.reset(); !inMeshItPolygon.isDone(); inMeshItPolygon.next()) {

        inMeshItPolygon.getVertices(fvArray);
        int nverts = fvArray.length();

        vertsPerFace[inMeshItPolygon.index()] = nverts;

        for (int i=0; i<nverts; ++i) {
            *vertIndices++ = fvArray[i];
        }
    }

    // Add Face-Varying data to the descriptor
    Descriptor::FVarChannel * channels = NULL;
    *hasUVs = uvSet_uCoords.size() > 0 && uvSet_vCoords.size() > 0;
    *hasColors = colorSet_colors.size() > 0;

    // Note : Only supports 1 channel of UVs and 1 channel of color
    if (*hasUVs || *hasColors) {

        // Create 2 face-varying channel descriptor that will hold UVs and color
        desc.numFVarChannels = 2;
        channels = new Descriptor::FVarChannel[desc.numFVarChannels];
        desc.fvarChannels = channels;

        int * uvIndices = new int[numFaceVertices];
        channels[CHANNELUV].valueIndices = uvIndices;
        channels[CHANNELUV].numValues = 0;

        int * colorIndices = new int[numFaceVertices];
        channels[CHANNELCOLOR].valueIndices = colorIndices;
        channels[CHANNELCOLOR].numValues = 0;

        // Obtain UV information
        if (*hasUVs) {
            inMeshFn.getUVs(uvSet_uCoords[0], uvSet_vCoords[0], &uvSetNames[0]);
            assert( uvSet_uCoords[0].length() == uvSet_vCoords[0].length() );

            int uvId = 0, nUVs = 0;
            for (int faceIndex = 0; faceIndex < inMeshFn.numPolygons(); ++faceIndex)
            {
                int numVertices = inMeshFn.polygonVertexCount(faceIndex);
                for (int v = 0; v < numVertices; v++)
                {
                    inMeshFn.getPolygonUVid(faceIndex, v, uvId, &uvSetNames[0]);
                    uvIndices[nUVs++] = uvId;
                }
            }

            channels[CHANNELUV].numValues = uvSet_uCoords[0].length();
        }

        // Obtain color information
        if (*hasColors) {  
            inMeshFn.getColors(colorSet_colors[0], &colorSetNames[0]);

            int colorId = 0, nColors = 0;
            bool addDefaultColor = true;
            for (int faceIndex = 0; faceIndex < inMeshFn.numPolygons(); ++faceIndex)
            {
                int numVertices = inMeshFn.polygonVertexCount(faceIndex);
                for ( int v = 0 ; v < numVertices; v++ )
                {
                    inMeshFn.getColorIndex(faceIndex, v, colorId, &colorSetNames[0]);
                    if (colorId == -1)
                    {
                        if (addDefaultColor)
                        {
                            addDefaultColor = false;
                            colorSet_colors[0].append(MColor(1.0, 1.0, 1.0, 1.0));
                        }
                        colorId = colorSet_colors[0].length() - 1;
                    }
                    colorIndices[nColors ++] = colorId;
                }
            }
            
            channels[CHANNELCOLOR].numValues = colorSet_colors[0].length();
        }
    }

    // Apply Creases
    float maxEdgeCrease = getCreaseEdges( inMeshFn, desc );
    float maxVertexCrease = getCreaseVertices( inMeshFn, desc );

    OpenSubdiv::Far::TopologyRefiner * refiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Descriptor>::Create(desc,
            OpenSubdiv::Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    delete [] desc.numVertsPerFace;
    delete [] desc.vertIndicesPerFace;
    delete [] desc.creaseVertexIndexPairs;
    delete [] desc.creaseWeights;
    delete [] desc.cornerVertexIndices;
    delete [] desc.cornerWeights;

    if (*hasUVs || *hasColors) {
        for(int i = 0 ; i < desc.numFVarChannels ; i ++) {
            delete [] channels[i].valueIndices;
        }
        delete [] channels;
    }

    if (maxCreaseSharpness) {
        *maxCreaseSharpness = std::max(maxEdgeCrease, maxVertexCrease);
    }

    return refiner;
}

static inline int
computeNumSubfaces( int nverts, int level ) {

    return nverts==4 ? 1<<(level<<1) : nverts*(1<<((level-1)<<1));
}

// Propagate objectGroups from inMesh to subdivided outMesh
// Note: Currently only supporting facet groups (for per-facet shading)
MStatus
createSmoothMesh_objectGroups( MFnMesh const & inMeshFn,
    MFnMeshData const & inMeshDat, MFnMeshData &newMeshDat, int level, int numSubfaces ) {

    MStatus status;

    MIntArray newCompElems;

    std::vector<unsigned int> offsets; // mapping of offsets for subdivided faces

    for(unsigned int gi=0; gi<inMeshDat.objectGroupCount(); gi++) {

        unsigned int compId = inMeshDat.objectGroup(gi, &status);
        MCHECKERR(status, "cannot get objectGroup() comp ID.");

        MFn::Type compType = inMeshDat.objectGroupType(compId, &status);
        MCHECKERR(status, "cannot get objectGroupType().");

        // Only supporting kMeshPolygonComponent ObjectGroups at this time
        // Skip the other types
        if (compType == MFn::kMeshPolygonComponent) {

            // get elements from inMesh objectGroupComponent
            MIntArray compElems;
            MFnSingleIndexedComponent compFn(
                inMeshDat.objectGroupComponent(compId), &status );
            MCHECKERR(status, "cannot get MFnSingleIndexedComponent for inMeshDat.objectGroupComponent().");
            compFn.getElements(compElems);

            if (compElems.length()==0) {
                continue;
            }

            // over-allocation to maximum possible length
            newCompElems.setLength( numSubfaces );

            if (offsets.empty()) {
                // lazy population of the subface offsets table
                int nfaces = inMeshFn.numPolygons();

                offsets.resize(nfaces);

                for (int i=0, count=0; i<nfaces; ++i) {

                    int nverts = inMeshFn.polygonVertexCount(i),
                        nsubfaces = computeNumSubfaces(nverts, level);

                    offsets[i] = count;
                    count+=nsubfaces;
                }
            }

            unsigned int idx = 0;

            // convert/populate newCompElems from compElems of inMesh
            // (with new face indices) to outMesh
            for (unsigned int i=0; i < compElems.length(); i++) {

                int nverts = inMeshFn.polygonVertexCount(compElems[i]),
                    nsubfaces = computeNumSubfaces(nverts, level);

                unsigned int subFaceOffset = offsets[compElems[i]];

                for (int j=0; j<nsubfaces; ++j) {
                    newCompElems[idx++] = subFaceOffset++;
                }
            }

            // resize to actual length
            newCompElems.setLength( idx );

            // create comp
            createComp(newMeshDat, compType, compId, newCompElems);
        }
    }
    return MS::kSuccess;
}

// Vertex container implementation.
struct Vertex {

    Vertex() { }

    Vertex(Vertex const & src) {
        position[0] = src.position[0];
        position[1] = src.position[1];
        position[2] = src.position[2];
    }

    void Clear( void * =0 ) {
        position[0]=position[1]=position[2]=0.0f;
    }

    void AddWithWeight(Vertex const & src, float weight) {
        position[0]+=weight*src.position[0];
        position[1]+=weight*src.position[1];
        position[2]+=weight*src.position[2];
    }

    float position[3];
};

// Face-varying container implementation for UVs
struct FVarVertexUV {

    FVarVertexUV() { 
        u=v=0.0f;
    }

    FVarVertexUV(FVarVertexUV const & src) {
        u = src.u;
        v = src.v;
    }

    void Clear() {
        u=v=0.0f;
    }

    void AddWithWeight(FVarVertexUV const & src, float weight) {
        u += weight * src.u;
        v += weight * src.v;
    }

    // Basic 'uv' layout channel
    float u,v;
};

// Face-varying container implementation for Color
struct FVarVertexColor {

    FVarVertexColor() { 
        r=g=b=a=0.0f;
    } 

    FVarVertexColor(FVarVertexColor const & src) {
        r = src.r;
        g = src.g;
        b = src.b;
        a = src.a;
    }

    void Clear() {
        r=g=b=a=0.0f;
    }

    void AddWithWeight(FVarVertexColor const & src, float weight) {
        r += weight * src.r;
        g += weight * src.g;
        b += weight * src.b;
        a += weight * src.a;
    }

    // Basic 'rgba' layout channel
    float r,g,b,a;
};

static MStatus
convertToMayaMeshData(OpenSubdiv::Far::TopologyRefiner const & refiner,
    std::vector<Vertex> const & refinedVerts,
    bool hasUVs, std::vector<FVarVertexUV> const & refinedUVs,
    bool hasColors, std::vector<FVarVertexColor> const & refinedColors,
    MFnMesh & inMeshFn, MObject newMeshDataObj) {

    MStatus status;

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    int maxlevel = refiner.GetMaxLevel();

    OpenSubdiv::Far::TopologyLevel const & refLastLevel 
                                                = refiner.GetLevel(maxlevel);

    int nfaces = refLastLevel.GetNumFaces();

    // Init Maya Data

    // Face Counts
    MIntArray faceCounts(nfaces);
    for (int face=0; face < nfaces; ++face) {
        faceCounts[face] = 4;
    }

    // Face Connects
    MIntArray faceConnects(nfaces*4);
    for (int face=0, idx=0; face < nfaces; ++face) {
        IndexArray fverts = refLastLevel.GetFaceVertices(face);
        for (int vert=0; vert < fverts.size(); ++vert) {
            faceConnects[idx++] = fverts[vert];
        }
    }

    // Points
    int nverts = refLastLevel.GetNumVertices();
    int firstOfLastVert = refiner.GetNumVerticesTotal() 
                        - nverts 
                        - refiner.GetLevel(0).GetNumVertices();

    MFloatPointArray points(nverts);
    for (int vIt = 0; vIt < nverts; ++vIt) {
        Vertex const & v = refinedVerts[firstOfLastVert + vIt];
        points.set(vIt, v.position[0], v.position[1], v.position[2]);
    }

    // Create New Mesh from MFnMesh
    MFnMesh newMeshFn;
    MObject newMeshObj = newMeshFn.create(points.length(), faceCounts.length(),
        points, faceCounts, faceConnects, newMeshDataObj, &status);
    MCHECKERR(status, "Cannot create new mesh");

    // Get face-varying set names and other info from the inMesh
    MStringArray uvSetNames;
    MStringArray colorSetNames;
    std::vector<int> colorSetChannels;
    std::vector<MFnMesh::MColorRepresentation> colorSetReps;
    int totalColorSetChannels = 0;
    status = getMayaFvarFieldParams(inMeshFn, uvSetNames, colorSetNames,
        colorSetChannels, colorSetReps, totalColorSetChannels);

    // Add new UVs back to the mesh if needed
    if (hasUVs) {

        MIntArray fvarConnects(faceConnects.length());
        int count = 0;
        for (int f = 0; f < refLastLevel.GetNumFaces(); ++f) {
            IndexArray faceIndices = refLastLevel.GetFaceFVarValues(f, CHANNELUV);
            for (int index = 0 ; index < faceIndices.size() ; ++index) {
                fvarConnects[count++] = faceIndices[index];
            }
        }

        int nuvs = refLastLevel.GetNumFVarValues(CHANNELUV);
        int firstOfLastUvs = refiner.GetNumFVarValuesTotal(CHANNELUV) 
                             - nuvs
                             - refiner.GetLevel(0).GetNumFVarValues(CHANNELUV);

        MFloatArray uCoord(nuvs), vCoord(nuvs);
        for (int uvIt = 0; uvIt < nuvs; ++uvIt) {
            FVarVertexUV const & uv = refinedUVs[firstOfLastUvs + uvIt];
            uCoord[uvIt] = uv.u;
            vCoord[uvIt] = uv.v;
        }

        // Currently, the plugin only supports one UV set
        int uvSetIndex = 0;
        if (uvSetIndex > 0) {
            status = newMeshFn.createUVSetDataMesh( uvSetNames[uvSetIndex] );
            MCHECKERR(status, "Cannot create UVSet");
        }
        static MString defaultUVName("map1");
        MString const * uvname = uvSetIndex==0 ? 
            &defaultUVName : &uvSetNames[uvSetIndex];
        status = newMeshFn.setUVs(uCoord, vCoord, uvname);                  
        MCHECKERR(status, "Cannot set UVs for set : "+*uvname);

        status = newMeshFn.assignUVs(faceCounts, fvarConnects, uvname);     
        MCHECKERR(status, "Cannot assign UVs");
    }

    // Add new colors back to the mesh if needed
    if (hasColors) {

        int count = 0;
        MIntArray fvarConnects2(faceConnects.length());
        for (int f = 0 ; f < refLastLevel.GetNumFaces(); ++f) {
            IndexArray faceIndices = refLastLevel.GetFaceFVarValues(f, CHANNELCOLOR);
            for (int index = 0 ; index < faceIndices.size() ; ++index) {
                fvarConnects2[count++] = faceIndices[index];
            }
        }
  
        int ncols = refLastLevel.GetNumFVarValues(CHANNELCOLOR);
        int firstOfLastCols = refiner.GetNumFVarValuesTotal(CHANNELCOLOR) 
                              - ncols
                              - refiner.GetLevel(0).GetNumFVarValues(CHANNELCOLOR);

        MColorArray colorArray(ncols);
        for (int colIt = 0; colIt < ncols; ++colIt) {
            FVarVertexColor const & c = refinedColors[firstOfLastCols + colIt];
            colorArray.set(colIt, c.r, c.g, c.b, c.a);
        }

        // Currently, the plugin only supports one color sets
        int colorSetIndex = 0;

        // Assign color buffer and map the ids for each face-vertex
        // API Limitation: Cannot set MColorRepresentation here
        status = newMeshFn.createColorSetDataMesh(
            colorSetNames[colorSetIndex]);                                
        MCHECKERR(status, "Cannot create ColorSet");

        bool isColorClamped = inMeshFn.isColorClamped(
            colorSetNames[colorSetIndex], &status);                   
        MCHECKERR(status, "Can not get Color Clamped ");

        status = newMeshFn.setIsColorClamped(
            colorSetNames[colorSetIndex], isColorClamped);                     
        MCHECKERR(status, "Can not set Color Clamped : " + isColorClamped);

        status = newMeshFn.setColors(
            colorArray, &colorSetNames[colorSetIndex], 
            colorSetReps[colorSetIndex]);   
        MCHECKERR(status, "Can not set Colors");

        status = newMeshFn.assignColors(
            fvarConnects2, &colorSetNames[colorSetIndex]);                          
        MCHECKERR(status, "Can not assign Colors");
    }

    return MS::kSuccess;
}

MStatus
MayaPolySmooth::compute( const MPlug& plug, MDataBlock& data ) {

    MStatus status;

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

            // == Get Mesh Functions and Iterators ==========================
            MFnMeshData inMeshDat(inMeshObj);
            MFnMesh inMeshFn(inMeshObj, &status);
            MCHECKERR(status, "ERROR getting inMeshFn\n");
            MItMeshPolygon inMeshItPolygon(inMeshObj, &status);
            MCHECKERR(status, "ERROR getting inMeshItPolygon\n");

            // Convert attr values to OSD enums
            OpenSubdiv::Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

            // == Create Far topology ==========================
            OpenSubdiv::Sdc::Options options;
            options.SetVtxBoundaryInterpolation(ConvertMayaVtxBoundary(vertBoundaryMethod));
            options.SetFVarLinearInterpolation(ConvertMayaFVarBoundary(fvarBoundaryMethod, fvarPropCorners));
            options.SetCreasingMethod(creaseMethodVal ?
                 OpenSubdiv::Sdc::Options::CREASE_CHAIKIN : OpenSubdiv::Sdc::Options::CREASE_UNIFORM);
            options.SetTriangleSubdivision(smoothTriangles ?
                 OpenSubdiv::Sdc::Options::TRI_SUB_SMOOTH : OpenSubdiv::Sdc::Options::TRI_SUB_CATMARK);

            // Storage for face-varying values (UV sets, vertex colors...)
            std::vector<MFloatArray> uvSet_uCoords;
            std::vector<MFloatArray> uvSet_vCoords;
            std::vector<MColorArray> colorSet_colors;

            bool hasUVs = false, hasColors = false;
            float maxCreaseSharpness=0.0f;
            OpenSubdiv::Far::TopologyRefiner * refiner = gatherTopology(
                inMeshFn, inMeshItPolygon, type, options, &hasUVs, &hasColors,
                uvSet_uCoords, uvSet_vCoords, colorSet_colors, &maxCreaseSharpness);

            assert(refiner);

            // == Refine & Interpolate ==========================
            refiner->RefineUniform(OpenSubdiv::Far::TopologyRefiner::UniformOptions(subdivisionLevel));

            // Prepare vertex information
            Vertex const * initialVerts = 
                reinterpret_cast<Vertex const *>(inMeshFn.getRawPoints(&status));
            std::vector<Vertex> refinedVerts(
                refiner->GetNumVerticesTotal() - refiner->GetLevel(0).GetNumVertices());
            Vertex const * srcVerts = &initialVerts[0];
            Vertex * dstVerts = &refinedVerts[0];
           
            // Verify the refiner has the correct number of values 
            // needed to interpolate the different channels
            int numInitialUVs = refiner->GetLevel(0).GetNumFVarValues(CHANNELUV);
            int numInitialColors = refiner->GetLevel(0).GetNumFVarValues(CHANNELCOLOR);

            if (hasUVs && numInitialUVs <= 0) {
                hasUVs = false;
                MGlobal::displayError("Model with incorrect data, the UV channel will not be interpolated.");
            }

            if (hasColors && numInitialColors <= 0) {
                hasColors = false;  
                MGlobal::displayError("Model with incorrect data, the color channel will not be interpolated.");
            } 

            // Prepare UV information if needed
            std::vector<FVarVertexUV> initialUVs, refinedUVs;
            FVarVertexUV const * srcUV = NULL;
            FVarVertexUV * dstUV = NULL;
            if(hasUVs) {
                initialUVs.resize(numInitialUVs);
                refinedUVs.resize(refiner->GetNumFVarValuesTotal(CHANNELUV));
                for (int i=0; i<numInitialUVs; ++i) {
                    initialUVs[i].u = uvSet_uCoords[0][i];
                    initialUVs[i].v = uvSet_vCoords[0][i];
                }
                srcUV = &initialUVs[0];
                dstUV = &refinedUVs[0];
            }

            // Prepare color information if needed
            std::vector<FVarVertexColor> initialColors, refinedColors;
            FVarVertexColor const * srcColor = NULL;
            FVarVertexColor * dstColor = NULL;
            if(hasColors) {
                initialColors.resize(numInitialColors);
                refinedColors.resize(refiner->GetNumFVarValuesTotal(CHANNELCOLOR));
                for (int i=0; i<numInitialColors; ++i) {
                    initialColors[i].r = colorSet_colors[0][i].r;
                    initialColors[i].g = colorSet_colors[0][i].g;
                    initialColors[i].b = colorSet_colors[0][i].b;
                    initialColors[i].a = colorSet_colors[0][i].a;
                }
                srcColor = &initialColors[0];
                dstColor = &refinedColors[0];
            }

            // Interpolate the vertices and the different channels
            OpenSubdiv::Far::PrimvarRefiner primvarRefiner(*refiner); 
            
            for (int level = 1; level <= subdivisionLevel; ++level) {
                
                // Interpolate vertices
                primvarRefiner.Interpolate(level, srcVerts, dstVerts);
                srcVerts = dstVerts;
                dstVerts += refiner->GetLevel(level).GetNumVertices();

                // Interpolate the uv set
                if(hasUVs) {
                    primvarRefiner.InterpolateFaceVarying(level, srcUV, dstUV, CHANNELUV);
                    srcUV = dstUV;
                    dstUV += refiner->GetLevel(level).GetNumFVarValues(CHANNELUV);
                }

                // Interpolate any color set
                if(hasColors) {
                    primvarRefiner.InterpolateFaceVarying(level, srcColor, dstColor, CHANNELCOLOR);
                    srcColor = dstColor;
                    dstColor += refiner->GetLevel(level).GetNumFVarValues(CHANNELCOLOR);
                }
            }

            // == Convert subdivided OpenSubdiv mesh to MFnMesh Data outputMesh =============

            // Create New Mesh Data Object
            MFnMeshData newMeshData;
            MObject     newMeshDataObj = newMeshData.create(&status);
            MCHECKERR(status, "ERROR creating outputData");

            // Create out mesh
            status = convertToMayaMeshData(*refiner, refinedVerts, hasUVs, 
                refinedUVs, hasColors, refinedColors, inMeshFn, newMeshDataObj);
            MCHECKERR(status, "ERROR convertOsdFarToMayaMesh");

            // Propagate objectGroups from inMesh to outMesh (for per-facet shading, etc)
            status = createSmoothMesh_objectGroups(inMeshFn, inMeshDat,
                newMeshData, subdivisionLevel, refiner->GetLevel(subdivisionLevel).GetNumFaces());

            // Write to output plug
            MDataHandle outMeshH = data.outputValue(a_output, &status);
            MCHECKERR(status, "ERROR getting polygon data handle\n");
            outMeshH.set(newMeshDataObj);

            int isolation = std::min(10,(int)ceil(maxCreaseSharpness)+1);
            data.outputValue(a_recommendedIsolation).set(isolation);

            // == Cleanup OSD ============================================

            // REVISIT: Re-add these deletes
            delete refiner;

            // note that the subd mesh was created (see the section below if !createdSubdMesh)
            createdSubdMesh = true;
        }

        // Pass-through inMesh to outMesh if not created the subd mesh
        if (!createdSubdMesh) {
            MDataHandle outMeshH = data.outputValue(a_output, &status);
            status = outMeshH.copy(data.outputValue(a_inputPolymesh, &status));
            MCHECKERR(status, "ERROR getting polygon data handle\n");
        }

        // Clean up Maya Plugs
        data.setClean(plug);

    } else {
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
void* MayaPolySmooth::creator() {

    return new MayaPolySmooth;
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
MStatus
MayaPolySmooth::initialize() {

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
    MCHECKERR( stat, "cannot create MayaPolySmooth::inputPolymesh" );
    stat = tAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::inputPolymesh.setReadable()" );
    stat = tAttr.setWritable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::inputPolymesh.setWritable()" );
    stat = tAttr.setHidden(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::inputPolymesh.setHidden()" );
    stat = addAttribute( a_inputPolymesh );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_inputPolymesh)" );

    // a_output : This is a description for this attribute
    a_output = tAttr.create("output", "out", MFnData::kMesh, MObject::kNullObj, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::output" );
    stat = tAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::output.setReadable()" );
    stat = tAttr.setWritable(false);
    MCHECKERR( stat, "cannot MayaPolySmooth::output.setWritable()" );
    stat = tAttr.setHidden(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::output.setHidden()" );
    stat = addAttribute( a_output );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_output)" );

    // a_subdivisionLevels : The number of recursive quad subdivisions to perform on each face.
    a_subdivisionLevels = nAttr.create("subdivisionLevels", "sl", MFnNumericData::kInt, 0.0, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::subdivisionLevels" );
    stat = nAttr.setDefault(2);
    MCHECKERR( stat, "cannot MayaPolySmooth::subdivisionLevels.setDefault(2)" );
    stat = nAttr.setMin(0);
    MCHECKERR( stat, "cannot MayaPolySmooth::subdivisionLevels.setMin(0)" );
    stat = nAttr.setMax(10);
    MCHECKERR( stat, "cannot MayaPolySmooth::subdivisionLevels.setMax(10)" );
    stat = nAttr.setSoftMax(4);
    MCHECKERR( stat, "cannot MayaPolySmooth::subdivisionLevels.setSoftMax(4)" );
    stat = nAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::subdivisionLevels.setReadable()" );
    stat = nAttr.setWritable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::subdivisionLevels.setWritable()" );
    stat = addAttribute( a_subdivisionLevels );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_subdivisionLevels)" );

    // a_recommendedIsolation : The number of recursive quad subdivisions to perform on each face.
    a_recommendedIsolation = nAttr.create("recommendedIsolation", "ri", MFnNumericData::kInt, 0.0, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::recommendedIsolation" );
    stat = nAttr.setDefault(2);
    MCHECKERR( stat, "cannot MayaPolySmooth::recommendedIsolation.setDefault(0)" );
    stat = nAttr.setMin(0);
    MCHECKERR( stat, "cannot MayaPolySmooth::recommendedIsolation.setMin(0)" );
    stat = nAttr.setMax(10);
    MCHECKERR( stat, "cannot MayaPolySmooth::recommendedIsolation.setSoftMax(10)" );
    stat = nAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::recommendedIsolation.setReadable()" );
    stat = nAttr.setWritable(false);
    MCHECKERR( stat, "cannot MayaPolySmooth::recommendedIsolation.setWritable()" );
    stat = nAttr.setHidden(false);
    MCHECKERR( stat, "cannot MayaPolySmooth::recommendedIsolation.setHidden()" );
    stat = addAttribute( a_recommendedIsolation );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_recommendedIsolation)" );

    // a_vertBoundaryMethod : Controls how boundary edges and vertices are interpolated. <ul> <li>Smooth, Edges: Renderman: InterpolateBoundaryEdgeOnly</li> <li>Smooth, Edges and Corners: Renderman: InterpolateBoundaryEdgeAndCorner</li> </ul>
    a_vertBoundaryMethod = eAttr.create("vertBoundaryMethod", "vbm", 0, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::vertBoundaryMethod" );
    stat = eAttr.addField("Interpolate Edges", k_BoundaryMethod_InterpolateBoundaryEdgeOnly);
    MCHECKERR( stat, "cannot MayaPolySmooth::vertBoundaryMethod.addField(Interpolate Edges, k_BoundaryMethod_InterpolateBoundaryEdgeOnly)" );
    stat = eAttr.addField("Interpolate Edges And Corners", k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner);
    MCHECKERR( stat, "cannot MayaPolySmooth::vertBoundaryMethod.addField(Interpolate Edges And Corners, k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner)" );
    stat = eAttr.setDefault(k_BoundaryMethod_InterpolateBoundaryEdgeOnly);
    MCHECKERR( stat, "cannot MayaPolySmooth::vertBoundaryMethod.setDefault(k_BoundaryMethod_InterpolateBoundaryEdgeOnly)" );
    stat = eAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::vertBoundaryMethod.setReadable()" );
    stat = eAttr.setWritable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::vertBoundaryMethod.setWritable()" );
    stat = addAttribute( a_vertBoundaryMethod );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_vertBoundaryMethod)" );

    // a_fvarBoundaryMethod : Controls how boundaries are treated for face-varying data (UVs and Vertex Colors). <ul> <li>Bi-linear (None): Renderman: InterpolateBoundaryNone</li> <li>Smooth, (Edge Only): Renderman: InterpolateBoundaryEdgeOnly</li> <li>Smooth, (Edges and Corners: Renderman: InterpolateBoundaryEdgeAndCorner</li> <li>Smooth, (ZBrush and Maya "Smooth Internal Only"): Renderman: InterpolateBoundaryAlwaysSharp</li> </ul>
    a_fvarBoundaryMethod = eAttr.create("fvarBoundaryMethod", "fvbm", 0, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::fvarBoundaryMethod" );
    stat = eAttr.addField("Bi-linear (None)", k_BoundaryMethod_InterpolateBoundaryNone);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarBoundaryMethod.addField(Bi-linear (None), k_BoundaryMethod_InterpolateBoundaryNone)" );
    stat = eAttr.addField("Smooth (Edge Only)", k_BoundaryMethod_InterpolateBoundaryEdgeOnly);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarBoundaryMethod.addField(Smooth (Edge Only), k_BoundaryMethod_InterpolateBoundaryEdgeOnly)" );
    stat = eAttr.addField("Smooth (Edge and Corner)", k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarBoundaryMethod.addField(Smooth (Edge and Corner), k_BoundaryMethod_InterpolateBoundaryEdgeAndCorner)" );
    stat = eAttr.addField("Smooth (Always Sharp)", k_BoundaryMethod_InterpolateBoundaryAlwaysSharp);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarBoundaryMethod.addField(Smooth (Always Sharp), k_BoundaryMethod_InterpolateBoundaryAlwaysSharp)" );
    stat = eAttr.setDefault(k_BoundaryMethod_InterpolateBoundaryNone);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarBoundaryMethod.setDefault(k_BoundaryMethod_InterpolateBoundaryNone)" );
    stat = eAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarBoundaryMethod.setReadable()" );
    stat = eAttr.setWritable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarBoundaryMethod.setWritable()" );
    stat = addAttribute( a_fvarBoundaryMethod );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_fvarBoundaryMethod)" );

    // a_fvarPropagateCorners :
    a_fvarPropagateCorners = nAttr.create("fvarPropagateCorners", "fvpc", MFnNumericData::kBoolean, 0.0, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::fvarPropagateCorners" );
    stat = nAttr.setDefault(false);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarPropagateCorners.setDefault(false)" );
    stat = nAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarPropagateCorners.setReadable()" );
    stat = nAttr.setWritable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::fvarPropagateCorners.setWritable()" );
    stat = addAttribute( a_fvarPropagateCorners );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_fvarPropagateCorners)" );

    // a_smoothTriangles : Apply a special subdivision rule be applied to all triangular faces that was empirically determined to make triangles subdivide more smoothly.
    a_smoothTriangles = nAttr.create("smoothTriangles", "stri", MFnNumericData::kBoolean, 0.0, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::smoothTriangles" );
    stat = nAttr.setDefault(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::smoothTriangles.setDefault(true)" );
    stat = nAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::smoothTriangles.setReadable()" );
    stat = nAttr.setWritable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::smoothTriangles.setWritable()" );
    stat = addAttribute( a_smoothTriangles );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_smoothTriangles)" );

    // a_creaseMethod : Controls how boundary edges and vertices are interpolated. <ul> <li>Normal</li> <li>Chaikin: Improves the appearance of multiedge creases with varying weight</li> </ul>
    a_creaseMethod = eAttr.create("creaseMethod", "crm", 0, &stat);
    MCHECKERR( stat, "cannot create MayaPolySmooth::creaseMethod" );
    stat = eAttr.addField("Normal", k_creaseMethod_normal);
    MCHECKERR( stat, "cannot MayaPolySmooth::creaseMethod.addField(Normal, k_creaseMethod_normal)" );
    stat = eAttr.addField("Chaikin", k_creaseMethod_chaikin);
    MCHECKERR( stat, "cannot MayaPolySmooth::creaseMethod.addField(Chaikin, k_creaseMethod_chaikin)" );
    stat = eAttr.setDefault(0);
    MCHECKERR( stat, "cannot MayaPolySmooth::creaseMethod.setDefault(0)" );
    stat = eAttr.setReadable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::creaseMethod.setReadable()" );
    stat = eAttr.setWritable(true);
    MCHECKERR( stat, "cannot MayaPolySmooth::creaseMethod.setWritable()" );
    stat = addAttribute( a_creaseMethod );
    MCHECKERR( stat, "cannot MayaPolySmooth::addAttribute(a_creaseMethod)" );

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

    stat = attributeAffects( a_creaseMethod, a_recommendedIsolation );
    MCHECKERR( stat, "cannot have attribute creaseMethod affect .si output" );
    stat = attributeAffects( a_inputPolymesh, a_recommendedIsolation );
    MCHECKERR( stat, "cannot have attribute inputPolymesh affect .si output" );
    stat = attributeAffects( a_subdivisionLevels, a_recommendedIsolation );
    MCHECKERR( stat, "cannot have attribute subdivisionLevels affect .si output" );
    stat = attributeAffects( a_smoothTriangles, a_recommendedIsolation );
    MCHECKERR( stat, "cannot have attribute smoothTriangles affect .si output" );
    stat = attributeAffects( a_fvarPropagateCorners, a_recommendedIsolation );
    MCHECKERR( stat, "cannot have attribute fvarPropagateCorners affect .si output" );
    stat = attributeAffects( a_vertBoundaryMethod, a_recommendedIsolation );
    MCHECKERR( stat, "cannot have attribute vertBoundaryMethod affect .si output" );
    stat = attributeAffects( a_fvarBoundaryMethod, a_recommendedIsolation );
    MCHECKERR( stat, "cannot have attribute fvarBoundaryMethod affect .si output" );
    // MAYA_NODE_BUILDER:END [ATTRIBUTE DEPENDS] ==========

    return MS::kSuccess;
}


// ==========================================
// Plugin
// ==========================================

MStatus initializePlugin( MObject obj ) {

    MStatus   status = MS::kSuccess;
    MFnPlugin plugin( obj, "MayaPolySmooth", "1.0", "Any");

    status = plugin.registerNode(
        MayaPolySmooth::typeNameStr,
        MayaPolySmooth::id,
        MayaPolySmooth::creator,
        MayaPolySmooth::initialize);
    MCHECKERR(status, "registerNode");

    // Source UI scripts
    MString path = plugin.loadPath()+"/mayaPolySmooth.mel";
    if (not MGlobal::sourceFile(path)) {
        path = "mayaPolySmooth.mel";
        if (not MGlobal::sourceFile(path)) {
            MGlobal::displayWarning("Failed to source mayaPolySmooth.mel.");
        }
    }

    // RegisterUI
    status = plugin.registerUI("mayaPolySmooth_addUI()", "mayaPolySmooth_removeUI()");
    MCHECKERR(status, "registerUI");

    return status;
}

MStatus uninitializePlugin( MObject obj) {

    MStatus   returnStatus = MS::kSuccess;
    MFnPlugin plugin( obj );

    returnStatus = plugin.deregisterNode( MayaPolySmooth::id );
    MCHECKERR(returnStatus, "deregisterNode");

    return returnStatus;
}
