//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#if defined(__APPLE__)
    #include <maya/OpenMayaMac.h>
#else
    // Include GLEW before Maya and OSD includes
    #include <GL/glew.h>
#endif

#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MFloatArray.h>
#include <maya/MGlobal.h>

#include "osdMeshData.h"

#include <osd/cpuDispatcher.h>
#include <osd/cpuComputeController.h>

#ifdef OPENSUBDIV_HAS_OPENMP
#include <osd/ompDispatcher.h>
#include <osd/ompComputeController.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
#include <osd/clDispatcher.h>
#include <osd/clComputeController.h>
extern cl_context g_clContext;
extern cl_command_queue g_clQueue;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
#include <osd/cudaDispatcher.h>
#include <osd/cudaComputeController.h>
#endif

#include <vector>

#include "../common/maya_util.h"
#include "hbrUtil.h"
#include "OpenSubdivShader.h"


// #### Constructor
//
//      Initialize all context and buffers to NULL
//
OsdMeshData::OsdMeshData(const MDagPath& meshDagPath)
    : MUserData(false), 
      _meshDagPath(meshDagPath),
      _meshTopoDirty(true),
      _hbrmesh(NULL), 
      _mesh(NULL),
      _level(0),
      _kernel(kCPU),
      _adaptive(true),
      _fvarDesc(NULL),
      _needsUpdate(false),
      _needsInitializeMesh(false) 
{
}

// #### Destructor
//
//      Delete meshes, clear contexts and buffers
//
OsdMeshData::~OsdMeshData() 
{
    delete _hbrmesh;
    delete _mesh;
    delete _fvarDesc;
}

bool
uvSetNameIsValid( MFnMesh& meshFn, const MString& uvSet )
{
    // list should be short, just do linear search through existing names
    MStringArray setNames;
    meshFn.getUVSetNames(setNames);
    for (int i = 0; i < (int)setNames.length(); ++i) {
        if (setNames[i] == uvSet)
            return true;
    }
    return false;
}

// #### buildUVList
//
// Face-varying data expects a list of per-face per-vertex
// floats.  This method reads the UVs from the mesh and 
// concatenates them into such a list.
//
MStatus
OsdMeshData::buildUVList( MFnMesh& meshFn, std::vector<float>& uvList )
{
    MStatus status = MS::kSuccess;

    MItMeshPolygon polyIt( _meshDagPath );

    MFloatArray uArray;
    MFloatArray vArray;

    // If user hasn't given us a UV set, use the current one
    MString *uvSetPtr=NULL;
    if ( _uvSet.numChars() > 0 ) {
        if (uvSetNameIsValid(meshFn, _uvSet)) {
            uvSetPtr = &_uvSet;
        }
        else {
            MGlobal::displayWarning(MString("OpenSubdivShader:  uvSet \""+_uvSet+"\" does not exist."));
        }
    } else {
        uvSetPtr = NULL;
    }

    // pull UVs from Maya mesh
    status = meshFn.getUVs( uArray, vArray, uvSetPtr );
    MCHECK_RETURN(status, "OpenSubdivShader: Error reading UVs");

    if ( uArray.length() == 0 || vArray.length() == 0 )
    {
        MGlobal::displayWarning("OpenSubdivShader: Mesh has no UVs");
        return MS::kFailure;
    }

    // list of UV values
    uvList.clear();
    uvList.resize( meshFn.numFaceVertices()*2 );
    int uvListIdx = 0;

    // for each face-vertex copy UVs into list, adjusting for renderman orientation
    for ( polyIt.reset(); !polyIt.isDone(); polyIt.next() ) 
    { 
        unsigned int numPolyVerts = polyIt.polygonVertexCount();

        for ( unsigned int faceVertIdx = 0; 
                           faceVertIdx < numPolyVerts; 
                           faceVertIdx++ )
        {
            int uvIdx;
            polyIt.getUVIndex( faceVertIdx, uvIdx, uvSetPtr );
            // convert maya UV orientation to renderman orientation
            uvList[ uvListIdx++ ] = uArray[ uvIdx ];
            uvList[ uvListIdx++ ] = 1.0f - vArray[ uvIdx ];
        }
    }

    return status;
}


// #### rebuildHbrMeshIfNeeded
//
//      If the topology of the mesh changes, or any attributes that affect
//      how the mesh is computed the original HBR needs to be rebuilt
//      which will trigger a rebuild of the FAR mesh and subsequent buffers.
//
void
OsdMeshData::rebuildHbrMeshIfNeeded(OpenSubdivShader *shader)
{
    MStatus status = MS::kSuccess;

    if (!_meshTopoDirty && !shader->getHbrMeshDirty())
        return;

    MFnMesh meshFn(_meshDagPath);

    // Cache attribute values
    _level      = shader->getLevel();
    _kernel     = shader->getKernel();
    _adaptive   = shader->isAdaptive();
    _uvSet      = shader->getUVSet();

    // Get Maya vertex topology and crease data
    MIntArray vertexCount;
    MIntArray vertexList;
    meshFn.getVertices(vertexCount, vertexList);

    MUintArray edgeIds;
    MDoubleArray edgeCreaseData;
    meshFn.getCreaseEdges(edgeIds, edgeCreaseData);

    MUintArray vtxIds;
    MDoubleArray vtxCreaseData;
    meshFn.getCreaseVertices(vtxIds, vtxCreaseData);

    if (vertexCount.length() == 0) return;

    // Copy Maya vectors into std::vectors
    std::vector<int> numIndices(&vertexCount[0], &vertexCount[vertexCount.length()]);
    std::vector<int> faceIndices(&vertexList[0], &vertexList[vertexList.length()]);
    std::vector<int> vtxCreaseIndices(&vtxIds[0], &vtxIds[vtxIds.length()]);
    std::vector<double> vtxCreases(&vtxCreaseData[0], &vtxCreaseData[vtxCreaseData.length()]);
    std::vector<double> edgeCreases(&edgeCreaseData[0], &edgeCreaseData[edgeCreaseData.length()]);

    // Edge crease index is stored as pairs of vertex ids
    int nEdgeIds = edgeIds.length();
    std::vector<int> edgeCreaseIndices;
    edgeCreaseIndices.resize(nEdgeIds*2);
    for (int i = 0; i < nEdgeIds; ++i) {
        int2 vertices;
        status = meshFn.getEdgeVertices(edgeIds[i], vertices);
        if (status.error()) {
            MERROR(status, "OpenSubdivShader: Can't get edge vertices");
            continue;
        }
        edgeCreaseIndices[i*2] = vertices[0];
        edgeCreaseIndices[i*2+1] = vertices[1];
    }

    // Convert attribute enums to HBR enums (this is why the enums need to match)
    HbrMeshUtil::SchemeType hbrScheme = (HbrMeshUtil::SchemeType) shader->getScheme();
    OsdHbrMesh::InterpolateBoundaryMethod hbrInterpBoundary = 
            (OsdHbrMesh::InterpolateBoundaryMethod) shader->getInterpolateBoundary();
    OsdHbrMesh::InterpolateBoundaryMethod hbrInterpUVBoundary = 
            (OsdHbrMesh::InterpolateBoundaryMethod) shader->getInterpolateUVBoundary();


    // clear any existing face-varying descriptor
    if (_fvarDesc) {
        delete _fvarDesc;
        _fvarDesc = NULL;
    }

    // read UV data from maya and build per-face per-vert list of UVs for HBR face-varying data
    std::vector< float > uvList;
    status = buildUVList( meshFn, uvList );
    if (! status.error()) {
        // Create face-varying data descriptor.  The memory required for indices
        // and widths needs to stay alive as the HBR library only takes in the
        // pointers and assumes the client will maintain the memory so keep _fvarDesc
        // around as long as _hbrmesh is around.
        int fvarIndices[] = { 0, 1 };
        int fvarWidths[] = { 1, 1 };
        _fvarDesc = new FVarDataDesc( 2, fvarIndices, fvarWidths, 2, hbrInterpUVBoundary );
    }

    if (_fvarDesc && hbrScheme != HbrMeshUtil::kCatmark) {
        MGlobal::displayWarning("Face-varying not yet supported for Loop/Bilinear, using Catmull-Clark");
        hbrScheme = HbrMeshUtil::kCatmark;
    }

    // Convert Maya mesh to internal HBR representation
    _hbrmesh = ConvertToHBR(meshFn.numVertices(), numIndices, faceIndices,
                            vtxCreaseIndices, vtxCreases,
                            std::vector<int>(), std::vector<float>(),
                            edgeCreaseIndices, edgeCreases,
                            hbrInterpBoundary, 
                            hbrScheme,
                            false,                      // no ptex
                            _fvarDesc, 
                            _fvarDesc?&uvList:NULL);    // yes fvar (if have UVs)

    // note: GL function can't be used in prepareForDraw API.
    _needsInitializeMesh = true;

    // Mesh topology data is up to date
    _meshTopoDirty = false;
    shader->setHbrMeshDirty(false);
}


void
OsdMeshData::initializeMesh() 
{
    if (!_hbrmesh)
        return;

    OpenSubdiv::OsdMeshBitset bits;
    bits.set(OpenSubdiv::MeshAdaptive, _adaptive!=0);
    bits.set(OpenSubdiv::MeshFVarData, true);

    if (_kernel == kCPU) {
        _mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdCpuComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(_hbrmesh, 3, _level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        _mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdOmpComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(_hbrmesh, 3, _level, bits);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(_kernel == kCUDA) {
        _mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCudaGLVertexBuffer,
                                         OpenSubdiv::OsdCudaComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(_hbrmesh, 3, _level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(_kernel == kCL) {
        _mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCLGLVertexBuffer,
                                         OpenSubdiv::OsdCLComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(_hbrmesh, 3, _level, bits, g_clContext, g_clQueue);
#endif
    }

    delete _hbrmesh;
    _hbrmesh = NULL;

    _needsInitializeMesh = false;

    // get geometry from maya mesh
    MFnMesh meshFn(_meshDagPath);
    meshFn.getPoints(_pointArray);

    _needsUpdate = true;
}

void
OsdMeshData::prepare() 
{
    if (_needsInitializeMesh) {
        initializeMesh();
    }
}

void
OsdMeshData::updateGeometry(const MHWRender::MVertexBuffer *points)
{
    // Update coarse vertex

    int nCoarsePoints = _pointArray.length();

    GLuint mayaPositionVBO = *static_cast<GLuint*>(points->resourceHandle());
    int size = nCoarsePoints * 3 * sizeof(float);

    glBindBuffer(GL_COPY_READ_BUFFER, mayaPositionVBO);
    glBindBuffer(GL_COPY_WRITE_BUFFER, _mesh->BindVertexBuffer());
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, size);
    _mesh->Refine();

    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    _needsUpdate = false;
}
