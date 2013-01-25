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

#if not defined(__APPLE__)
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
#endif

#include <maya/MFnMesh.h>

#include "osdPtexMeshData.h"

#include <osd/cpuDispatcher.h>
#include <osd/cpuComputeController.h>
extern OpenSubdiv::OsdCpuComputeController *g_cpuComputeController;

#ifdef OPENSUBDIV_HAS_OPENMP
#include <osd/ompDispatcher.h>
#include <osd/ompComputeController.h>
extern OpenSubdiv::OsdOmpComputeController *g_ompComputeController;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
#include <osd/clDispatcher.h>
#include <osd/clComputeController.h>
extern cl_context g_clContext;
extern cl_command_queue g_clQueue;
extern OpenSubdiv::OsdCLComputeController *g_clComputeController;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
#include <osd/cudaDispatcher.h>
#include <osd/cudaComputeController.h>
extern OpenSubdiv::OsdCudaComputeController *g_cudaComputeController;
#endif

#include <osd/glDrawContext.h>
#include <algorithm>
#include <vector>

#include "OpenSubdivPtexShader.h"
#include "hbrUtil.h"


// Constructor
OsdPtexMeshData::OsdPtexMeshData(const MDagPath& meshDagPath)
    : MUserData(false), 
      _meshDagPath(meshDagPath),
      _meshTopoDirty(true),
      _hbrmesh(NULL), 
      _farmesh(NULL),
      _drawContext(NULL),
      _level(0),
      _scheme(kCatmark),
      _kernel(kCPU),
      _adaptive(true),
      _interpBoundary(kInterpolateBoundaryNone),
      _needsUpdate(false),
      _needsInitializeMesh(false) 
{
    _cpuComputeContext = NULL;
    _cpuPositionBuffer = NULL;
    _cpuNormalBuffer = NULL;

#ifdef OPENSUBDIV_HAS_OPENCL
    _clComputeContext = NULL;
    _clPositionBuffer = NULL;
    _clNormalBuffer = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    _cudaComputeContext = NULL;
    _cudaPositionBuffer = NULL;
    _cudaNormalBuffer = NULL;
#endif
}

OsdPtexMeshData::~OsdPtexMeshData() {
    delete _hbrmesh;
    delete _farmesh;
    delete _drawContext;

    clearComputeContextAndVertexBuffer();
}

void
OsdPtexMeshData::clearComputeContextAndVertexBuffer() {
    delete _cpuComputeContext;
    _cpuComputeContext = NULL;
    delete _cpuPositionBuffer;
    _cpuPositionBuffer = NULL;
    delete _cpuNormalBuffer;
    _cpuNormalBuffer = NULL;

#ifdef OPENSUBDIV_HAS_CUDA
    delete _cudaComputeContext;
    _cudaComputeContext = NULL;
    delete _cudaPositionBuffer;
    _cudaPositionBuffer = NULL;
    delete _cudaNormalBuffer;
    _cudaNormalBuffer = NULL;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    delete _clComputeContext;
    _clComputeContext = NULL;
    delete _clPositionBuffer;
    _clPositionBuffer = NULL;
    delete _clNormalBuffer;
    _clNormalBuffer = NULL;
#endif
}

void
OsdPtexMeshData::rebuildHbrMeshIfNeeded(OpenSubdivPtexShader *shader)
{
    MStatus status;

    if (!_meshTopoDirty && !shader->getHbrMeshDirty())
        return;

    MFnMesh meshFn(_meshDagPath, &status);
    if (status != MS::kSuccess) return;

    int level = shader->getLevel();
    if (level < 1) level =1;

    SchemeType scheme = shader->getScheme();
    if (scheme == kLoop) scheme = kCatmark;  // XXX: avoid loop for now

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

    // Cache attribute values
    _level              = shader->getLevel();
    _scheme             = shader->getScheme();
    _kernel             = shader->getKernel();
    _adaptive           = shader->isAdaptive();
    _interpBoundary     = shader->getInterpolateBoundary();

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
            status.perror("ERROR can't get creased edge vertices");
            continue;
        }
        edgeCreaseIndices[i*2] = vertices[0];
        edgeCreaseIndices[i*2+1] = vertices[1];
    }

    // Convert attribute enums to HBR enums (this is why the enums need to match)
    // XXX use some sort of built-in transmorgification avoid assumption?
    HbrMeshUtil::SchemeType hbrScheme = (HbrMeshUtil::SchemeType) _scheme;
    OsdHbrMesh::InterpolateBoundaryMethod hbrInterpBoundary = 
            (OsdHbrMesh::InterpolateBoundaryMethod) _interpBoundary;

    // Convert Maya mesh to internal HBR representation
    _hbrmesh = ConvertToHBR(meshFn.numVertices(), numIndices, faceIndices,
                            vtxCreaseIndices, vtxCreases,
                            std::vector<int>(), std::vector<float>(),
                            edgeCreaseIndices, edgeCreases,
                            hbrInterpBoundary, 
                            hbrScheme,
                            true );                     // add ptex indices to HBR

    // note: GL function can't be used in prepareForDraw API.
    _needsInitializeMesh = true;

    // Mesh topology data is up to date
    _meshTopoDirty = false;
    shader->setHbrMeshDirty(false);
}


void
OsdPtexMeshData::initializeMesh() 
{
    if (!_hbrmesh)
        return;

    // create far mesh
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex>
        meshFactory(_hbrmesh, _level, _adaptive);

    _farmesh = meshFactory.Create(true /*ptex coords*/);

    delete _hbrmesh;
    _hbrmesh = NULL;

    int numTotalVertices = _farmesh->GetNumVertices();

    // create context and vertex buffer
    clearComputeContextAndVertexBuffer();

    if (_kernel == kCPU) {
        _cpuComputeContext = OpenSubdiv::OsdCpuComputeContext::Create(_farmesh);
        _cpuPositionBuffer = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, numTotalVertices);
        if (not _adaptive)
            _cpuNormalBuffer = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, numTotalVertices);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        _cpuComputeContext = OpenSubdiv::OsdCpuComputeContext::Create(_farmesh);
        _cpuPositionBuffer = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, numTotalVertices);
        if (not _adaptive)
            _cpuNormalBuffer = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, numTotalVertices);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        _cudaComputeContext = OpenSubdiv::OsdCudaComputeContext::Create(_farmesh);
        _cudaPositionBuffer = OpenSubdiv::OsdCudaGLVertexBuffer::Create(3, numTotalVertices);
        if (not _adaptive)
            _cudaNormalBuffer = OpenSubdiv::OsdCudaGLVertexBuffer::Create(3, numTotalVertices);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        _clComputeContext = OpenSubdiv::OsdCLComputeContext::Create(_farmesh, g_clContext);
        _clPositionBuffer = OpenSubdiv::OsdCLGLVertexBuffer::Create(3, numTotalVertices,
                                                                    g_clContext);
        if (not _adaptive)
            _clNormalBuffer = OpenSubdiv::OsdCLGLVertexBuffer::Create(3, numTotalVertices,
                                                                      g_clContext);
#endif
    }

    _needsInitializeMesh = false;

    // get geometry from maya mesh
    MFnMesh meshFn(_meshDagPath);
    meshFn.getPoints(_pointArray);

    _needsUpdate = true;
}

void
OsdPtexMeshData::updateGeometry(const MHWRender::MVertexBuffer *points,
                                const MHWRender::MVertexBuffer *normals) 
{
    // Update coarse vertex

    int nCoarsePoints = _pointArray.length();

    GLuint mayaPositionVBO = *static_cast<GLuint*>(points->resourceHandle());
    GLuint mayaNormalVBO = normals ? *static_cast<GLuint*>(normals->resourceHandle()) : NULL;
    int size = nCoarsePoints * 3 * sizeof(float);

    if (_kernel == kCPU || _kernel == kOPENMP) {
        float *d_pos = _cpuPositionBuffer->BindCpuBuffer();
        glBindBuffer(GL_ARRAY_BUFFER, mayaPositionVBO);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, size, d_pos);
        g_cpuComputeController->Refine(_cpuComputeContext, _cpuPositionBuffer);

        if (not _adaptive) {
            d_pos = _cpuNormalBuffer->BindCpuBuffer();
            glBindBuffer(GL_ARRAY_BUFFER, mayaNormalVBO);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, size, d_pos);

            g_cpuComputeController->Refine(_cpuComputeContext, _cpuNormalBuffer);
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);

#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        glBindBuffer(GL_COPY_READ_BUFFER, mayaPositionVBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, _cudaPositionBuffer->BindVBO());
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                            0, 0, size);
        g_cudaComputeController->Refine(_cudaComputeContext, _cudaPositionBuffer);

        if (not _adaptive) {
            glBindBuffer(GL_COPY_READ_BUFFER, mayaNormalVBO);
            glBindBuffer(GL_COPY_WRITE_BUFFER, _cudaNormalBuffer->BindVBO());
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                0, 0, size);

            g_cudaComputeController->Refine(_cudaComputeContext, _cudaNormalBuffer);
        }

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        glBindBuffer(GL_COPY_READ_BUFFER, mayaPositionVBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, _clPositionBuffer->BindVBO());
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                            0, 0, size);
        g_clComputeController->Refine(_clComputeContext, _clPositionBuffer);

        if (not _adaptive) {
            glBindBuffer(GL_COPY_READ_BUFFER, mayaNormalVBO);
            glBindBuffer(GL_COPY_WRITE_BUFFER, _clNormalBuffer->BindVBO());
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                0, 0, size);
            g_clComputeController->Refine(_clComputeContext, _clNormalBuffer);
        }

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
#endif
    }

    _needsUpdate = false;
}

void
OsdPtexMeshData::initializeIndexBuffer() 
{
    // create element array buffer
    delete _drawContext;

    if (_kernel == kCPU) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _cpuPositionBuffer,
                                                            true);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _cpuPositionBuffer,
                                                            true);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _cudaPositionBuffer,
                                                            true);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        _drawContext = OpenSubdiv::OsdGLDrawContext::Create(_farmesh,
                                                            _clPositionBuffer,
                                                            true);
#endif
    } else {
        assert(false);
    }
}

void
OsdPtexMeshData::prepare() 
{
    if (_needsInitializeMesh) {
        initializeMesh();
        initializeIndexBuffer();
    }
}

GLuint
OsdPtexMeshData::bindPositionVBO() 
{
    if (_kernel == kCPU) {
        return _cpuPositionBuffer->BindVBO();
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        return _cpuPositionBuffer->BindVBO();
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        return _cudaPositionBuffer->BindVBO();
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        return _clPositionBuffer->BindVBO();
#endif
    }
    return 0;
}

GLuint
OsdPtexMeshData::bindNormalVBO() 
{
    if (_adaptive) return 0;
    if (_kernel == kCPU) {
        return _cpuNormalBuffer->BindVBO();
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (_kernel == kOPENMP) {
        return _cpuNormalBuffer->BindVBO();
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (_kernel == kCUDA) {
        return _cudaNormalBuffer->BindVBO();
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (_kernel == kCL) {
        return _clNormalBuffer->BindVBO();
#endif
    }
    return 0;
}
