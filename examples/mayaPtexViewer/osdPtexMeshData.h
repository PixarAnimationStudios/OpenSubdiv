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
#ifndef EXAMPLES_MAYAPTEXVIEWER_OSDPTEXMESHDATA_H_
#define EXAMPLES_MAYAPTEXVIEWER_OSDPTEXMESHDATA_H_

#include <far/meshFactory.h>
#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/glDrawContext.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
#endif

// #include <maya/MViewport2Renderer.h>
#include <maya/MDagPath.h>
#include <maya/MUserData.h>
#include <maya/MFloatPointArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MHWGeometry.h>
#include <maya/MIntArray.h>
#include <maya/MUintArray.h>

class OpenSubdivPtexShader;         // for getting attributes in rebuildHbrMeshIfNeeded

// XXX replicated from hbrUtil.h
typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> OsdHbrMesh;


class OsdPtexMeshData : public MUserData 
{
public:
    explicit OsdPtexMeshData(const MDagPath& meshDagPath);
    virtual ~OsdPtexMeshData();

    void rebuildHbrMeshIfNeeded(OpenSubdivPtexShader *shader);
    void prepare();
    void updateGeometry(const MHWRender::MVertexBuffer *point,
                        const MHWRender::MVertexBuffer *normal);

    GLuint bindPositionVBO();
    GLuint bindNormalVBO();
    OpenSubdiv::OsdGLDrawContext * getDrawContext() { return _drawContext; }

    // accessors
    const MDagPath& getDagPath() { return _meshDagPath; }
    void setMeshTopoDirty() { _meshTopoDirty = true; }


public:

    // XXX should these be here or somewhere outside of plugin?
    //     needed by both shader definition (attrs) and implementation
    enum KernelType { kCPU = 0,
                      kOPENMP = 1,
                      kCUDA = 2,
                      kCL = 3,
                      kGLSL = 4,
                      kGLSLCompute = 5 };

    enum SchemeType { 
                // needs to match HbrMeshUtil::SchemeType enums (or whatever hbrUtil uses)
                kCatmark=0, 
                kLoop=1, 
                kBilinear=2 };

    enum InterpolateBoundaryType {
                // needs to match OsdHbrMesh::InterpolateBoundaryMethod enums
                kInterpolateBoundaryNone,
                kInterpolateBoundaryEdgeOnly,
                kInterpolateBoundaryEdgeAndCorner,
                kInterpolateBoundaryAlwaysSharp };

private:
    void initializeMesh();
    void initializeIndexBuffer();
    void clearComputeContextAndVertexBuffer();

    MDagPath _meshDagPath;
    bool     _meshTopoDirty;

    OsdHbrMesh *_hbrmesh;
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *_farmesh;

    OpenSubdiv::OsdCpuComputeContext *_cpuComputeContext;
    OpenSubdiv::OsdCpuGLVertexBuffer *_cpuPositionBuffer, *_cpuNormalBuffer;
#ifdef OPENSUBDIV_HAS_OPENCL
    OpenSubdiv::OsdCLComputeContext *_clComputeContext;
    OpenSubdiv::OsdCLGLVertexBuffer *_clPositionBuffer, *_clNormalBuffer;
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    OpenSubdiv::OsdCudaComputeContext *_cudaComputeContext;
    OpenSubdiv::OsdCudaGLVertexBuffer *_cudaPositionBuffer, *_cudaNormalBuffer;
#endif

    OpenSubdiv::OsdGLDrawContext *_drawContext;

    MFloatPointArray _pointArray;

    // topology cache
    MIntArray _vertexList;
    MUintArray _edgeIds, _vtxIds;
    MDoubleArray _edgeCreaseData, _vtxCreaseData;
    int _level;
    int _scheme;
    int _kernel;
    bool _adaptive;
    InterpolateBoundaryType _interpBoundary;

    bool _needsUpdate;
    bool _needsInitializeMesh;
};

#endif  // EXAMPLES_MAYAPTEXVIEWER_OSDPTEXMESHDATA_H_
