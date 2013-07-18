//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
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
