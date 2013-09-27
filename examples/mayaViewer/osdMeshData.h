//
//   Copyright 2013 Pixar
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

#ifndef EXAMPLES_MAYAVIEWER_OSDMESHDATA_H_
#define EXAMPLES_MAYAVIEWER_OSDMESHDATA_H_

#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/glMesh.h>

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

#include <maya/MDagPath.h>
#include <maya/MUserData.h>
#include <maya/MFloatPointArray.h>
#include <maya/MFnMesh.h>
#include <maya/MDoubleArray.h>
#include <maya/MHWGeometry.h>
#include <maya/MIntArray.h>
#include <maya/MUintArray.h>

class OpenSubdivShader;         // for getting attributes in rebuildHbrMeshIfNeeded
class FVarDataDesc;

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> OsdHbrMesh;



class OsdMeshData : public MUserData 
{

public:
    explicit OsdMeshData(const MDagPath& meshDagPath);
    virtual ~OsdMeshData();

    void rebuildHbrMeshIfNeeded(OpenSubdivShader *shader);
    void prepare();
    void updateGeometry(const MHWRender::MVertexBuffer *point);

    GLuint bindPositionVBO() { return _mesh->BindVertexBuffer(); }
    OpenSubdiv::OsdGLDrawContext * getDrawContext() { return _mesh->GetDrawContext(); }

    // accessors
    const MDagPath& getDagPath() { return _meshDagPath; }
    void setMeshTopoDirty() { _meshTopoDirty = true; }

public:

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
    void clearComputeContextAndVertexBuffer();

    MStatus buildUVList( MFnMesh& meshFn, std::vector<float>& uvList );

    MDagPath _meshDagPath;
    bool     _meshTopoDirty;

    OsdHbrMesh *_hbrmesh;

    OpenSubdiv::OsdGLMeshInterface *_mesh;

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
    InterpolateBoundaryType _interpUVBoundary;

    // UV face-varying
    MString       _uvSet;
    FVarDataDesc *_fvarDesc;

    bool _needsUpdate;
    bool _needsInitializeMesh;
};

#endif  // EXAMPLES_MAYAVIEWER_OSDMESHDATA_H_
