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
#include <string.h>

#include "../version.h"
#include "../osd/mesh.h"
#include "../osd/local.h"
#include "../osd/dump.cpp"
#include "../osd/kernelDispatcher.h"
#include "../osd/cpuDispatcher.h"

#include "../far/meshFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdMesh::OsdMesh(int numVertexElements, int numVaryingElements) 
    : _numVertexElements(numVertexElements), _numVaryingElements(numVaryingElements), _mMesh(NULL), _dispatcher(NULL) {
    
    glGenBuffers(1, &_vertexBuffer);
    glGenBuffers(1, &_varyingBuffer);
}

OsdMesh::~OsdMesh() {

    if(_dispatcher) delete _dispatcher;
    if(_mMesh) delete _mMesh;

    glDeleteBuffers(1, &_vertexBuffer);
    glDeleteBuffers(1, &_varyingBuffer);
}

bool
OsdMesh::Create(OsdHbrMesh *hbrMesh, int level, const std::string &kernel) {

    if (_dispatcher)
        delete _dispatcher;
    _dispatcher = OsdKernelDispatcher::CreateKernelDispatcher(kernel, level, _numVertexElements, _numVaryingElements);
    if(_dispatcher == NULL){
        OSD_ERROR("Unknown kernel %s\n", kernel.c_str());
        return false;
    }

    _level = level;
        
    // create Far mesh
    OSD_DEBUG("Create MeshFactory\n");

    FarMeshFactory<OsdVertex> meshFactory(hbrMesh, _level);

    _mMesh = meshFactory.Create(_dispatcher);
    
    OSD_DEBUG("PREP: NumCoarseVertex = %d\n", _mMesh->GetNumCoarseVertices());
    OSD_DEBUG("PREP: NumVertex = %d\n", _mMesh->GetNumVertices());
    OSD_DEBUG("PREP: NumTables = %d\n", _mMesh->GetNumSubdivisionTables());

    const FarSubdivisionTables<OsdVertex>* table = _mMesh->GetSubdivision();

    _dispatcher->UpdateTable(OsdKernelDispatcher::E_IT, table->Get_E_IT());
    _dispatcher->UpdateTable(OsdKernelDispatcher::V_IT, table->Get_V_IT());
    _dispatcher->UpdateTable(OsdKernelDispatcher::V_ITa, table->Get_V_ITa());
    _dispatcher->UpdateTable(OsdKernelDispatcher::E_W, table->Get_E_W());
    _dispatcher->UpdateTable(OsdKernelDispatcher::V_W, table->Get_V_W());

    if ( const FarCatmarkSubdivisionTables<OsdVertex> * cctable = 
       dynamic_cast<const FarCatmarkSubdivisionTables<OsdVertex>*>(table) ) {
        // catmark
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_IT, cctable->Get_F_IT());
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_ITa, cctable->Get_F_ITa());
    } else if ( const FarBilinearSubdivisionTables<OsdVertex> * btable = 
       dynamic_cast<const FarBilinearSubdivisionTables<OsdVertex>*>(table) ) {
        // bilinear
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_IT, btable->Get_F_IT());
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_ITa, btable->Get_F_ITa());
    } else {
        // XXX for glsl shader...
        _dispatcher->CopyTable(OsdKernelDispatcher::F_IT, 0, NULL);
        _dispatcher->CopyTable(OsdKernelDispatcher::F_ITa, 0, NULL);
    }

    CHECK_GL_ERROR("Mesh, update tables\n");

    // vbo prep
    int vertexSize = _mMesh->GetNumVertices() * _numVertexElements * sizeof(float);
    glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertexSize, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_GL_ERROR("Mesh, vertex buffer %d\n", _vertexBuffer);

    int varyingSize = _mMesh->GetNumVertices() * _numVaryingElements * sizeof(float);
    glBindBuffer(GL_ARRAY_BUFFER, _varyingBuffer);
    glBufferData(GL_ARRAY_BUFFER, varyingSize, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_GL_ERROR("Mesh, varying buffer %d\n", _varyingBuffer);
        
    _dispatcher->BindVertexBuffer(_vertexBuffer, varyingSize ? _varyingBuffer : 0);

    return true;
}

void
OsdMesh::UpdatePoints(const std::vector<float> &points) {

    int numCoarseVertices = _mMesh->GetNumCoarseVertices();
    if(numCoarseVertices * _numVertexElements != points.size()) {
        OSD_ERROR("UpdatePoints points size mismatch %d != %d\n", numCoarseVertices, (int)points.size());
        return;
    }

    float * updateVertexBuffer = new float[GetNumCoarseVertices() * _numVertexElements];
    float *p = updateVertexBuffer;
    for (int i = 0; i < numCoarseVertices; ++i)
        for (int j = 0; j < _numVertexElements; ++j)
            *p++ = points[i*_numVertexElements+j];

    // send to gpu
    _dispatcher->MapVertexBuffer();

    // copy _updateVertexBuffer to each kernel's local memory
    int size = numCoarseVertices * _numVertexElements * sizeof(float);
    _dispatcher->UpdateVertexBuffer(size, updateVertexBuffer);

    _dispatcher->UnmapVertexBuffer();

    delete[] updateVertexBuffer;
}

void
OsdMesh::UpdateVaryings(const std::vector<float> &varyings) {

    int numCoarseVertices = _mMesh->GetNumCoarseVertices();
    if (numCoarseVertices * _numVaryingElements != varyings.size()) {
        OSD_ERROR("UpdateVaryings array size mismatch %d != %d\n", numCoarseVertices, (int)varyings.size());
        return;
    }

    // send to gpu
    _dispatcher->MapVaryingBuffer();

    int size = numCoarseVertices * _numVaryingElements * sizeof(float);
    _dispatcher->UpdateVaryingBuffer(size, (void*)&varyings[0]);

    _dispatcher->UnmapVaryingBuffer();
}

void
OsdMesh::Subdivide() {

    _dispatcher->MapVertexBuffer();
    _dispatcher->MapVaryingBuffer();
    _dispatcher->BeginLaunchKernel();

    _mMesh->Subdivide(_level+1);

    _dispatcher->EndLaunchKernel();
    _dispatcher->UnmapVertexBuffer();
    _dispatcher->UnmapVaryingBuffer();
}

void
OsdMesh::Synchronize() {

    _dispatcher->Synchronize();
}

void
OsdMesh::GetRefinedPoints(std::vector<float> &refinedPoints) {

    glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
    
    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    
    int numTotalVertices = _mMesh->GetNumVertices();
    if (size != numTotalVertices*_numVertexElements*sizeof(float)) {
        OSD_ERROR("vertex count mismatch %d != %d\n", (int)(size/_numVertexElements/sizeof(float)), numTotalVertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return;
    }

    refinedPoints.resize(numTotalVertices*_numVertexElements);
    float *p = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if (p)
        for (int i = 0; i < numTotalVertices; ++i)
            for (int j = 0; j < _numVertexElements; ++j)
                refinedPoints[i*_numVertexElements+j] = *p++;

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

