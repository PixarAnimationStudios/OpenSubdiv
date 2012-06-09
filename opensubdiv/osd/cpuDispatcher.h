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
#ifndef OSD_CPU_DISPATCHER_H
#define OSD_CPU_DISPATCHER_H

#include "../osd/kernelDispatcher.h"

#include <GL/glew.h>

// Including omp.h makes sure we link against the correct OpenMP run-time
// library.  This is Windows specific, but we want a given header file to
// include the same set of files across all platforms if possible.
#include <omp.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCpuKernelDispatcher : public OsdKernelDispatcher
{
public:
    OsdCpuKernelDispatcher(int levels, int numVertexElements, int numVaryingElements);
    virtual ~OsdCpuKernelDispatcher();

    virtual void ApplyCatmarkFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;
    
    virtual void ApplyCatmarkEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;
    
    virtual void ApplyCatmarkVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;
    
    virtual void ApplyCatmarkVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;

    virtual void ApplyLoopEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;
    
    virtual void ApplyLoopVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;
    
    virtual void ApplyLoopVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;

    virtual void CopyTable(int tableIndex, size_t size, const void *ptr);

    virtual void BeginLaunchKernel();

    virtual void EndLaunchKernel();

    virtual void BindVertexBuffer(GLuint vertexBuffer, GLuint varyingBuffer);

    virtual void UpdateVertexBuffer(size_t size, void *ptr);

    virtual void UpdateVaryingBuffer(size_t size, void *ptr);

    virtual void MapVertexBuffer();

    virtual void MapVaryingBuffer();

    virtual void UnmapVertexBuffer();

    virtual void UnmapVaryingBuffer();

    virtual void Synchronize();

    static OsdKernelDispatcher * Create(int levels, int numVertexElements, int numVaryingElements) {
        return new OsdCpuKernelDispatcher(levels, numVertexElements, numVaryingElements);
    }
    static void Register() {
        Factory::GetInstance().Register("omp", Create);
    }

protected:
    
    struct DeviceTable {
        DeviceTable() : devicePtr(NULL) { }
       ~DeviceTable();

        void Copy(int size, const void *ptr);
        
	void *devicePtr;
    };

    std::vector<DeviceTable> _tables;

    GLuint _vertexBuffer, 
           _varyingBuffer;
	   
    float *_vbo, 
          *_varyingVbo;
	  
    int _vboSize, 
        _varyingVboSize,
        _numVertexElements, 
        _numVaryingElements;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_DISPATCHER_H */
