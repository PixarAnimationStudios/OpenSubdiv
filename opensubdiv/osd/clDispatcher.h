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
#ifndef OSD_CL_DISPATCHER_H
#define OSD_CL_DISPATCHER_H

#include "../version.h"
#include "../osd/kernelDispatcher.h"
#include "../osd/vertexBuffer.h"

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdClVertexBuffer : public OsdGpuVertexBuffer {
public:
    OsdClVertexBuffer(int numElements, int numVertices, cl_context context, cl_command_queue queue);
    virtual ~OsdClVertexBuffer();

    virtual void UpdateData(const float *src, int numVertices);
    void Map();
    void Unmap();

    cl_mem const * GetClBuffer() const { return &_clVbo; }

protected:
    cl_mem _clVbo;
    cl_command_queue _clQueue;
};

class OsdClKernelDispatcher : public OsdKernelDispatcher
{
public:
    OsdClKernelDispatcher(int levels);
    virtual ~OsdClKernelDispatcher();


    virtual void ApplyBilinearFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyBilinearEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyBilinearVertexVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;


    virtual void ApplyCatmarkFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;


    virtual void ApplyLoopEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyLoopVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyLoopVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;

    virtual void ApplyVertexEdit(FarMesh<OsdVertex> *mesh, int offset, int level, void * clientdata) const {}


    virtual void CopyTable(int tableIndex, size_t size, const void *ptr);

    virtual void AllocateEditTables(int n) {}

    virtual void UpdateEditTable(int tableIndex, const FarTable<unsigned int> &offsets, const FarTable<float> &values,
                                 int operation, int primVarOffset, int primVarWidth) {}

    virtual void OnKernelLaunch() {}

    virtual void OnKernelFinish() {}

    virtual OsdVertexBuffer *InitializeVertexBuffer(int numElements, int count);

    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);

    virtual void UnbindVertexBuffer();

    virtual void Synchronize();

    static OsdKernelDispatcher * Create(int levels) {
        return new OsdClKernelDispatcher(levels);
    }

    static void Register() {
        Factory::GetInstance().Register(Create, kCL);
    }

protected:
    class ClKernel {
    public:
        ClKernel();
        ~ClKernel();

        bool Compile(cl_context clContext, int numVertexElements, int numVaryingElements);

        cl_kernel GetBilinearEdgeKernel() const   { return _clBilinearEdge; }

        cl_kernel GetBilinearVertexKernel() const { return _clBilinearVertex; }

        cl_kernel GetCatmarkFaceKernel() const    { return _clCatmarkFace; }

        cl_kernel GetCatmarkEdgeKernel() const    { return _clCatmarkEdge; }

        cl_kernel GetCatmarkVertexKernelA() const { return _clCatmarkVertexA; }

        cl_kernel GetCatmarkVertexKernelB() const { return _clCatmarkVertexB; }

        cl_kernel GetLoopEdgeKernel() const       { return _clLoopEdge; }

        cl_kernel GetLoopVertexKernelA() const    { return _clLoopVertexA; }

        cl_kernel GetLoopVertexKernelB() const    { return _clLoopVertexB; }

        struct Match {
        Match(int numVertexElements, int numVaryingElements) :
            _numVertexElements(numVertexElements), _numVaryingElements(numVaryingElements) {}
            bool operator() (const ClKernel &kernel) {
                return (kernel._numVertexElements == _numVertexElements
                        && kernel._numVaryingElements == _numVaryingElements);
            }
            int _numVertexElements, _numVaryingElements;
        };

        friend struct Match;

    protected:
        cl_program _clProgram;

        cl_kernel _clBilinearEdge,
                  _clBilinearVertex,
                  _clCatmarkFace,
                  _clCatmarkEdge,
                  _clCatmarkVertexA,
                  _clCatmarkVertexB,
                  _clLoopEdge,
                  _clLoopVertexA,
                  _clLoopVertexB;

        int _numVertexElements,
            _numVaryingElements;
    };

    struct DeviceTable
    {
        DeviceTable() : devicePtr(NULL) {}
        ~DeviceTable();

        void Copy(cl_context context, int size, const void *ptr);

        cl_mem devicePtr;
    };

    cl_mem const * GetVertexBuffer() const {

        static cl_mem nul = NULL;
        return _currentVertexBuffer ? _currentVertexBuffer->GetClBuffer() : &nul;
    }

    cl_mem const * GetVaryingBuffer() const {

        static cl_mem nul = NULL;
        return _currentVaryingBuffer ? _currentVaryingBuffer->GetClBuffer() : &nul;
    }

    void initCL();

    void uninitCL();

    OsdClVertexBuffer *_currentVertexBuffer,
                      *_currentVaryingBuffer;

    std::vector<DeviceTable> _tables;

    ClKernel * _clKernel;

    // XXX: context and queue should be moved to client code
    static cl_context       _clContext;
    static cl_command_queue _clQueue;
    static cl_device_id     _clDevice;

    // static shader registry (XXX tentative..)
    static std::vector<ClKernel> kernelRegistry;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_CL_DISPATCHER_H
