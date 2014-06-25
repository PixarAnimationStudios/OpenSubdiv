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

#ifndef OSD_CL_KERNEL_BUNDLE_H
#define OSD_CL_KERNEL_BUNDLE_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/opencl.h"
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCLKernelBundle : OsdNonCopyable<OsdCLKernelBundle> {

public:
    OsdCLKernelBundle();
    ~OsdCLKernelBundle();

    bool Compile(cl_context clContext,
                 OsdVertexBufferDescriptor const &vertexDesc,
                 OsdVertexBufferDescriptor const &varyingDesc);

    cl_kernel GetBilinearEdgeKernel() const              { return _clBilinearEdge; }

    cl_kernel GetBilinearVertexKernel() const            { return _clBilinearVertex; }

    cl_kernel GetCatmarkFaceKernel() const               { return _clCatmarkFace; }

    cl_kernel GetCatmarkQuadFaceKernel() const           { return _clCatmarkQuadFace; }

    cl_kernel GetCatmarkTriQuadFaceKernel() const        { return _clCatmarkQuadFace; }

    cl_kernel GetCatmarkEdgeKernel() const               { return _clCatmarkEdge; }

    cl_kernel GetCatmarkRestrictedEdgeKernel() const     { return _clCatmarkRestrictedEdge; }

    cl_kernel GetCatmarkVertexKernelA() const            { return _clCatmarkVertexA; }

    cl_kernel GetCatmarkVertexKernelB() const            { return _clCatmarkVertexB; }

    cl_kernel GetCatmarkRestrictedVertexKernelA() const  { return _clCatmarkRestrictedVertexA; }

    cl_kernel GetCatmarkRestrictedVertexKernelB1() const { return _clCatmarkRestrictedVertexB1; }

    cl_kernel GetCatmarkRestrictedVertexKernelB2() const { return _clCatmarkRestrictedVertexB2; }

    cl_kernel GetLoopEdgeKernel() const                  { return _clLoopEdge; }

    cl_kernel GetLoopVertexKernelA() const               { return _clLoopVertexA; }

    cl_kernel GetLoopVertexKernelB() const               { return _clLoopVertexB; }

    cl_kernel GetVertexEditAdd() const                   { return _clVertexEditAdd; }

    struct Match {
        /// Constructor
        Match(OsdVertexBufferDescriptor const &vertex,
              OsdVertexBufferDescriptor const &varying)
            : vertexDesc(vertex), varyingDesc(varying) {
        }

        bool operator() (OsdCLKernelBundle const *kernel) {
            // offset is dynamic. just comparing length and stride here,
            // returns true if they are equal
            return (vertexDesc.length == kernel->_numVertexElements and
                    vertexDesc.stride == kernel->_vertexStride and
                    varyingDesc.length == kernel->_numVaryingElements and
                    varyingDesc.stride == kernel->_varyingStride);
        }

        OsdVertexBufferDescriptor vertexDesc;
        OsdVertexBufferDescriptor varyingDesc;
    };

    friend struct Match;

protected:
    cl_program _clProgram;

    cl_kernel _clBilinearEdge,
              _clBilinearVertex,
              _clCatmarkFace,
              _clCatmarkQuadFace,
              _clCatmarkTriQuadFace,
              _clCatmarkEdge,
              _clCatmarkRestrictedEdge,
              _clCatmarkVertexA,
              _clCatmarkVertexB,
              _clCatmarkRestrictedVertexA,
              _clCatmarkRestrictedVertexB1,
              _clCatmarkRestrictedVertexB2,
              _clLoopEdge,
              _clLoopVertexA,
              _clLoopVertexB,
              _clVertexEditAdd;

    int _numVertexElements;
    int _vertexStride;
    int _numVaryingElements;
    int _varyingStride;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CL_KERNEL_BUNDLE_H
