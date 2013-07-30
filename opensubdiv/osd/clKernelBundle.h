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
#ifndef OSD_CL_KERNEL_BUNDLE_H
#define OSD_CL_KERNEL_BUNDLE_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/vertexDescriptor.h"

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCLKernelBundle : OsdNonCopyable<OsdCLKernelBundle> {

public:
    OsdCLKernelBundle();
    ~OsdCLKernelBundle();

    bool Compile(cl_context clContext,
                 int numVertexElements, int numVaryingElements);

    cl_kernel GetBilinearEdgeKernel() const   { return _clBilinearEdge; }

    cl_kernel GetBilinearVertexKernel() const { return _clBilinearVertex; }

    cl_kernel GetCatmarkFaceKernel() const    { return _clCatmarkFace; }

    cl_kernel GetCatmarkEdgeKernel() const    { return _clCatmarkEdge; }

    cl_kernel GetCatmarkVertexKernelA() const { return _clCatmarkVertexA; }

    cl_kernel GetCatmarkVertexKernelB() const { return _clCatmarkVertexB; }

    cl_kernel GetLoopEdgeKernel() const       { return _clLoopEdge; }

    cl_kernel GetLoopVertexKernelA() const    { return _clLoopVertexA; }

    cl_kernel GetLoopVertexKernelB() const    { return _clLoopVertexB; }

    cl_kernel GetVertexEditAdd() const        { return _clVertexEditAdd; }

    struct Match {
    
        /// Constructor
        Match(int numVertexElements, int numVaryingElements)
            : vdesc(numVertexElements, numVaryingElements) {
        }
        
        bool operator() (OsdCLKernelBundle const *kernel) {
            return vdesc == kernel->_vdesc;
        }
        
        OsdVertexDescriptor vdesc;
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
              _clLoopVertexB,
              _clVertexEditAdd;

    OsdVertexDescriptor _vdesc;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CL_KERNEL_BUNDLE_H
