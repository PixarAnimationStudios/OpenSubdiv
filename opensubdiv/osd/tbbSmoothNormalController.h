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

#ifndef OSD_TBB_SMOOTHNORMAL_CONTROLLER_H
#define OSD_TBB_SMOOTHNORMAL_CONTROLLER_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/cpuSmoothNormalContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class TbbSmoothNormalController {

public:

    /// Constructor
    TbbSmoothNormalController();

    /// Destructor
    ~TbbSmoothNormalController();

    /// Computes smooth vertex normals
    template<class VERTEX_BUFFER>
    void SmootheNormals( CpuSmoothNormalContext * context,
                         VERTEX_BUFFER * iBuffer, int iOfs,
                         VERTEX_BUFFER * oBuffer, int oOfs ) {

         if (not context) return;

         context->Bind(iBuffer, iOfs, oBuffer, oOfs);

         _smootheNormals(context);

         context->Unbind();
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

private:

    void _smootheNormals(CpuSmoothNormalContext * context);
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CPU_SMOOTHNORMAL_CONTROLLER_H
