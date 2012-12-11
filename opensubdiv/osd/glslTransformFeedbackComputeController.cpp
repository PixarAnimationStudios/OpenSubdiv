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
#else
    #include <OpenGL/gl3.h>
#endif

#include "../osd/glslTransformFeedbackComputeController.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/glslTransformFeedbackDispatcher.h"

#include <algorithm>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLSLTransformFeedbackComputeController::OsdGLSLTransformFeedbackComputeController() {
}

OsdGLSLTransformFeedbackComputeController::~OsdGLSLTransformFeedbackComputeController() {

    for (std::vector<OsdGLSLTransformFeedbackKernelBundle*>::iterator it =
             _kernelRegistry.begin();
         it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
}

void
OsdGLSLTransformFeedbackComputeController::Synchronize() {

    glFinish();
}

OsdGLSLTransformFeedbackKernelBundle *
OsdGLSLTransformFeedbackComputeController::getKernels(int numVertexElements,
                                     int numVaryingElements) {

    std::vector<OsdGLSLTransformFeedbackKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdGLSLTransformFeedbackKernelBundle::Match(numVertexElements,
                                                numVaryingElements));
    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdGLSLTransformFeedbackKernelBundle *kernelBundle = new OsdGLSLTransformFeedbackKernelBundle();
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(numVertexElements, numVaryingElements);
        return kernelBundle;
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
