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
#ifndef OSD_GCD_DISPATCHER_H
#define OSD_GCD_DISPATCHER_H

#include <dispatch/dispatch.h>

#include "../version.h"

#include "../osd/vertex.h"
#include "../far/dispatcher.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCpuComputeContext;

class OsdGcdKernelDispatcher : public FarDispatcher<OsdVertex>
{
public:
    OsdGcdKernelDispatcher();

    virtual ~OsdGcdKernelDispatcher();

    void Refine(FarMesh<OsdVertex> * mesh, OsdCpuComputeContext *context) const;

    static OsdGcdKernelDispatcher * GetInstance();

protected:
    virtual void ApplyBilinearFaceVerticesKernel(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyBilinearEdgeVerticesKernel(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyBilinearVertexVerticesKernel(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;


    virtual void ApplyCatmarkFaceVerticesKernel(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyCatmarkEdgeVerticesKernel(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyCatmarkVertexVerticesKernelB(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyCatmarkVertexVerticesKernelA(
        FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
        int start, int end, void * clientdata) const;


    virtual void ApplyLoopEdgeVerticesKernel(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyLoopVertexVerticesKernelB(
        FarMesh<OsdVertex> * mesh, int offset, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyLoopVertexVerticesKernelA(
        FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
        int start, int end, void * clientdata) const;

    virtual void ApplyVertexEdits(
        FarMesh<OsdVertex> *mesh, int offset, int level,
        void * clientdata) const;

private:
    dispatch_queue_t _gcd_queue;

};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_GCD_DISPATCHER_H
