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
#ifndef OSD_CPU_COMPUTE_CONTEXT_H
#define OSD_CPU_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/table.h"
#include "../far/subdivisionTables.h"
#include "../far/vertexEditTables.h"
#include "../osd/computeContext.h"
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexDescriptor;

class OsdCpuComputeContext : public OsdComputeContext {
public:
    static OsdCpuComputeContext * Create(FarMesh<OsdVertex> *farmesh);

    virtual ~OsdCpuComputeContext();

    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying) {

        _currentVertexBuffer = vertex ? vertex->BindCpuBuffer() : 0;
        _currentVaryingBuffer = varying ? varying->BindCpuBuffer() : 0;

        int numVertexElements = vertex ? vertex->GetNumElements() : 0;
        int numVaryingElements = varying ? varying->GetNumElements() : 0;
        _vdesc = new OsdVertexDescriptor(numVertexElements, numVaryingElements);
    }

    void Unbind() {
        _currentVertexBuffer = 0;
        _currentVaryingBuffer = 0;

        delete _vdesc;
        _vdesc = 0;
    }

    const void * GetTablePtr(int tableIndex, int level) const;

    OsdVertexDescriptor * GetVertexDescriptor() const;

    int GetNumEditTables() const;

    const FarVertexEditTables<OsdVertex>::VertexEditBatch *
        GetEditTable(int tableIndex) const;

    float * GetCurrentVertexBuffer() const;

    float * GetCurrentVaryingBuffer() const;

protected:
    explicit OsdCpuComputeContext(FarMesh<OsdVertex> *farMesh);

private:
    // XXX: shared pointer with farmesh?
    FarSubdivisionTables<OsdVertex> const *_tables;
    FarVertexEditTables<OsdVertex> const *_editTables;

    float *_currentVertexBuffer, *_currentVaryingBuffer;

    OsdVertexDescriptor *_vdesc;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CPU_COMPUTE_CONTEXT_H


