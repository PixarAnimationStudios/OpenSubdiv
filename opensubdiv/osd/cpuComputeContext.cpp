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

#include "../far/mesh.h"
#include "../far/dispatcher.h"
#include "../far/catmarkSubdivisionTables.h"
#include "../far/bilinearSubdivisionTables.h"

#include "../osd/cpuComputeContext.h"
#include "../osd/cpuKernel.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/error.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuComputeContext::OsdCpuComputeContext(FarMesh<OsdVertex> *farMesh)
    : OsdComputeContext(farMesh) {

    _tables = farMesh->GetSubdivisionTables();
    _editTables = farMesh->GetVertexEdit();
    _vdesc = 0;
    _currentVertexBuffer = 0;
    _currentVaryingBuffer = 0;
}

OsdCpuComputeContext::~OsdCpuComputeContext() {

    if (_vdesc) delete _vdesc;
}

const void *
OsdCpuComputeContext::GetTablePtr(int tableIndex, int level) const {

    if (tableIndex == Table::E_IT) {
        return _tables->Get_E_IT()[level];
    } else if (tableIndex == Table::E_W) {
        return _tables->Get_E_W()[level];
    } else if (tableIndex == Table::V_ITa) {
        return _tables->Get_V_ITa()[level];
    } else if (tableIndex == Table::V_IT) {
        return _tables->Get_V_IT()[level];
    } else if (tableIndex == Table::V_W) {
        return _tables->Get_V_W()[level];
    } else {
        const FarCatmarkSubdivisionTables<OsdVertex> * ccTables =
            dynamic_cast<const FarCatmarkSubdivisionTables<OsdVertex>*>(_tables);
        const FarBilinearSubdivisionTables<OsdVertex> * bTables =
            dynamic_cast<const FarBilinearSubdivisionTables<OsdVertex>*>(_tables);

        if (tableIndex == Table::F_IT) {
            if (ccTables) return ccTables->Get_F_IT()[level];
            else if (bTables) return bTables->Get_F_IT()[level];
        } else if (tableIndex == Table::F_ITa) {
            if (ccTables) return ccTables->Get_F_ITa()[level];
            else if (bTables) return bTables->Get_F_ITa()[level];
        }
    }
    OsdError(OSD_INTERNAL_CODING_ERROR);
    return 0;
}

OsdVertexDescriptor *
OsdCpuComputeContext::GetVertexDescriptor() const {

    return _vdesc;
}

int
OsdCpuComputeContext::GetNumEditTables() const {

    if (_editTables)
        return _editTables->GetNumBatches();
    return 0;
}

const FarVertexEditTables<OsdVertex>::VertexEditBatch *
OsdCpuComputeContext::GetEditTable(int tableIndex) const {

    if (_editTables)
        return &_editTables->GetBatch(tableIndex);
    return 0;
}

float *
OsdCpuComputeContext::GetCurrentVertexBuffer() const {

    return _currentVertexBuffer;
}

float *
OsdCpuComputeContext::GetCurrentVaryingBuffer() const {

    return _currentVaryingBuffer;
}

OsdCpuComputeContext *
OsdCpuComputeContext::Create(FarMesh<OsdVertex> *farmesh) {

    return new OsdCpuComputeContext(farmesh);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
