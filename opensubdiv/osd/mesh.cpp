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
//#include "../osd/dump.cpp"
#include "../osd/kernelDispatcher.h"
#include "../osd/cpuDispatcher.h"

#include "../far/meshFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdMesh::OsdMesh() : _fMesh(NULL), _dispatcher(NULL) { }

OsdMesh::~OsdMesh() {

    if(_dispatcher) 
        delete _dispatcher;
        
    if(_fMesh) 
        delete _fMesh;
}

void
OsdMesh::createTables( FarSubdivisionTables<OsdVertex> const * tables ) {

    _dispatcher->UpdateTable(OsdKernelDispatcher::E_IT,  tables->Get_E_IT());
    _dispatcher->UpdateTable(OsdKernelDispatcher::V_IT,  tables->Get_V_IT());
    _dispatcher->UpdateTable(OsdKernelDispatcher::V_ITa, tables->Get_V_ITa());
    _dispatcher->UpdateTable(OsdKernelDispatcher::E_W,   tables->Get_E_W());
    _dispatcher->UpdateTable(OsdKernelDispatcher::V_W,   tables->Get_V_W());

    if ( const FarCatmarkSubdivisionTables<OsdVertex> * cctable = 
       dynamic_cast<const FarCatmarkSubdivisionTables<OsdVertex>*>(tables) ) {
        // catmark
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_IT, cctable->Get_F_IT());
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_ITa, cctable->Get_F_ITa());
    } else if ( const FarBilinearSubdivisionTables<OsdVertex> * btable = 
       dynamic_cast<const FarBilinearSubdivisionTables<OsdVertex>*>(tables) ) {
        // bilinear
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_IT, btable->Get_F_IT());
        _dispatcher->UpdateTable(OsdKernelDispatcher::F_ITa, btable->Get_F_ITa());
    } else {
        // XXX for glsl shader...
        _dispatcher->CopyTable(OsdKernelDispatcher::F_IT, 0, NULL);
        _dispatcher->CopyTable(OsdKernelDispatcher::F_ITa, 0, NULL);
    }

    CHECK_GL_ERROR("Mesh, update tables\n");
}

bool
OsdMesh::Create(OsdHbrMesh *hbrMesh, int level, int kernel, std::vector<int> * remap) {

    if (_dispatcher)
        delete _dispatcher;
    _dispatcher = OsdKernelDispatcher::CreateKernelDispatcher(level, kernel);

    if (not _dispatcher) {
        OSD_ERROR("Unknown kernel %d\n", kernel);
        return false;
    }

    _level = level;
        
    // create Far mesh
    OSD_DEBUG("Create MeshFactory\n");

    FarMeshFactory<OsdVertex> meshFactory(hbrMesh, _level);

    _fMesh = meshFactory.Create(_dispatcher);
    
    OSD_DEBUG("PREP: NumCoarseVertex = %d\n", _fMesh->GetNumCoarseVertices());
    OSD_DEBUG("PREP: NumVertex = %d\n", _fMesh->GetNumVertices());

    createTables( _fMesh->GetSubdivision() );
    
    // copy the remapping table if the client needs to remap vertex indices from
    // Osd to Hbr for comparison / regression purposes.
    if (remap)
        (*remap)=meshFactory.GetRemappingTable();

    return true;
}

OsdVertexBuffer *
OsdMesh::InitializeVertexBuffer(int numElements)
{
    if (!_dispatcher) return NULL;
    return _dispatcher->InitializeVertexBuffer(numElements, GetTotalVertices());
}

void
OsdMesh::Subdivide(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    _dispatcher->BindVertexBuffer(vertex, varying);

    _dispatcher->OnKernelLaunch();

    _fMesh->Subdivide(_level+1);

    _dispatcher->OnKernelFinish();

    _dispatcher->UnbindVertexBuffer();
}

void
OsdMesh::Synchronize() {

    _dispatcher->Synchronize();
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

