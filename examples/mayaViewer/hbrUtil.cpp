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
#include "hbrUtil.h"
#include <far/mesh.h>
#include <hbr/mesh.h>
#include <hbr/bilinear.h>
#include <hbr/loop.h>
#include <hbr/catmark.h>

#define OSD_ERROR printf  // XXXX

OpenSubdiv::OsdHbrMesh * ConvertToHBR(int nVertices,
                                      std::vector<int>   const & numIndices,
                                      std::vector<int>   const & faceIndices,
                                      std::vector<int>   const & vtxCreaseIndices,
                                      std::vector<float> const & vtxCreases,
                                      std::vector<int>   const & edgeCrease1Indices, // face index, local edge index
                                      std::vector<float> const & edgeCreases1,
                                      std::vector<int>   const & edgeCrease2Indices, // 2 vertex indices (Maya friendly)
                                      std::vector<float> const & edgeCreases2,
                                      int interpBoundary, bool loop)
{
    static OpenSubdiv::HbrBilinearSubdivision<OpenSubdiv::OsdVertex> _bilinear;
    static OpenSubdiv::HbrLoopSubdivision<OpenSubdiv::OsdVertex> _loop;
    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex> _catmark;

    OpenSubdiv::OsdHbrMesh *hbrMesh;
    if (loop)
        hbrMesh = new OpenSubdiv::OsdHbrMesh(&_loop);
    else
        hbrMesh = new OpenSubdiv::OsdHbrMesh(&_catmark);

    OpenSubdiv::OsdVertex v;
    for(int i = 0; i < nVertices; ++i){
        // create empty vertex : actual vertices will be initialized in UpdatePoints();
        hbrMesh->NewVertex(i, v);
    }

    // get face indices
    std::vector<int> vIndex;
    int nFaces = (int)numIndices.size(), offset = 0;
    for (int i = 0; i < nFaces; ++i) {
        int numVertex = numIndices[i];
        vIndex.resize(numVertex);

        bool valid=true;
        for (int j=0; j<numVertex; ++j) {
            vIndex[j] = faceIndices[j + offset];
            int vNextIndex = faceIndices[(j+1)%numVertex + offset];

            // check for non-manifold face
            OpenSubdiv::OsdHbrVertex * origin = hbrMesh->GetVertex( vIndex[j] );
            OpenSubdiv::OsdHbrVertex * destination = hbrMesh->GetVertex( vNextIndex );
            if (!origin || !destination) {
                OSD_ERROR("ERROR : An edge was specified that connected a nonexistent vertex");
                valid=false;
            }

            if (origin == destination) {
                OSD_ERROR("ERROR : An edge was specified that connected a vertex to itself");
                valid=false;
            }

            OpenSubdiv::OsdHbrHalfedge * opposite = destination->GetEdge(origin);
            if (opposite && opposite->GetOpposite()) {
                OSD_ERROR("ERROR : A non-manifold edge incident to more than 2 faces was found");
                valid=false;
            }

            if (origin->GetEdge(destination)) {
                OSD_ERROR("ERROR : An edge connecting two vertices was specified more than once. "
                          "It's likely that an incident face was flipped");
                valid=false;
            }
        }

        if ( valid )
            hbrMesh->NewFace(numVertex, &(vIndex[0]), 0);
        else
            OSD_ERROR("Face %d will be ignored\n", i);

        offset += numVertex;
    }

    // XXX: use hbr enum or redefine same enum in gsd
    hbrMesh->SetInterpolateBoundaryMethod((OpenSubdiv::OsdHbrMesh::InterpolateBoundaryMethod)interpBoundary);

    // set edge crease in two different indexing way
    int nEdgeCreases = (int)edgeCreases1.size();
    for (int i = 0; i < nEdgeCreases; ++i) {
        if( edgeCreases1[i] <= 0. )
            continue;

        OpenSubdiv::OsdHbrHalfedge * e = hbrMesh->GetFace(edgeCrease1Indices[i*2])->GetEdge(edgeCrease1Indices[i*2+1]);
        if (!e) {
            OSD_ERROR("Can't find edge (face %d edge %d)\n", edgeCrease1Indices[i*2], edgeCrease1Indices[i*2+1]);
            continue;
        }
        e->SetSharpness( (float)edgeCreases1[i] );
    }
    nEdgeCreases = (int)edgeCreases2.size();
    for (int i = 0; i < nEdgeCreases; ++i) {
        if( edgeCreases1[i] <= 0. )
            continue;

        OpenSubdiv::OsdHbrVertex * v0 = hbrMesh->GetVertex(edgeCrease2Indices[i*2]);
        OpenSubdiv::OsdHbrVertex * v1 = hbrMesh->GetVertex(edgeCrease2Indices[i*2+1]);
        OpenSubdiv::OsdHbrHalfedge * e = NULL;

        if ( v0 && v1 )
            if ( ! (e = v0->GetEdge(v1)) )
                e = v1->GetEdge(v0);
        if (!e) {
            OSD_ERROR("ERROR can't find edge");
            continue;
        }
        e->SetSharpness( (float)edgeCreases2[i] );
    }

    // set corner
    {
        int nVertexCreases = (int)vtxCreases.size();
        for ( int i = 0; i< nVertexCreases; ++i ) {
            if( vtxCreases[i] <= 0. )
                continue;
            OpenSubdiv::OsdHbrVertex * v = hbrMesh->GetVertex(vtxCreaseIndices[i]);
            if (!v) {
                OSD_ERROR("Can't find vertex %d\n", vtxCreaseIndices[i]);
                continue;
            }
            v->SetSharpness( (float)vtxCreases[i] );
        }
    }

    hbrMesh->Finish();
    return hbrMesh;
}

