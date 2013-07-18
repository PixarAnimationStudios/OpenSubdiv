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

#include "hbrUtil.h"

#include <hbr/mesh.h>
#include <hbr/bilinear.h>
#include <hbr/loop.h>
#include <hbr/catmark.h>

#include <vector>

#define OSD_ERROR printf  // XXXX

OsdHbrMesh * 
ConvertToHBR( int nVertices,
              std::vector<int>   const & faceVertCounts,
              std::vector<int>   const & faceIndices,
              std::vector<int>   const & vtxCreaseIndices,
              std::vector<double> const & vtxCreases,
              std::vector<int>   const & edgeCrease1Indices,  // face index, local edge index
              std::vector<float> const & edgeCreases1,
              std::vector<int>   const & edgeCrease2Indices,  // 2 vertex indices (Maya friendly)
              std::vector<double> const & edgeCreases2,
              OsdHbrMesh::InterpolateBoundaryMethod interpBoundary,
              HbrMeshUtil::SchemeType scheme,
              bool usingPtex,
              FVarDataDesc const * fvarDesc,
              std::vector<float> const * fvarData
            )
{

    static OpenSubdiv::HbrBilinearSubdivision<OpenSubdiv::OsdVertex> _bilinear;
    static OpenSubdiv::HbrLoopSubdivision<OpenSubdiv::OsdVertex> _loop;
    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex> _catmark;

    // Build HBR mesh with/without face varying data, according to input data.
    // If a face-varying descriptor is passed in its memory needs to stay 
    // alive as long as this hbrMesh is alive (for indices and widths arrays). 
    OsdHbrMesh *hbrMesh;
    if ( fvarDesc )
    {
        if (scheme == HbrMeshUtil::kCatmark)
            hbrMesh = new OsdHbrMesh(&_catmark,  fvarDesc->getCount(), 
                                                 fvarDesc->getIndices(), 
                                                 fvarDesc->getWidths(), 
                                                 fvarDesc->getTotalWidth());
        else if (scheme == HbrMeshUtil::kLoop)
            hbrMesh = new OsdHbrMesh(&_loop,     fvarDesc->getCount(), 
                                                 fvarDesc->getIndices(), 
                                                 fvarDesc->getWidths(), 
                                                 fvarDesc->getTotalWidth());
        else 
            hbrMesh = new OsdHbrMesh(&_bilinear, fvarDesc->getCount(),
                                                 fvarDesc->getIndices(), 
                                                 fvarDesc->getWidths(), 
                                                 fvarDesc->getTotalWidth());
    }
    else
    {
        if (scheme == HbrMeshUtil::kCatmark)
            hbrMesh = new OsdHbrMesh(&_catmark);
        else if (scheme == HbrMeshUtil::kLoop)
            hbrMesh = new OsdHbrMesh(&_loop);
        else
            hbrMesh = new OsdHbrMesh(&_bilinear);
    }


    // create empty verts: actual vertices initialized in UpdatePoints();
    OpenSubdiv::OsdVertex v;
    for (int i = 0; i < nVertices; ++i) {
        hbrMesh->NewVertex(i, v);
    }

    std::vector<int> vIndex;
    int nFaces = (int)faceVertCounts.size();
    int fvcOffset = 0;          // face-vertex count offset
    int ptxIdx = 0;

    for (int fi = 0; fi < nFaces; ++fi) 
    {
        int nFaceVerts = faceVertCounts[fi];
        vIndex.resize(nFaceVerts);

        bool valid = true;
        for (int fvi = 0; fvi < nFaceVerts; ++fvi) 
        {
            vIndex[fvi] = faceIndices[fvi + fvcOffset];
            int vNextIndex = faceIndices[(fvi+1) % nFaceVerts + fvcOffset];

            // check for non-manifold face
            OsdHbrVertex * origin = hbrMesh->GetVertex(vIndex[fvi]);
            OsdHbrVertex * destination = hbrMesh->GetVertex(vNextIndex);
            if (!origin || !destination) {
                OSD_ERROR("ERROR : An edge was specified that connected a nonexistent vertex");
                valid = false;
            }

            if (origin == destination) {
                OSD_ERROR("ERROR : An edge was specified that connected a vertex to itself");
                valid = false;
            }

            OsdHbrHalfedge * opposite = destination->GetEdge(origin);
            if (opposite && opposite->GetOpposite()) {
                OSD_ERROR("ERROR : A non-manifold edge incident to more than 2 faces was found");
                valid = false;
            }

            if (origin->GetEdge(destination)) {
                OSD_ERROR("ERROR : An edge connecting two vertices was specified more than once. "
                          "It's likely that an incident face was flipped");
                valid = false;
            }
        }

        if ( valid ) 
        {
            if (scheme == HbrMeshUtil::kLoop) {
                // For Loop subdivision, triangulate from vertex indices
                int triangle[3];
                triangle[0] = vIndex[0];
                for (int fvi = 2; fvi < nFaceVerts; ++fvi) {
                    triangle[1] = vIndex[fvi-1];
                    triangle[2] = vIndex[fvi];
                    hbrMesh->NewFace(3, triangle, 0);
                }
                /* ptex not fully implemented for loop, yet */
                /* fvar not fully implemented for loop, yet */

            } else {

                // For Catmull-Clark subdivision, create a quad face from vertices
                /* bilinear subdivision not fully implemented */
                OsdHbrFace *face = hbrMesh->NewFace(nFaceVerts, &(vIndex[0]), 0);

                if (usingPtex) {
                    // ptex textures will be used, set up ptex coordinates
                    face->SetPtexIndex(ptxIdx);
                    ptxIdx += (nFaceVerts == 4) ? 1 : nFaceVerts;
                }

                if (fvarData) {
                    // Face-varying data has been passed in, get pointer to data
                    int fvarWidth = hbrMesh->GetTotalFVarWidth();
                    const float *faceData = &(*fvarData)[ fvcOffset*fvarWidth ];

                    // For each face vertex copy fvar data into hbr mesh
                    for(int fvi=0; fvi<nFaceVerts; ++fvi)
                    {
                        OsdHbrVertex *v = hbrMesh->GetVertex( vIndex[fvi] );
                        OsdHbrFVarData& fvarData = v->GetFVarData(face);
                        if ( ! fvarData.IsInitialized() )
                        {
                            fvarData.SetAllData( fvarWidth, faceData );
                        }
                        else if (!fvarData.CompareAll(fvarWidth, faceData))
                        {
                            // If data exists for this face vertex, but is different
                            // (e.g. we're on a UV seam) create another fvar datum
                            OsdHbrFVarData& fvarData = v->NewFVarData(face);
                            fvarData.SetAllData( fvarWidth, faceData );
                        }

                        // Advance pointer to next set of face-varying data
                        faceData += fvarWidth;
                    }
                }
            }
        } else {
            OSD_ERROR("Face %d will be ignored\n", fi);
        }

        fvcOffset += nFaceVerts;
    }

    // Assign boundary interpolation methods
    hbrMesh->SetInterpolateBoundaryMethod(interpBoundary);
    if ( fvarDesc ) 
        hbrMesh->SetFVarInterpolateBoundaryMethod(fvarDesc->getInterpBoundary());

    // Set edge crease in two different indexing way
    size_t nEdgeCreases = edgeCreases1.size();
    for (size_t i = 0; i < nEdgeCreases; ++i) {
        if (edgeCreases1[i] <= 0.0)
            continue;

        OsdHbrHalfedge * e = hbrMesh->
            GetFace(edgeCrease1Indices[i*2])->
            GetEdge(edgeCrease1Indices[i*2+1]);

        if (!e) {
            OSD_ERROR("Can't find edge (face %d edge %d)\n",
                      edgeCrease1Indices[i*2], edgeCrease1Indices[i*2+1]);
            continue;
        }
        e->SetSharpness(static_cast<float>(edgeCreases1[i]));
    }
    nEdgeCreases = edgeCreases2.size();
    for (size_t i = 0; i < nEdgeCreases; ++i) {
        if (edgeCreases2[i] <= 0.0)
            continue;

        OsdHbrVertex * v0 = hbrMesh->GetVertex(edgeCrease2Indices[i*2]);
        OsdHbrVertex * v1 = hbrMesh->GetVertex(edgeCrease2Indices[i*2+1]);
        OsdHbrHalfedge * e = NULL;

        if (v0 && v1)
            if (!(e = v0->GetEdge(v1)))
                e = v1->GetEdge(v0);
        if (!e) {
            OSD_ERROR("ERROR can't find edge");
            continue;
        }
        e->SetSharpness(static_cast<float>(edgeCreases2[i]));
    }

    // Set corner
    {
        size_t nVertexCreases = vtxCreases.size();
        for (size_t i = 0; i < nVertexCreases; ++i) {
            if (vtxCreases[i] <= 0.0)
                continue;
            OsdHbrVertex * v = hbrMesh->GetVertex(vtxCreaseIndices[i]);
            if (!v) {
                OSD_ERROR("Can't find vertex %d\n", vtxCreaseIndices[i]);
                continue;
            }
            v->SetSharpness(static_cast<float>(vtxCreases[i]));
        }
    }

    // Call finish to complete build of HBR mesh
    hbrMesh->Finish();

    return hbrMesh;
}

