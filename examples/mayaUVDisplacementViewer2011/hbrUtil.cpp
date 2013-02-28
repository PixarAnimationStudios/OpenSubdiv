
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
									  std::vector<std::vector<float>> const & alluvs,
                                      int interpBoundary, int scheme,
									  int varCount,int * fvarindices,int* fvarwidth,int totalfarwidth)
{
    static OpenSubdiv::HbrBilinearSubdivision<OpenSubdiv::OsdVertex> _bilinear;
    static OpenSubdiv::HbrLoopSubdivision<OpenSubdiv::OsdVertex> _loop;
    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex> _catmark;

    OpenSubdiv::OsdHbrMesh *hbrMesh;
    if (scheme == 0)
        hbrMesh = new OpenSubdiv::OsdHbrMesh(&_catmark,varCount,fvarindices,fvarwidth,totalfarwidth);
		//hbrMesh = new OpenSubdiv::OsdHbrMesh(&_catmark);
    else if (scheme == 1)
        hbrMesh = new OpenSubdiv::OsdHbrMesh(&_loop);
    else 
        hbrMesh = new OpenSubdiv::OsdHbrMesh(&_bilinear);

	//hbrMesh->SetFVarInterpolateBoundaryMethod(OpenSubdiv::OsdHbrMesh::k_InterpolateBoundaryAlwaysSharp);


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

        if ( valid ) {
            if (scheme == 1) { // loop
                // triangulate
                int triangle[3];
                triangle[0] = vIndex[0];
                for (int j=2; j<numVertex; ++j) {
                    triangle[1] = vIndex[j-1];
                    triangle[2] = vIndex[j];
                    hbrMesh->NewFace(3, triangle, 0);
                }

            } else {
                OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex> *face=
					hbrMesh->NewFace(numVertex, &(vIndex[0]), 0);


 				for(int mm=0;mm<numVertex;mm++)
 				{
 					OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex> *aVert=face->GetVertex(mm);
 					OpenSubdiv::HbrFVarData<OpenSubdiv::OsdVertex>& newData=aVert->NewFVarData(face);
 					newData.SetAllData(hbrMesh->GetTotalFVarWidth(),(float*)&alluvs[i][mm*hbrMesh->GetTotalFVarWidth()]);
 				}

            }
        } else {
            OSD_ERROR("Face %d will be ignored\n", i);
        }

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
        if( edgeCreases2[i] <= 0. )
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

