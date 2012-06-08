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
#ifndef HBRSUBDIVISION_H
#define HBRSUBDIVISION_H

template <class T> class HbrFace;
template <class T> class HbrVertex;
template <class T> class HbrHalfedge;
template <class T> class HbrMesh;
template <class T> class HbrSubdivision {
public:
    HbrSubdivision<T>()
	: creaseSubdivision(k_CreaseNormal) {}

    virtual ~HbrSubdivision<T>() {}

    virtual HbrSubdivision<T>* Clone() const = 0;
    
    // How to subdivide a face
    virtual void Refine(HbrMesh<T>* mesh, HbrFace<T>* face) = 0;

    // Subdivide a face only at a particular vertex (creating one child)
    virtual HbrFace<T>* RefineFaceAtVertex(HbrMesh<T>* mesh, HbrFace<T>* face, HbrVertex<T>* vertex) = 0;

    // Refine all faces around a particular vertex
    virtual void RefineAtVertex(HbrMesh<T>* mesh, HbrVertex<T>* vertex);
    
    // Given an edge, try to ensure the edge's opposite exists by
    // forcing refinement up the hierarchy
    virtual void GuaranteeNeighbor(HbrMesh<T>* mesh, HbrHalfedge<T>* edge) = 0;

    // Given an vertex, ensure all faces in the ring around it exist
    // by forcing refinement up the hierarchy
    virtual void GuaranteeNeighbors(HbrMesh<T>* mesh, HbrVertex<T>* vertex) = 0;

    // Returns true if the vertex, edge, or face has a limit point,
    // curve, or surface associated with it
    virtual bool HasLimit(HbrMesh<T>* /* mesh */, HbrFace<T>* /* face */) { return true; }
    virtual bool HasLimit(HbrMesh<T>* /* mesh */, HbrHalfedge<T>* /* edge */) { return true; }
    virtual bool HasLimit(HbrMesh<T>* /* mesh */, HbrVertex<T>* /* vertex */) { return true; }
    
    // How to turn faces, edges, and vertices into vertices
    virtual HbrVertex<T>* Subdivide(HbrMesh<T>* mesh, HbrFace<T>* face) = 0;
    virtual HbrVertex<T>* Subdivide(HbrMesh<T>* mesh, HbrHalfedge<T>* edge) = 0;
    virtual HbrVertex<T>* Subdivide(HbrMesh<T>* mesh, HbrVertex<T>* vertex) = 0;

    // Returns true if the vertex is extraordinary in the subdivision scheme
    virtual bool VertexIsExtraordinary(HbrMesh<T>* /* mesh */, HbrVertex<T>* /* vertex */) { return false; }

    // Crease subdivision rules. When subdividing a edge with a crease
    // strength, we get two child subedges, and we need to determine
    // what weights to assign these subedges. The "normal" rule
    // is to simply assign the current edge's crease strength - 1
    // to both of the child subedges. The "Chaikin" rule looks at the
    // current edge and incident edges to the current edge's end
    // vertices, and weighs them; for more information consult
    // the Geri's Game paper.
    enum CreaseSubdivision {
	k_CreaseNormal,
	k_CreaseChaikin
    };
    CreaseSubdivision GetCreaseSubdivisionMethod() const { return creaseSubdivision; }
    void SetCreaseSubdivisionMethod(CreaseSubdivision method) { creaseSubdivision = method; }

    // Figures out how to assign a crease weight on an edge to its
    // subedge. The subedge must be a child of the parent edge
    // (either subedge->GetOrgVertex() or subedge->GetDestVertex()
    // == edge->Subdivide()). The vertex supplied must NOT be
    // a parent of the subedge; it is either the origin or
    // destination vertex of edge.
    void SubdivideCreaseWeight(HbrHalfedge<T>* edge, HbrVertex<T>* vertex, HbrHalfedge<T>* subedge);

    // Returns the expected number of children faces after subdivision
    // for a face with the given number of vertices.
    virtual int GetFaceChildrenCount(int nvertices) const = 0;
    
protected:
    CreaseSubdivision creaseSubdivision;

    // Helper routine for subclasses: for a given vertex, sums
    // contributions from surrounding vertices
    void AddSurroundingVerticesWithWeight(HbrMesh<T>* mesh, HbrVertex<T>* vertex, float weight, T* data);

    // Helper routine for subclasses: for a given vertex with a crease
    // mask, adds contributions from the two crease edges
    void AddCreaseEdgesWithWeight(HbrMesh<T>* mesh, HbrVertex<T>* vertex, bool next, float weight, T* data);
    
private:
    // Helper class used by AddSurroundingVerticesWithWeight
    class SmoothSubdivisionVertexOperator : public HbrVertexOperator<T> {
    public:
        SmoothSubdivisionVertexOperator(T* data, bool meshHasEdits, float weight)
            : m_data(data),
              m_meshHasEdits(meshHasEdits),
              m_weight(weight)
        {
        }
        virtual void operator() (HbrVertex<T> &vertex) {
            // Must ensure vertex edits have been applied        
            if (m_meshHasEdits) {
                vertex.GuaranteeNeighbors();
            }
            m_data->AddWithWeight(vertex.GetData(), m_weight);
        }
    private:
        T* m_data;
        const bool m_meshHasEdits;
        const float m_weight;
    };

    // Helper class used by AddCreaseEdgesWithWeight
    class CreaseSubdivisionHalfedgeOperator : public HbrHalfedgeOperator<T> {
    public:
        CreaseSubdivisionHalfedgeOperator(HbrVertex<T> *vertex, T* data, bool meshHasEdits, bool next, float weight)
            : m_vertex(vertex),
              m_data(data),
              m_meshHasEdits(meshHasEdits),
              m_next(next),
              m_weight(weight),
              m_count(0)
        {
        }
        virtual void operator() (HbrHalfedge<T> &edge) {
            if (m_count < 2 && edge.IsSharp(m_next)) {
                HbrVertex<T>* a = edge.GetDestVertex();
                if (a == m_vertex) a = edge.GetOrgVertex();
                // Must ensure vertex edits have been applied
                if (m_meshHasEdits) {
                    a->GuaranteeNeighbors();
                }			
                m_data->AddWithWeight(a->GetData(), m_weight);
                m_count++;
            }
        }
    private:
        HbrVertex<T>* m_vertex;
        T* m_data;
        const bool m_meshHasEdits;
        const bool m_next;
        const float m_weight;
        int m_count;
    };

private:
    // Helper class used by RefineAtVertex.
    class RefineFaceAtVertexOperator : public HbrFaceOperator<T> {
    public:
        RefineFaceAtVertexOperator(HbrSubdivision<T>* subdivision, HbrMesh<T>* mesh, HbrVertex<T> *vertex)
            : m_subdivision(subdivision),
              m_mesh(mesh),
              m_vertex(vertex)
        {
        }
        virtual void operator() (HbrFace<T> &face) {
            m_subdivision->RefineFaceAtVertex(m_mesh, &face, m_vertex);
        }
    private:
        HbrSubdivision<T>* const m_subdivision;
        HbrMesh<T>* const m_mesh;
        HbrVertex<T>* const m_vertex;
    };
    
};

template <class T>
void
HbrSubdivision<T>::RefineAtVertex(HbrMesh<T>* mesh, HbrVertex<T>* vertex) {
    GuaranteeNeighbors(mesh, vertex);
    RefineFaceAtVertexOperator op(this, mesh, vertex);
    vertex->ApplyOperatorSurroundingFaces(op);
}

template <class T>
void
HbrSubdivision<T>::SubdivideCreaseWeight(HbrHalfedge<T>* edge, HbrVertex<T>* vertex, HbrHalfedge<T>* subedge) {

    float sharpness = edge->GetSharpness();

    // In all methods, if the parent edge is infinitely sharp, the
    // child edge is also infinitely sharp
    if (sharpness >= HbrHalfedge<T>::k_InfinitelySharp) {
	subedge->SetSharpness(HbrHalfedge<T>::k_InfinitelySharp);
    }

    // Chaikin's curve subdivision: use 3/4 of the parent sharpness,
    // plus 1/4 of crease sharpnesses incident to vertex
    else if (creaseSubdivision == HbrSubdivision<T>::k_CreaseChaikin) {

	float childsharp = 0.0f;

	// Add 1/4 of the sharpness of all crease edges incident to
	// the vertex (other than this crease edge)
	std::list<HbrHalfedge<T>*> edges;
	vertex->GuaranteeNeighbors();
	vertex->GetSurroundingEdges(edges);

	int n = 0;
	for (typename std::list<HbrHalfedge<T>*>::iterator ei = edges.begin(); ei != edges.end(); ++ei) {
	    if (*ei == edge) continue;
	    if ((*ei)->GetSharpness() > HbrHalfedge<T>::k_Smooth) {
		childsharp += (*ei)->GetSharpness();
		n++;
	    }
	}

	if (n) {
	    childsharp = childsharp * 0.25f / n;
	}
		    
	// Add 3/4 of the sharpness of this crease edge
	childsharp += sharpness * 0.75f;
	childsharp -= 1.0f;
	if (childsharp < (float) HbrHalfedge<T>::k_Smooth) {
	    childsharp = (float) HbrHalfedge<T>::k_Smooth;
	}
	subedge->SetSharpness(childsharp);
	
    } else {
	sharpness -= 1.0f;
	if (sharpness < (float) HbrHalfedge<T>::k_Smooth) {
	    sharpness = (float) HbrHalfedge<T>::k_Smooth;
	}
	subedge->SetSharpness(sharpness);
    }
}

template <class T>
void
HbrSubdivision<T>::AddSurroundingVerticesWithWeight(HbrMesh<T>* mesh, HbrVertex<T>* vertex, float weight, T* data) {
    SmoothSubdivisionVertexOperator op(data, mesh->HasVertexEdits(), weight);
    vertex->ApplyOperatorSurroundingVertices(op);
}

template <class T>
void
HbrSubdivision<T>::AddCreaseEdgesWithWeight(HbrMesh<T>* mesh, HbrVertex<T>* vertex, bool next, float weight, T* data) {
    CreaseSubdivisionHalfedgeOperator op(vertex, data, mesh->HasVertexEdits(), next, weight);
    vertex->ApplyOperatorSurroundingEdges(op);
}
#endif /* HBRSUBDIVISION_H */
