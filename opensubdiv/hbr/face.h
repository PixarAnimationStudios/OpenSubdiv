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
#ifndef HBRFACE_H
#define HBRFACE_H

#include <assert.h>
#include <cstdio>
#include <functional>
#include <iostream>
#include <algorithm>
#include <vector>

#include "../hbr/fvarData.h"
#include "../hbr/allocator.h"
#ifdef HBRSTITCH
#include "libgprims/stitch.h"
#include "libgprims/stitchInternal.h"
#endif

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrVertex;
template <class T> class HbrHalfedge;
template <class T> class HbrFace;
template <class T> class HbrMesh;
template <class T> class HbrHierarchicalEdit;

template <class T> std::ostream& operator<<(std::ostream& out, const HbrFace<T>& face);

// A descriptor for a path to a face
struct HbrFacePath {
    void Print() const {
        printf("%d", topface);
        for (std::vector<int>::const_reverse_iterator i = remainder.rbegin(); i != remainder.rend(); ++i) {
            printf(" %d", *i);
        }
        printf("\n");
    }

    int topface;
    // Note that the elements in remainder are stored in reverse order.
    std::vector<int> remainder;
    friend bool operator< (const HbrFacePath& x, const HbrFacePath& y);
};

inline bool operator< (const HbrFacePath& x, const HbrFacePath& y) {
    if (x.topface != y.topface) {
        return x.topface < y.topface;
    } else if (x.remainder.size() != y.remainder.size()) {
        return x.remainder.size() < y.remainder.size();
    } else {
        std::vector<int>::const_reverse_iterator i = x.remainder.rbegin();
        std::vector<int>::const_reverse_iterator j = y.remainder.rbegin();
        for ( ; i != x.remainder.rend(); ++i, ++j) {
            if (*i != *j) return (*i < *j);
        }
        return true;
    }
}

// A simple wrapper around an array of four children. Used to block
// allocate pointers to children of HbrFace in the common case
template <class T>
class HbrFaceChildren {
public:
    HbrFace<T> *& operator[](const int index) {
        return children[index];
    }

    const HbrFace<T> *& operator[](const int index) const {
        return children[index];
    }
    

private:
    friend class HbrAllocator<HbrFaceChildren<T> >;

    // Used by block allocator
    HbrFaceChildren<T>*& GetNext() { return (HbrFaceChildren<T>*&) children; }
    
    HbrFaceChildren() {}

    ~HbrFaceChildren() {}

    HbrFace<T> *children[4];
};

template <class T> class HbrFace {

private:
    friend class HbrAllocator<HbrFace<T> >;
    friend class HbrHalfedge<T>;
    HbrFace();
    ~HbrFace();

public:

    void Initialize(HbrMesh<T>* mesh, HbrFace<T>* parent, int childindex, int id, int uindex, int nvertices, HbrVertex<T>** vertices, int fvarwidth = 0, int depth = 0);
    void Destroy();

    // Returns the mesh to which this face belongs
    HbrMesh<T>* GetMesh() const { return mesh; }

    // Return number of vertices
    int GetNumVertices() const { return nvertices; }

    // Return face ID
    int GetID() const { return id; }

    // Return the first halfedge of the face
    HbrHalfedge<T>* GetFirstEdge() const {
        if (nvertices > 4) {
            return const_cast<HbrHalfedge<T>*>(&extraedges[0]);
        } else {
            return const_cast<HbrHalfedge<T>*>(&edges[0]);
        }
    }

    // Return the halfedge which originates at the vertex with the
    // indicated origin index
    HbrHalfedge<T>* GetEdge(int index) const;

    // Return the vertex with the indicated index
    HbrVertex<T>* GetVertex(int index) const;

    // Return the parent of this face
    HbrFace<T>* GetParent() const { return parent; }

    // Set the child
    void SetChild(int index, HbrFace<T>* face);

    // Return the child with the indicated index
    HbrFace<T>* GetChild(int index) const {
        int nchildren = mesh->GetSubdivision()->GetFaceChildrenCount(nvertices);
	if (!children.children || index < 0 || index >= nchildren) return 0;
        if (nchildren > 4) {
            return children.extrachildren[index];
        } else {
            return (*children.children)[index];
        }
    }

    // Subdivide the face into a vertex if needed and return
    HbrVertex<T>* Subdivide();

    // Remove the reference to subdivided vertex
    void RemoveChild() { vchild = 0; }

    // "Hole" flags used by subdivision to drop faces
    bool IsHole() const { return hole; }
    void SetHole() { hole = 1; }

    // Coarse faces are the top level faces of a mesh. This will be
    // set by mesh->Finish()
    bool IsCoarse() const { return coarse; }
    void SetCoarse() { coarse = 1; }

    // Protected faces cannot be garbage collected; this may be set on
    // coarse level faces if the mesh is shared
    bool IsProtected() const { return protect; }
    void SetProtected() { protect = 1; }
    void ClearProtected() { protect = 0; }

    // Simple bookkeeping needed for garbage collection by HbrMesh
    bool IsCollected() const { return collected; }
    void SetCollected() { collected = 1; }
    void ClearCollected() { collected = 0; }

    // Refine the face
    void Refine();

    // Unrefine the face
    void Unrefine();

    // Returns true if the face has a limit surface
    bool HasLimit();

    // Returns memory statistics
    unsigned long GetMemStats() const;

    // Return facevarying data from the appropriate vertex index
    // registered to this face. Note that this may either be "generic"
    // facevarying item (data.GetFace() == 0) or one specifically
    // registered to the face (data.GetFace() == this) - this is
    // important when trying to figure out whether the vertex has
    // created some storage for the item designed to store
    // discontinuous values for this face.
    HbrFVarData<T>& GetFVarData(int index) {
        return GetVertex(index)->GetFVarData(this);
    }

    // Mark this face as being used, which in turn increments the
    // usage counter of all vertices in the support for the face. A
    // used face can not be garbage collected
    void MarkUsage();

    // Clears the usage of this face, which in turn decrements the
    // usage counter of all vertices in the support for the face and
    // marks the face as a candidate for garbage collection
    void ClearUsage();

    // A face can be cleaned if all of its vertices are not being
    // used; has no children; and (for top level faces) deletion of
    // its edges will not leave singular vertices
    bool GarbageCollectable() const;

    // Connect this face to a list of hierarchical edits
    void SetHierarchicalEdits(HbrHierarchicalEdit<T>** edits);

    // Return the list of hierarchical edits associated with this face
    HbrHierarchicalEdit<T>** GetHierarchicalEdits() const { return edits; }

    // Whether the face has certain types of edits (not necessarily
    // local - could apply to a subface)
    bool HasVertexEdits() const { return hasVertexEdits; }
    void MarkVertexEdits() { hasVertexEdits = 1; }

    // Return the depth of the face
    int GetDepth() const { return static_cast<int>(depth); }

    // Return the uniform index of the face. This is different
    // from the ID because it may be shared with other faces
    int GetUniformIndex() const { return uindex; }

    // Set the uniform index of the face
    void SetUniformIndex(int i) { uindex = i; }

    // Return the ptex index
    int GetPtexIndex() const { return ptexindex; }

    // Set the ptex index of the face
    void SetPtexIndex(int i) { ptexindex = i; }

    // Used by block allocator
    HbrFace<T>*& GetNext() { return parent; }

    HbrFacePath GetPath() const {
        HbrFacePath path;
        path.remainder.reserve(GetDepth());
        const HbrFace<T>* f = this, *p = GetParent();
        while (p) {
            int nchildren = mesh->GetSubdivision()->GetFaceChildrenCount(p->nvertices);
            if (nchildren > 4) {
                for (int i = 0; i < nchildren; ++i) {
                    if (p->children.extrachildren[i] == f) {
                        path.remainder.push_back(i);
                        break;
                    }
                }
            } else {
                for (int i = 0; i < nchildren; ++i) {
                    if ((*p->children.children)[i] == f) {
                        path.remainder.push_back(i);
                        break;
                    }
                }
            }
            f = p;
            p = f->GetParent();
        }
        path.topface = f->GetID();
        assert(GetDepth() == 0 || static_cast<int>(path.remainder.size()) == GetDepth());
        return path;
    }

    void PrintPath() const {
        GetPath().Print();
    }

    // Returns the blind pointer to client data
    void *GetClientData() const {
        return clientData;
    }

    // Sets the blind pointer to client data
    void SetClientData(void *data) {
        clientData = data;
    }


private:

    // Mesh to which this face belongs
    HbrMesh<T>* mesh;

    // Unique id for this face
    int id;

    // Uniform index
    int uindex;

    // Ptex index
    int ptexindex;

    // Number of vertices (and number of edges)
    int nvertices;

    // Halfedge array for this face
    // HbrHalfedge::getIndex() relies on this being size 4
    HbrHalfedge<T> edges[4];

    // Edge storage if this face is not a triangle or quad
    HbrHalfedge<T>* extraedges;

    // Pointer to parent face
    HbrFace<T>* parent;

    // Pointer to children array. If there are four children or less,
    // we use the HbrFaceChildren pointer, otherwise we use
    // extrachildren
    union {
        HbrFaceChildren<T>* children;
        HbrFace<T>** extrachildren;
    } children;

    // Subdivided vertex child
    HbrVertex<T>* vchild;

    // Bits used by halfedges to track facevarying sharpnesses
    unsigned int *fvarbits;

#ifdef HBRSTITCH
    // Pointers to stitch edges and data used by the half edges.
    StitchEdge **stitchEdges;
    void **stitchDatas;
#endif

    // Pointer to a list of hierarchical edits applicable to this face
    HbrHierarchicalEdit<T>** edits;

    // Blind client data pointer
    void * clientData;

    // Depth of the face in the mesh hierarchy - coarse faces are
    // level 0. (Hmmm.. is it safe to assume that we'll never
    // subdivide to greater than 255?)
    unsigned char depth;

    unsigned short hole:1;
    unsigned short coarse:1;
    unsigned short protect:1;
    unsigned short collected:1;
    unsigned short hasVertexEdits:1;
    unsigned short initialized:1;
    unsigned short destroyed:1;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#include "../hbr/mesh.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T>
HbrFace<T>::HbrFace()
    : mesh(0), id(-1), uindex(-1), ptexindex(-1), nvertices(0), extraedges(0), parent(0), vchild(0), fvarbits(0),
#ifdef HBRSTITCH
      stitchEdges(0),
      stitchDatas(0),
#endif
      edits(0), clientData(0), depth(0), hole(0), coarse(0), protect(0), collected(0), hasVertexEdits(0), initialized(0), destroyed(0) {
    children.children = 0;
}

template <class T>
void
HbrFace<T>::Initialize(HbrMesh<T>* m, HbrFace<T>* _parent, int childindex, int fid, int _uindex, int nv, HbrVertex<T>** vertices, int /* fvarwidth */, int _depth) {
    mesh = m;
    id = fid;
    uindex = _uindex;
    ptexindex = -1;
    nvertices = nv;
    extraedges = 0;
    children.children = 0;
    vchild = 0;
    fvarbits = 0;
#ifdef HBRSTITCH
    stitchEdges = 0;
    stitchDatas = 0;
#endif
    clientData = 0;
    edits = 0;
    depth = _depth;
    hole = 0;
    coarse = 0;
    protect = 0;
    collected = 0;
    hasVertexEdits = 0;
    initialized = 1;
    destroyed = 0;

    int i;
    const int fvarcount = mesh->GetFVarCount();
    int fvarbitsSizePerEdge = ((fvarcount + 15) / 16);

    if (nv > 4) {

        // If we have more than four vertices, we ignore the
        // overallocation and allocate our own buffers for stitch
        // edges and facevarying data.
#ifdef HBRSTITCH
        if (mesh->GetStitchCount()) {
            stitchEdges = new StitchEdge*[mesh->GetStitchCount() * nv];
            stitchDatas = new void*[nv];
            for (i = 0; i < mesh->GetStitchCount() * nv; ++i) {
                stitchEdges[i] = 0;
            }
            for (i = 0; i < nv; ++i) {
                stitchDatas[i] = 0;
            }
        }
#endif
        if (fvarcount) {
            // We allocate fvarbits in one chunk.
            // fvarbits needs capacity for two bits per fvardatum per edge,
            // minimum size one integer per edge
            const size_t fvarbitsSize = nv * (fvarbitsSizePerEdge * sizeof(unsigned int));
            char *buffer = (char*) malloc(fvarbitsSize);
            fvarbits = (unsigned int*) buffer;
        }

        // We also ignore the edge array and allocate extra storage -
        // this simplifies GetNext and GetPrev math in HbrHalfedge
        extraedges = new HbrHalfedge<T>[nv];

    } else {
        // Under four vertices: upstream allocation for the class has
        // been over allocated to include storage for stitchEdges
        // and fvarbits. Just point our pointers at it.
        char *buffer = ((char *) this + sizeof(*this));
#ifdef HBRSTITCH
        if (mesh->GetStitchCount()) {
            stitchEdges = (StitchEdge**) buffer;
            buffer += 4 * mesh->GetStitchCount() * sizeof(StitchEdge*);
            stitchDatas = (void**) buffer;
            for (i = 0; i < mesh->GetStitchCount() * 4; ++i) {
                stitchEdges[i] = 0;
            }
            for (i = 0; i < 4; ++i) {
                stitchDatas[i] = 0;
            }
            buffer += 4 * sizeof(void*);
        }
#endif
        if (fvarcount) {
            fvarbits = (unsigned int*) buffer;
        }
    }

    // Must do this before we create edges
    if (_parent) {
        _parent->SetChild(childindex, this);
    }

    // Edges must be constructed in this two part approach: we must
    // ensure that opposite/next/previous ptrs are all set up
    // correctly, before we can begin adding incident edges to
    // vertices.
    int next;
    unsigned int *curfvarbits = fvarbits;
    for (i = 0, next = 1; i < nv; ++i, ++next) {
        if (next == nv) next = 0;
        HbrHalfedge<T>* opposite = vertices[next]->GetEdge(vertices[i]);
        GetEdge(i)->Initialize(opposite, i, vertices[i], curfvarbits, this);
        if (opposite) opposite->SetOpposite(GetEdge(i));
        if (fvarbits) {
            curfvarbits = curfvarbits + fvarbitsSizePerEdge;
        }
    }
    for (i = 0; i < nv; ++i) {
        vertices[i]->AddIncidentEdge(GetEdge(i));
    }
}

template <class T>
HbrFace<T>::~HbrFace() {
    Destroy();
}

template <class T>
void
HbrFace<T>::Destroy() {
    if (initialized && !destroyed) {
        int i;
#ifdef HBRSTITCH
        const int stitchCount = mesh->GetStitchCount();
#endif

        // Remove children's references to self
        if (children.children) {
            int nchildren = mesh->GetSubdivision()->GetFaceChildrenCount(nvertices);
            if (nchildren > 4) {
                for (i = 0; i < nchildren; ++i) {
                    if (children.extrachildren[i]) {
                        children.extrachildren[i]->parent = 0;
                        children.extrachildren[i] = 0;
                    }
                }
                delete[] children.extrachildren;
                children.extrachildren = 0;
            } else {
                for (i = 0; i < nchildren; ++i) {
                    if ((*children.children)[i]) {
                        (*children.children)[i]->parent = 0;
                        (*children.children)[i] = 0;
                    }
                }
                mesh->DeleteFaceChildren(children.children);
                children.children = 0;
            }
        }

        // Deleting the incident edges from the vertices in this way is
        // the safest way of doing things. Doing it in the halfedge
        // destructor will not work well because it disrupts cycle
        // finding/incident edge replacement in the vertex code.
        // We also take this time to clean up any orphaned stitches
        // still belonging to the edges.
        for (i = 0; i < nvertices; ++i) {
            HbrHalfedge<T> *edge = GetEdge(i);
#ifdef HBRSTITCH
            edge->DestroyStitchEdges(stitchCount);
#endif
            HbrVertex<T>* vertex = edge->GetOrgVertex();
            if (fvarbits) {
                HbrFVarData<T>& fvt = vertex->GetFVarData(this);
                if (fvt.GetFace() == this) {
                    fvt.SetFace(0);
                }
            }
            vertex->RemoveIncidentEdge(edge);
            vertex->UnGuaranteeNeighbors();
        }
        if (extraedges) {
            delete[] extraedges;
            extraedges = 0;
        }

        // Remove parent's reference to self
        if (parent) {
            bool parentHasOtherKids = false;
            int nchildren = mesh->GetSubdivision()->GetFaceChildrenCount(parent->nvertices);
            if (nchildren > 4) {
                for (i = 0; i < nchildren; ++i) {
                    if (parent->children.extrachildren[i] == this) {
                        parent->children.extrachildren[i] = 0;
                    } else if (parent->children.extrachildren[i]) parentHasOtherKids = true;
                }
                // After cleaning the parent's reference to self, the parent
                // may be able to clean itself up
                if (!parentHasOtherKids) {
                    delete[] parent->children.extrachildren;
                    parent->children.extrachildren = 0;
                    if (parent->GarbageCollectable()) {
                        mesh->DeleteFace(parent);
                    }
                }
            } else {
                for (i = 0; i < nchildren; ++i) {
                    if ((*parent->children.children)[i] == this) {
                        (*parent->children.children)[i] = 0;
                    } else if ((*parent->children.children)[i]) parentHasOtherKids = true;
                }
                // After cleaning the parent's reference to self, the parent
                // may be able to clean itself up
                if (!parentHasOtherKids) {
                    mesh->DeleteFaceChildren(parent->children.children);
                    parent->children.children = 0;
                    if (parent->GarbageCollectable()) {
                        mesh->DeleteFace(parent);
                    }
                }
            }
            parent = 0;
        }

        // Orphan the child vertex
        if (vchild) {
            vchild->SetParent(static_cast<HbrFace*>(0));
            vchild = 0;
        }

        if (nvertices > 4 && fvarbits) {
            free(fvarbits);
#ifdef HBRSTITCH
            if (stitchEdges) {
                delete[] stitchEdges;
            }
            if (stitchDatas) {
                delete[] stitchDatas;
            }
#endif
        }
        fvarbits = 0;
#ifdef HBRSTITCH
        stitchEdges = 0;
        stitchDatas = 0;
#endif

        // Make sure the four edges intrinsic to face are properly cleared
        // if they were used
        if (nvertices <= 4) {
            for (i = 0; i < nvertices; ++i) {
                GetEdge(i)->Clear();
            }
        }
        nvertices = 0;
        initialized = 0;
        destroyed = 1;
    }
}

template <class T>
HbrHalfedge<T>*
HbrFace<T>::GetEdge(int index) const {
    assert(index >= 0 && index < nvertices);
    if (nvertices > 4) {
        return extraedges + index;
    } else {
        return const_cast<HbrHalfedge<T>*>(edges + index);
    }
}

template <class T>
HbrVertex<T>*
HbrFace<T>::GetVertex(int index) const {
    assert(index >= 0 && index < nvertices);
    if (nvertices > 4) {
        return extraedges[index].GetOrgVertex();
    } else {
        return edges[index].GetOrgVertex();
    }
}

template <class T>
void
HbrFace<T>::SetChild(int index, HbrFace<T>* face) {
    int nchildren = mesh->GetSubdivision()->GetFaceChildrenCount(nvertices);
    // Construct the children array if it doesn't already exist
    if (!children.children) {
        int i;
        if (nchildren > 4) {
            children.extrachildren = new HbrFace<T>*[nchildren];
            for (i = 0; i < nchildren; ++i) {
                children.extrachildren[i] = 0;
            }
        } else {
            children.children = mesh->NewFaceChildren();
            for (i = 0; i < nchildren; ++i) {
                (*children.children)[i] = 0;
            }
        }
    }
    if (nchildren > 4) {
        children.extrachildren[index] = face;
    } else {
        (*children.children)[index] = face;
    }
    face->parent = this;
}

template <class T>
HbrVertex<T>*
HbrFace<T>::Subdivide() {
    if (vchild) return vchild;
    vchild = mesh->GetSubdivision()->Subdivide(mesh, this);
    vchild->SetParent(this);
    return vchild;
}

template <class T>
void
HbrFace<T>::Refine() {
    mesh->GetSubdivision()->Refine(mesh, this);
}

template <class T>
void
HbrFace<T>::Unrefine() {
    // Delete the children, via the mesh (so that the mesh loses
    // references to the children)
    if (children) {
        int nchildren = mesh->GetSubdivision()->GetFaceChildrenCount(nvertices);
        for (int i = 0; i < nchildren; ++i) {
            if (children[i]) mesh->DeleteFace(children[i]);
        }
        delete[] children;
        children = 0;
    }
}

template <class T>
bool
HbrFace<T>::HasLimit() {
    return mesh->GetSubdivision()->HasLimit(mesh, this);
}

template <class T>
unsigned long
HbrFace<T>::GetMemStats() const {
    return sizeof(HbrFace<T>);
}

template <class T>
void
HbrFace<T>::MarkUsage() {
    // Must increment the usage on all vertices which are in the
    // support for this face
    HbrVertex<T>* v;
    HbrHalfedge<T>* e = GetFirstEdge(), *ee, *eee, *start;
    for (int i = 0; i < nvertices; ++i) {
        v = e->GetOrgVertex();
        v->GuaranteeNeighbors();
        start = v->GetIncidentEdge();
        ee = start;
        do {
            HbrFace<T>* f = ee->GetLeftFace();
            eee = f->GetFirstEdge();
            for (int j = 0; j < f->GetNumVertices(); ++j) {
                eee->GetOrgVertex()->IncrementUsage();
                eee = eee->GetNext();
            }
            ee = v->GetNextEdge(ee);
            if (ee == start) break;
        } while (ee);
        e = e->GetNext();
    }
}

template <class T>
void
HbrFace<T>::ClearUsage() {
    bool gc = false;

    // Must mark all vertices which may affect this face
    HbrVertex<T>* v, *vv;
    HbrHalfedge<T>* e = GetFirstEdge(), *ee, *eee, *start;
    for (int i = 0; i < nvertices; ++i) {
        v = e->GetOrgVertex();
        start = v->GetIncidentEdge();
        ee = start;
        do {
            HbrFace<T>* f = ee->GetLeftFace();
            eee = f->GetFirstEdge();
            for (int j = 0; j < f->GetNumVertices(); ++j) {
                vv = eee->GetOrgVertex();
                vv->DecrementUsage();
                if (!vv->IsUsed()) {
                    mesh->AddGarbageCollectableVertex(vv);
                    gc = true;
                }
                eee = eee->GetNext();
            }
            ee = v->GetNextEdge(ee);
            if (ee == start) break;
        } while (ee);
        e = e->GetNext();
    }
    if (gc) mesh->GarbageCollect();
}

template <class T>
bool
HbrFace<T>::GarbageCollectable() const {
    if (children.children || protect) return false;
    for (int i = 0; i < nvertices; ++i) {
        HbrHalfedge<T>* edge = GetEdge(i);
        HbrVertex<T>* vertex = edge->GetOrgVertex();
        if (vertex->IsUsed()) return false;
        if (!GetParent() && vertex->EdgeRemovalWillMakeSingular(edge)) {
            return false;
        }
    }
    return true;
}

template <class T>
void
HbrFace<T>::SetHierarchicalEdits(HbrHierarchicalEdit<T>** _edits) {
    edits = _edits;

    // Walk the list of edits and look for any which apply locally.
    while (HbrHierarchicalEdit<T>* edit = *_edits) {
        if (!edit->IsRelevantToFace(this)) break;
        edit->ApplyEditToFace(this);
        _edits++;
    }
}

template <class T>
std::ostream& operator<<(std::ostream& out, const HbrFace<T>& face) {
    out << "face " << face.GetID() << ", " << face.GetNumVertices() << " vertices (";
    for (int i = 0; i < face.GetNumVertices(); ++i) {
        HbrHalfedge<T>* e = face.GetEdge(i);
        out << *(e->GetOrgVertex());
        if (e->IsBoundary()) {
            out << " -/-> ";
        } else {
            out << " ---> ";
        }
    }
    out << ")";
    return out;
}

template <class T>
class HbrFaceOperator {
public:
    virtual void operator() (HbrFace<T> &face) = 0;
    virtual ~HbrFaceOperator() {}
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* HBRFACE_H */
