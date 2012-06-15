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
#ifndef HBRMESH_H
#define HBRMESH_H

#include <algorithm>
#include <cstring>
#include <deque>
#include <list>
#include <set>
#include <iostream>

#if PRMAN
#include "path-to-TgSpinLock"
#elif MENV
#include "bedrock/tf/mutex.h"
#else
#include "IlmThreadMutex.h"
#endif

#include "../hbr/vertex.h"
#include "../hbr/face.h"
#include "../hbr/hierarchicalEdit.h"
#include "../hbr/vertexEdit.h"
#include "../hbr/creaseEdit.h"
#include "../hbr/allocator.h"

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrSubdivision;

template <class T> class HbrMesh {
public:
    HbrMesh(HbrSubdivision<T>* subdivision = 0, int fvarcount = 0, const int *fvarindices = 0, const int *fvarwidths = 0, int totalfvarwidth = 0
#ifdef HBRSTITCH
    , int stitchCount = 0
#endif
    );
    ~HbrMesh();
    
    // Create vertex with the indicated ID and data
    HbrVertex<T>* NewVertex(int id, const T &data);

    // Create vertex with the indicated data. The ID will be assigned
    // by the mesh.
    HbrVertex<T>* NewVertex(const T &data);
    
    // Create vertex without an ID - one will be assigned by the mesh,
    // and the data implicitly created will share the same id
    HbrVertex<T>* NewVertex();

    // Ask for vertex with the indicated ID
    HbrVertex<T>* GetVertex(int id) const;

    // Create face from a list of vertex IDs
    HbrFace<T>* NewFace(int nvertices, int *vtx, int uindex);

    // Create face from a list of vertices
    HbrFace<T>* NewFace(int nvertices, HbrVertex<T>** vtx, HbrFace<T>* parent, int childindex);

    // "Create" a new uniform index
    int NewUniformIndex() { return ++maxUniformIndex; }

    // Finishes initialization of the mesh
    void Finish();
    
    // Remove the indicated face from the mesh
    void DeleteFace(HbrFace<T>* face);

    // Remove the indicated vertex from the mesh
    void DeleteVertex(HbrVertex<T>* vertex);
    
    // Returns number of vertices in the mesh
    int GetNumVertices() const;

    // Returns number of disconnected vertices in the mesh
    int GetNumDisconnectedVertices() const;

    // Returns number of faces in the mesh    
    int GetNumFaces() const;

    // Returns number of coarse faces in the mesh    
    int GetNumCoarseFaces() const;

    // Ask for face with the indicated ID
    HbrFace<T>* GetFace(int id) const;
    
    // Returns a collection of all vertices in the mesh
    void GetVertices(std::list<HbrVertex<T>*>& vertices) const;

    // Applies operator to all vertices
    void ApplyOperatorAllVertices(HbrVertexOperator<T> &op) const;
    
    // Returns a collection of all faces in the mesh
    void GetFaces(std::list<HbrFace<T>*>& faces) const;

    // Returns the subdivision method
    HbrSubdivision<T>* GetSubdivision() const { return subdivision; }

    // Return the number of facevarying variables
    int GetFVarCount() const { return fvarcount; }

    // Return a table of the start index of each facevarying variable
    const int *GetFVarIndices() const { return fvarindices; }
    
    // Return a table of the size of each facevarying variable
    const int *GetFVarWidths() const { return fvarwidths; }

    // Return the sum size of facevarying variables per vertex
    int GetTotalFVarWidth() const { return totalfvarwidth; }

#ifdef HBRSTITCH
    int GetStitchCount() const { return stitchCount; }
#endif

    void PrintStats(std::ostream& out);

    // Returns memory statistics
    size_t GetMemStats() const { return m_memory; }

    // Interpolate boundary management
    enum InterpolateBoundaryMethod {
	k_InterpolateBoundaryNone,
	k_InterpolateBoundaryEdgeOnly,
	k_InterpolateBoundaryEdgeAndCorner,
        k_InterpolateBoundaryAlwaysSharp
    };

    InterpolateBoundaryMethod GetInterpolateBoundaryMethod() const { return interpboundarymethod; }
    void SetInterpolateBoundaryMethod(InterpolateBoundaryMethod method) { interpboundarymethod = method; }
    InterpolateBoundaryMethod GetFVarInterpolateBoundaryMethod() const { return fvarinterpboundarymethod; }
    void SetFVarInterpolateBoundaryMethod(InterpolateBoundaryMethod method) { fvarinterpboundarymethod = method; }

    bool GetFVarPropagateCorners() const { return fvarpropagatecorners; }
    void SetFVarPropagateCorners(bool p) { fvarpropagatecorners = p; }
    
    // Register routines for keeping track of memory usage
    void RegisterMemoryRoutines(void (*increment)(unsigned long bytes), void (*decrement)(unsigned long bytes)) {
	m_faceAllocator.SetMemStatsIncrement(increment);
	m_faceAllocator.SetMemStatsDecrement(decrement);
	m_vertexAllocator.SetMemStatsIncrement(increment);
	m_vertexAllocator.SetMemStatsDecrement(decrement);
	s_memStatsIncrement = increment;
	s_memStatsDecrement = decrement;
    }

    // Add a vertex to consider for garbage collection. All
    // neighboring faces of that vertex will be examined to see if
    // they can be deleted
    void AddGarbageCollectableVertex(HbrVertex<T>* vertex) {
        if (!m_transientMode) {
            assert(vertex);
            if (!vertex->IsCollected()) {
                gcVertices.push_back(vertex); vertex->SetCollected();
            }
        }
    }

    // Apply garbage collection to the mesh
    void GarbageCollect();

    // Add a new hierarchical edit to the mesh
    void AddHierarchicalEdit(HbrHierarchicalEdit<T>* edit);

    // Return a pointer to the beginning of the list of hierarchical edits
    HbrHierarchicalEdit<T>** GetHierarchicalEdits() const { return hierarchicalEditArray; }

    // Whether the mesh has certain types of edits
    bool HasVertexEdits() const { return hasVertexEdits; }
    bool HasCreaseEdits() const { return hasCreaseEdits; }
    
    void Unrefine(int numCoarseVerts, int numCoarseFaces) {
	static int oldMaxFaceID = 0;
	if(oldMaxFaceID == 0) {
	    oldMaxFaceID = numCoarseFaces;
	}
	for (int i = numCoarseFaces; i < maxFaceID; ++i) {
	    if (faces[i]) {
		HbrFace<T>* f = faces[i];
		if(f && not f->IsCoarse())
		    DeleteFace(f);
	    }
	}
	//oldMaxFaceID = maxFaceID;
	maxFaceID = numCoarseFaces;

        int vert = numCoarseVerts % vsetsize;
        for( int set=(numCoarseVerts/vsetsize); set<nvsets; set++ ) {	    	    
	    for( ; vert<vsetsize; vert++ ) {
	        HbrVertex<T>* v = vertices[set][vert];
		if(v && not v->IsReferenced()) 
		    DeleteVertex(v);
	    }
            vert = 0;
	}
    }

    // Whether the mesh is in "transient" mode, i.e. all
    // vertices/faces that are created should be deemed temporary
    void SetTransientMode(bool mode) {
        m_transientMode = mode;
    }
    
    void FreeTransientData();
    
private:
    // The mutex type depends on where hbr is being used.
    #if PRMAN
        typedef TgSpinLock Mutex;
    #elif MENV
        typedef TfMutex Mutex;
    #else
        typedef IlmThread::Mutex Mutex;
    #endif

    #if PRMAN or MENV
    // Helper class used to automatically unlock a mutex
    // when an instance of this class goes out of scope.
    class ScopedLock {
    public:
         ScopedLock(Mutex *mutex) : _mutex(mutex) {
             mutex->Lock();
	 }
	
        ~ScopedLock() {
            Release();
        }
        void Release() {
            if (_mutex) {
                _mutex->Unlock();
            }
            _mutex = NULL;
        }
    private:
        Mutex *_mutex;
    };
    #endif
    
    // Mutex used to lock access to the "vertices" data member.
    mutable Mutex m_verticesMutex;

private:
    
    // Subdivision method used in this mesh
    HbrSubdivision<T>* subdivision;

    // Number of facevarying datums
    int fvarcount;

    // Start indices of the facevarying data we want to store
    const int *fvarindices;

    // Individual widths of the facevarying data we want to store
    const int *fvarwidths;

    // Total widths of the facevarying data
    const int totalfvarwidth;

#ifdef HBRSTITCH
    const int stitchCount;
#endif
    
    // Vertices which comprise this mesh
    HbrVertex<T>*** vertices;
    int nvsets;
    const int vsetsize;

    // Faces which comprise this mesh
    HbrFace<T>** faces;
    int nfaces;

    // Maximum vertex ID - may be needed when generating a unique
    // vertex ID
    int maxVertexID;

    // Maximum face ID - needed when generating a unique face ID
    int maxFaceID;

    // Maximum uniform index - needed to generate a new uniform index
    int maxUniformIndex;

    // Boundary interpolation method
    InterpolateBoundaryMethod interpboundarymethod;

    // Facevarying boundary interpolation method
    InterpolateBoundaryMethod fvarinterpboundarymethod;

    // Whether facevarying corners propagate their sharpness
    bool fvarpropagatecorners;
    
    // Memory statistics tracking routines
    HbrMemStatFunction s_memStatsIncrement;
    HbrMemStatFunction s_memStatsDecrement;

    // Vertices which may be garbage collected
    std::deque<HbrVertex<T>*> gcVertices;

    // List of vertex IDs which may be recycled
    std::set<int> recycleIDs;

    // Sorted hierarchical edits. This set is valid only until
    // Finish() is called, at which point the mesh should switch over
    // to using hierarchicalEditArray
    std::multiset<HbrHierarchicalEdit<T>*, HbrHierarchicalEditComparator<T> > hierarchicalEditSet;

    // Sorted array of hierarchical edits. This array is valid only
    // after Finish() has been called. Note that HbrFaces have
    // pointers directly into this array so manipulation of it should
    // be avoided
    int nHierarchicalEdits;
    HbrHierarchicalEdit<T>** hierarchicalEditArray;

    // Size of faces (including 4 facevarying bits and stitch edges)
    const size_t m_faceSize;
    HbrAllocator<HbrFace<T> > m_faceAllocator;

    // Size of vertices (includes storage for one piece of facevarying data)
    const size_t m_vertexSize;
    HbrAllocator<HbrVertex<T> > m_vertexAllocator;    

    // Memory used by this mesh alone, plus all its faces and vertices
    size_t m_memory;

    // Number of coarse faces. Initialized at Finish()
    int m_numCoarseFaces;
    
    // Flags which indicate whether the mesh has certain types of
    // edits
    unsigned hasVertexEdits:1;
    unsigned hasCreaseEdits:1;

    // True if the mesh is in "transient" mode, meaning all
    // vertices/faces that are created via NewVertex/NewFace should be
    // deemed temporary
    bool m_transientMode;

    // Vertices which are transient
    std::deque<HbrVertex<T>*> m_transientVertices;

    // Faces which are transient
    std::deque<HbrFace<T>*> m_transientFaces;
    
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#include <algorithm>
#include "../hbr/mesh.h"
#include "../hbr/halfedge.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T>
HbrMesh<T>::HbrMesh(HbrSubdivision<T>* s, int _fvarcount, const int *_fvarindices, const int *_fvarwidths, int _totalfvarwidth
#ifdef HBRSTITCH
    , int _stitchCount
#endif
    )
    : subdivision(s), fvarcount(_fvarcount), fvarindices(_fvarindices),
      fvarwidths(_fvarwidths), totalfvarwidth(_totalfvarwidth),
#ifdef HBRSTITCH
      stitchCount(_stitchCount),
#endif
      vertices(0), nvsets(0),
      vsetsize(2048), faces(0), nfaces(0),
      maxVertexID(0), maxFaceID(0), maxUniformIndex(0),
      interpboundarymethod(k_InterpolateBoundaryNone),
      fvarinterpboundarymethod(k_InterpolateBoundaryNone),
      fvarpropagatecorners(false),
      s_memStatsIncrement(0), s_memStatsDecrement(0),
      nHierarchicalEdits(0),
      hierarchicalEditArray(0),
      m_faceSize(sizeof(HbrFace<T>) + 4 *
                 ((fvarcount + 15) / 16 * sizeof(unsigned int)
#ifdef HBRSTITCH
                 + stitchCount * sizeof(StitchEdge*)
                 + sizeof(void*) // for stitch data
#endif
                  )),
      m_faceAllocator(&m_memory, 64, 0, 0, m_faceSize),
      m_vertexSize(sizeof(HbrVertex<T>) +
                   sizeof(HbrHalfedge<T>*) + // for incidentEdges[1]
                   totalfvarwidth * sizeof(float) + sizeof(HbrFVarData<T>)),
      m_vertexAllocator(&m_memory, 64, 0, 0, m_vertexSize),
      m_memory(0),
      m_numCoarseFaces(-1),
      hasVertexEdits(0),
      hasCreaseEdits(0),
      m_transientMode(false) {
}

template <class T>
HbrMesh<T>::~HbrMesh() {
    GarbageCollect();

    int i;
    if (faces) {
	for (i = 0; i < nfaces; ++i) {	
	    if (faces[i]) {
		faces[i]->Destroy();
		m_faceAllocator.Deallocate(faces[i]);
	    }
	}
	if (s_memStatsDecrement) {
	    s_memStatsDecrement(nfaces * sizeof(HbrFace<T>*));
	}
        m_memory -= nfaces * sizeof(HbrFace<T>*);
	delete[] faces;
    }

    if (nvsets) {
        for (int vi = 0; vi < nvsets; ++vi) {
            HbrVertex<T>** vset = vertices[vi];
            for (i = 0; i < vsetsize; ++i) {
                if (vset[i]) {
                    vset[i]->Destroy();
                    m_vertexAllocator.Deallocate(vset[i]);
                }
            }
            delete[] vset;
            if (s_memStatsDecrement) {
                s_memStatsDecrement(vsetsize * sizeof(HbrVertex<T>*));
            }
            m_memory -= vsetsize * sizeof(HbrVertex<T>*);
	}
        delete[] vertices;
    }
    if (hierarchicalEditArray) {
	for (i = 0; i < nHierarchicalEdits; ++i) {
	    delete hierarchicalEditArray[i];
	}
	delete[] hierarchicalEditArray;
    }
}

template <class T>
HbrVertex<T>*
HbrMesh<T>::NewVertex(int id, const T &data) {
    HbrVertex<T>* v = 0;

    int arrayindex = id / vsetsize;
    int vertindex = id % vsetsize;
#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
    HbrVertex<T>** vset = 0;
    if (arrayindex >= nvsets) {
        HbrVertex<T>*** nvertices = new HbrVertex<T>**[arrayindex + 1];
        for (int i = 0; i < nvsets; ++i) {
            nvertices[i] = vertices[i];
        }
        for (int i = nvsets; i < arrayindex + 1; ++i) {
            vset = new HbrVertex<T>*[vsetsize];
            if (s_memStatsIncrement) {
                s_memStatsIncrement(vsetsize * sizeof(HbrVertex<T>*));
            }
            m_memory += vsetsize * sizeof(HbrVertex<T>*);
            memset(vset, 0, vsetsize * sizeof(HbrVertex<T>*));
            nvertices[i] = vset;
        }
        nvsets = arrayindex + 1;
        delete[] vertices;
        vertices = nvertices;
    }
    vset = vertices[arrayindex];
#if PRMAN or MENV    
    lock.Release();
#else
    lock.release();
#endif    
    v = vset[vertindex];
    if (v) {
	v->Destroy();
    } else {
	v = m_vertexAllocator.Allocate();
    }
    v->Initialize(id, data, GetTotalFVarWidth());
    vset[vertindex] = v;
    
    if (id >= maxVertexID) {
	maxVertexID = id + 1;
    }

    // Newly created vertices are always candidates for garbage
    // collection, until they get "owned" by someone who
    // IncrementsUsage on the vertex.
    AddGarbageCollectableVertex(v);

    // If mesh is in transient mode, add vertex to transient list
    if (m_transientMode) {
        m_transientVertices.push_back(v);
    }
    return v;
}

template <class T>
HbrVertex<T>*
HbrMesh<T>::NewVertex(const T &data) {
    // Pick an ID - either the maximum vertex ID or a recycled ID if
    // we can
    int id = maxVertexID;
    if (!recycleIDs.empty()) {
	id = *recycleIDs.begin();
        recycleIDs.erase(recycleIDs.begin());
    }
    if (id >= maxVertexID) {
	maxVertexID = id + 1;
    }
    return NewVertex(id, data);
}

template <class T>
HbrVertex<T>*
HbrMesh<T>::NewVertex() {
    // Pick an ID - either the maximum vertex ID or a recycled ID if
    // we can
    int id = maxVertexID;
    if (!recycleIDs.empty()) {
	id = *recycleIDs.begin();
        recycleIDs.erase(recycleIDs.begin());        
    }
    if (id >= maxVertexID) {
	maxVertexID = id + 1;
    }
    T data(id);
    data.Clear();
    return NewVertex(id, data);
}

template <class T>
HbrVertex<T>*
HbrMesh<T>::GetVertex(int id) const {
    int arrayindex = id / vsetsize;
    int vertindex = id % vsetsize;

#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
    if (arrayindex >= nvsets) {
        return 0;
    }
    HbrVertex<T>** vset = vertices[arrayindex];
    return vset[vertindex];
}

template <class T>
HbrFace<T>*
HbrMesh<T>::NewFace(int nv, int *vtx, int uindex) {
    HbrVertex<T>** facevertices = reinterpret_cast<HbrVertex<T>**>(alloca(sizeof(HbrVertex<T>*) * nv));
    int i;
    for (i = 0; i < nv; ++i) {
	facevertices[i] = GetVertex(vtx[i]);
	if (!facevertices[i]) {
	    return 0;
	}
    }
    HbrFace<T> *f = 0;
    // Resize if needed
    if (nfaces <= maxFaceID) {
	int nnfaces = nfaces;
	while (nnfaces <= maxFaceID) {
            nnfaces *= 2;
            if (nnfaces < 1) nnfaces = 1;
	}
	HbrFace<T>** newfaces = new HbrFace<T>*[nnfaces];
	if (s_memStatsIncrement) {
	    s_memStatsIncrement(nnfaces * sizeof(HbrFace<T>*));
	}
        m_memory += nnfaces * sizeof(HbrFace<T>*);
	if (faces) {
	    for (i = 0; i < nfaces; ++i) {
		newfaces[i] = faces[i];
	    }
	    if (s_memStatsDecrement) {
		s_memStatsDecrement(nfaces * sizeof(HbrFace<T>*));
	    }
            m_memory -= nfaces * sizeof(HbrFace<T>*);
	    delete[] faces;
	}
	for (i = nfaces; i < nnfaces; ++i) {
	    newfaces[i] = 0;
	}
	faces = newfaces;
	nfaces = nnfaces;
    }
    f = faces[maxFaceID];
    if (f) {
	f->Destroy();
    } else {
	f = m_faceAllocator.Allocate();
    }
    f->Initialize(this, NULL, -1, maxFaceID, uindex, nv, facevertices, totalfvarwidth, 0);
    faces[maxFaceID] = f;
    maxFaceID++;
    // Update the maximum encountered uniform index
    if (uindex > maxUniformIndex) maxUniformIndex = uindex;

    // If mesh is in transient mode, add face to transient list
    if (m_transientMode) {
        m_transientFaces.push_back(f);
    }
    return f;
}

template <class T>
HbrFace<T>*
HbrMesh<T>::NewFace(int nv, HbrVertex<T> **vtx, HbrFace<T>* parent, int childindex) {
    HbrFace<T> *f = 0;
    // Resize if needed
    if (nfaces <= maxFaceID) {
	int nnfaces = nfaces;
	while (nnfaces <= maxFaceID) {
            nnfaces *= 2;
            if (nnfaces < 1) nnfaces = 1;
	}
	HbrFace<T>** newfaces = new HbrFace<T>*[nnfaces];
	if (s_memStatsIncrement) {
	    s_memStatsIncrement(nnfaces * sizeof(HbrFace<T>*));
	}
        m_memory += nnfaces * sizeof(HbrFace<T>*);
	if (faces) {
	    for (int i = 0; i < nfaces; ++i) {
		newfaces[i] = faces[i];
	    }
	    if (s_memStatsDecrement) {
		s_memStatsDecrement(nfaces * sizeof(HbrFace<T>*));
	    }
            m_memory -= nfaces * sizeof(HbrFace<T>*);
	    delete[] faces;
	}
	for (int i = nfaces; i < nnfaces; ++i) {
	    newfaces[i] = 0;
	}
	faces = newfaces;
	nfaces = nnfaces;
    }    
    f = faces[maxFaceID];
    if (f) {
	f->Destroy();
    } else {
	f = m_faceAllocator.Allocate();
    }
    f->Initialize(this, parent, childindex, maxFaceID, parent ? parent->GetUniformIndex() : 0, nv, vtx, totalfvarwidth, parent ? parent->GetDepth() + 1 : 0);
    if (parent) {
        f->SetPtexIndex(parent->GetPtexIndex());
    }
    faces[maxFaceID] = f;
    maxFaceID++;

    // If mesh is in transient mode, add face to transient list
    if (m_transientMode) {
        m_transientFaces.push_back(f);
    }
    return f;
}

template <class T>
void
HbrMesh<T>::Finish() {
    int i, j;
    m_numCoarseFaces = 0;
    for (i = 0; i < nfaces; ++i) {
	if (faces[i]) {
            faces[i]->SetCoarse();
            m_numCoarseFaces++;
        }
    }

    std::list<HbrVertex<T>*> vertexlist;
    GetVertices(vertexlist);
    for (typename std::list<HbrVertex<T>*>::iterator vi = vertexlist.begin();
         vi != vertexlist.end(); ++vi) {
        HbrVertex<T>* vertex = *vi;
        if (vertex->IsConnected()) vertex->Finish();
    }
    // If interpolateboundary is on, process boundary edges
    if (interpboundarymethod == k_InterpolateBoundaryEdgeOnly || interpboundarymethod == k_InterpolateBoundaryEdgeAndCorner) {
	for (i = 0; i < nfaces; ++i) {
	    if (HbrFace<T>* face = faces[i]) {
		int nv = face->GetNumVertices();
		for (int k = 0; k < nv; ++k) {
		    HbrHalfedge<T>* edge = face->GetEdge(k);
		    if (edge->IsBoundary()) {
			edge->SetSharpness(HbrHalfedge<T>::k_InfinitelySharp);
		    }
		}
	    }
	}
    }
    // Process corners
    if (interpboundarymethod == k_InterpolateBoundaryEdgeAndCorner) {
        for (typename std::list<HbrVertex<T>*>::iterator vi = vertexlist.begin();
             vi != vertexlist.end(); ++vi) {
            HbrVertex<T>* vertex = *vi;
            if (vertex && vertex->IsConnected() && vertex->OnBoundary() && vertex->GetCoarseValence() == 2) {
                vertex->SetSharpness(HbrVertex<T>::k_InfinitelySharp);
            }
        }
    }
    // Convert the sorted set of hierarchical edits to an array
    nHierarchicalEdits = hierarchicalEditSet.size();
    if (nHierarchicalEdits) {
	// Size the array by one extra; the last will be a sentinel
	// value (null)
	hierarchicalEditArray = new HbrHierarchicalEdit<T>*[nHierarchicalEdits + 1];
	i = 0;
	for (typename std::multiset<HbrHierarchicalEdit<T>*, HbrHierarchicalEditComparator<T> >::iterator pi = hierarchicalEditSet.begin(); pi != hierarchicalEditSet.end(); ++pi) {
	    hierarchicalEditArray[i++] = *pi;
	}
	assert(i == nHierarchicalEdits);
	hierarchicalEditArray[i] = 0;
	j = 0;
	// Link faces to hierarchical edits
	for (i = 0; i < nfaces; ++i) {
	    if (faces[i]) {
		while (j < nHierarchicalEdits && hierarchicalEditArray[j]->GetFaceID() < i) {
		    ++j;
		}
		if (j < nHierarchicalEdits && hierarchicalEditArray[j]->GetFaceID() == i) {
		    faces[i]->SetHierarchicalEdits(&hierarchicalEditArray[j]);
		}
	    }
	}
	hierarchicalEditSet.clear();
    }
}

template <class T>
void
HbrMesh<T>::DeleteFace(HbrFace<T>* face) {
    if (face->GetID() < nfaces) {
	HbrFace<T>* f = faces[face->GetID()];
	if (f == face) {
	    faces[face->GetID()] = 0;
	    face->Destroy();
	    m_faceAllocator.Deallocate(face);
	}
    }
}

template <class T>
void
HbrMesh<T>::DeleteVertex(HbrVertex<T>* vertex) {
    HbrVertex<T> *v = GetVertex(vertex->GetID());
    if (v == vertex) {
        recycleIDs.insert(vertex->GetID());
        int id = vertex->GetID();
        int arrayindex = id / vsetsize;
#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
        int vertindex = id % vsetsize;
        HbrVertex<T>** vset = vertices[arrayindex];
#if PRMAN or MENV    
    lock.Release();
#else
    lock.release();
#endif    
        vset[vertindex] = 0;
        vertex->Destroy();
        m_vertexAllocator.Deallocate(vertex);
    }
}

template <class T>
int
HbrMesh<T>::GetNumVertices() const {
    int count = 0;
#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
    for (int vi = 0; vi < nvsets; ++vi) {
        HbrVertex<T>** vset = vertices[vi];
        for (int i = 0; i < vsetsize; ++i) {
            if (vset[i]) count++;
        }
    }
    return count;
}

template <class T>
int
HbrMesh<T>::GetNumDisconnectedVertices() const {
    int disconnected = 0;
#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
    for (int vi = 0; vi < nvsets; ++vi) {
        HbrVertex<T>** vset = vertices[vi];
        for (int i = 0; i < vsetsize; ++i) {
            if (HbrVertex<T>* v = vset[i]) {
                if (!v->IsConnected()) {
                    disconnected++;
                }
            }
        }
    }
    return disconnected;
}

template <class T>
int
HbrMesh<T>::GetNumFaces() const {
    int count = 0;
    for (int i = 0; i < nfaces; ++i) {
	if (faces[i]) count++;
    }
    return count;
}

template <class T>
int
HbrMesh<T>::GetNumCoarseFaces() const {
    // Use the value computed by Finish() if it exists
    if (m_numCoarseFaces >= 0) return m_numCoarseFaces;
    // Otherwise we have to just count it up now
    int count = 0;
    for (int i = 0; i < nfaces; ++i) {
	if (faces[i] && faces[i]->IsCoarse()) count++;
    }
    return count;
}

template <class T>
HbrFace<T>*
HbrMesh<T>::GetFace(int id) const {
    if (id < nfaces) {
	return faces[id];
    }
    return 0;
}

template <class T>
void
HbrMesh<T>::GetVertices(std::list<HbrVertex<T>*>& lvertices) const {
#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
    for (int vi = 0; vi < nvsets; ++vi) {
        HbrVertex<T>** vset = vertices[vi];
        for (int i = 0; i < vsetsize; ++i) {
            if (vset[i]) lvertices.push_back(vset[i]);
        }
    }
}

template <class T>
void
HbrMesh<T>::ApplyOperatorAllVertices(HbrVertexOperator<T> &op) const {
#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
    for (int vi = 0; vi < nvsets; ++vi) {
        HbrVertex<T>** vset = vertices[vi];
        for (int i = 0; i < vsetsize; ++i) {
            if (vset[i]) op(*vset[i]);
        }
    }
}

template <class T>
void
HbrMesh<T>::GetFaces(std::list<HbrFace<T>*>& lfaces) const {
    for (int i = 0; i < nfaces; ++i) {
	if (faces[i]) lfaces.push_back(faces[i]);
    }
}

template <class T>
void
HbrMesh<T>::PrintStats(std::ostream &out) {
    int singular = 0;
    int sumvalence = 0;
    int i, nv = 0;
    int disconnected = 0;
    int extraordinary = 0;
    for (int vi = 0; vi < nvsets; ++vi) {
        HbrVertex<T>** vset = vertices[vi];
        for (i = 0; i < vsetsize; ++i) {
            if (HbrVertex<T>* v = vset[i]) {
                nv++;
                if (v->IsSingular()) {
                    out << "  singular: " << *v << "\n";
                    singular++;
                }
                else if (!v->IsConnected()) {
                    out << "  disconnected: " << *v << "\n";		
                    disconnected++;
                } else {
                    if (v->IsExtraordinary()) {
                        extraordinary++;
                    }
                    sumvalence += v->GetValence();
                }
            }
        }
    }
    out << "Mesh has " << nv << " vertices\n";
    out << "Total singular vertices " << singular << "\n";
    out << "Total disconnected vertices " << disconnected << "\n";
    out << "Total extraordinary vertices " << extraordinary << "\n";
    out << "Average valence " << (float) sumvalence / nv << "\n";

    int sumsides = 0;
    int numfaces = 0;
    for (i = 0; i < nfaces; ++i) {
	if (HbrFace<T>* f = faces[i]) {
	    numfaces++;
	    sumsides += f->GetNumVertices();
	}
    }
    out << "Mesh has " << nfaces << " faces\n";
    out << "Average sidedness " << (float) sumsides / nfaces << "\n";
}

#define HBR_MESH_BUFFERSIZE 4096

template <class T>
void
HbrMesh<T>::GarbageCollect() {
    if (gcVertices.empty()) return;

    if (gcVertices.size() <= HBR_MESH_BUFFERSIZE) return;
    // Go through the list of garbage collectable vertices and gather
    // up the neighboring faces of those vertices which can be garbage
    // collected.
    std::list<HbrFace<T>*> killlist;
    std::list<HbrVertex<T>*> vlist;

    while (gcVertices.size() > HBR_MESH_BUFFERSIZE / 2) {
	HbrVertex<T>* v = gcVertices.front(); gcVertices.pop_front();
	v->ClearCollected();
	if (v->IsUsed()) continue;
	vlist.push_back(v);
	HbrHalfedge<T>* start = v->GetIncidentEdge(), *edge;
	edge = start;
	while (edge) {
	    HbrFace<T>* f = edge->GetLeftFace();
	    if (!f->IsCollected()) {
                f->SetCollected();
		killlist.push_back(f);
	    }
	    edge = v->GetNextEdge(edge);
	    if (edge == start) break;
	}
    }

    // Delete those faces
    for (typename std::list<HbrFace<T>*>::iterator fi = killlist.begin(); fi != killlist.end(); ++fi) {
	if ((*fi)->GarbageCollectable()) {
	    DeleteFace(*fi);
	} else {
	    (*fi)->ClearCollected();
        }
    }

    // Delete as many vertices as we can
    for (typename std::list<HbrVertex<T>*>::iterator vi = vlist.begin(); vi != vlist.end(); ++vi) {
	HbrVertex<T>* v = *vi;
	if (!v->IsReferenced()) {
	    DeleteVertex(v);
	}
    }
}

template <class T>
void
HbrMesh<T>::AddHierarchicalEdit(HbrHierarchicalEdit<T>* edit) {
    hierarchicalEditSet.insert(edit);
    if (dynamic_cast<HbrVertexEdit<T>*>(edit) ||
	dynamic_cast<HbrMovingVertexEdit<T>*>(edit)) {
	hasVertexEdits = 1;
    } else if (dynamic_cast<HbrCreaseEdit<T>*>(edit)) {
	hasCreaseEdits = 1;
    }
}

template <class T>
void
HbrMesh<T>::FreeTransientData() {
    // When purging transient data, we must clear the faces first
    for (typename std::deque<HbrFace<T>*>::iterator fi = m_transientFaces.begin();
         fi != m_transientFaces.end(); ++fi) {
        DeleteFace(*fi);
    }
    // The vertices should now be trivial to purge after the transient
    // faces have been cleared
    for (typename std::deque<HbrVertex<T>*>::iterator vi = m_transientVertices.begin();
         vi != m_transientVertices.end(); ++vi) {
        DeleteVertex(*vi);
    }
    m_transientVertices.clear();
    m_transientFaces.clear();
    // Reset max face ID
    int i;
    for (i = nfaces - 1; i >= 0; --i) {
        if (faces[i]) {
            maxFaceID = i + 1;
            break;
        }
    }
    // Reset max vertex ID. Slightly more complicated
#if PRMAN or MENV    
    ScopedLock lock(&m_verticesMutex);
#else
    IlmThread::Lock lock(m_verticesMutex);
#endif    
    for (i = (nvsets * vsetsize) - 1; i >= 0; --i) {
        int arrayindex = i / vsetsize;
        int vertindex = i % vsetsize;
        if (vertices[arrayindex][vertindex]) {
            maxVertexID = i + 1;
            break;
        }
    }
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* HBRMESH_H */
