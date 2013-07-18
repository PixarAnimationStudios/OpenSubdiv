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

#ifndef FAR_MESH_FACTORY_H
#define FAR_MESH_FACTORY_H

#include "../version.h"

// Activate Hbr feature adaptive tagging : in order to process the HbrMesh 
// adaptively, some tag data is added to HbrFace, HbrVertex and HbrHalfedge.
// While small, these tags incur some performance costs and are by default
// disabled.
#define HBR_ADAPTIVE

#include "../hbr/mesh.h"
#include "../hbr/bilinear.h"
#include "../hbr/catmark.h"
#include "../hbr/loop.h"

#include "../far/mesh.h"
#include "../far/dispatcher.h"
#include "../far/bilinearSubdivisionTablesFactory.h"
#include "../far/catmarkSubdivisionTablesFactory.h"
#include "../far/loopSubdivisionTablesFactory.h"
#include "../far/patchTablesFactory.h"
#include "../far/vertexEditTablesFactory.h"

#include <typeinfo>
#include <set>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Instantiates a FarMesh from an HbrMesh.
///
/// FarMeshFactory requires a 2 steps process : 
/// 1. Instantiate a FarMeshFactory object from an HbrMesh
/// 2. Call "Create" to obtain the FarMesh instance
///
/// This tiered factory approach offers client-code the opportunity to access
/// useful transient information tied to the lifespan of the factory instance.
/// Specifically, regression code needs to access the remapping tables that
/// tie HbrMesh vertices to their FarMesh counterparts for comparison.
///
template <class T, class U=T> class FarMeshFactory {

public:

    /// \brief Constructor for the factory.
    /// Analyzes the HbrMesh and stores transient data used to create the 
    /// adaptive patch representation. Once the new rep has been instantiated
    /// with 'Create', this factory object can be deleted safely.
    ///
    /// @param mesh        The HbrMesh describing the topology (this mesh *WILL* be
    ///                    modified by this factory).
    ///
    /// @param maxlevel    In uniform subdivision mode : number of levels of
    ///                    subdivision. In feature adaptive mode : maximum 
    ///                    level of isolation around extraordinary topological
    ///                    features.
    ///
    /// @param firstLevel  First level of subdivision to use when building the
    ///                    FarMesh. The default -1 only generates a single patch
    ///                    array for the highest level of subdivision)
    ///                    Note : firstLevel is only applicable if adaptive is false
    ///
    /// @param adaptive Switch between uniform and feature adaptive mode
    ///
    FarMeshFactory(HbrMesh<T> * mesh, int maxlevel, bool adaptive=false, int firstLevel=-1);

    /// \brief Create a table-based mesh representation
    ///
    /// @param requireFVarData create a face-varying table
    ///
    /// @return a pointer to the FarMesh created
    ///
    FarMesh<U> * Create( bool requireFVarData=false );

    /// \brief Computes the minimum number of adaptive feature isolation levels required
    /// in order for the limit surface to be an accurate representation of the 
    /// shape given all the tags and edits.
    ///
    /// @param mesh           The HbrMesh describing the topology 
    ///
    /// @param nfaces         The number of faces in the HbrMesh 
    ///
    /// @param cornerIsolate  The level of isolation desired for patch corners
    ///
    /// @return               The minimum level of isolation of extraordinary
    ///                       topological features.
    ///
    static int ComputeMinIsolation( HbrMesh<T> const * mesh, int nfaces, int cornerIsolate=5 );

    /// \brief The Hbr mesh that this factory is converting
    HbrMesh<T> const * GetHbrMesh() const { return _hbrMesh; }

    /// \brief Maximum level of subidivision supported by this factory
    int GetMaxLevel() const { return _maxlevel; }

    /// \brief The number of coarse vertices found in the HbrMesh before refinement
    ///
    /// @return The number of coarse vertices
    ///
    int GetNumCoarseVertices() const { return _numCoarseVertices; }

    /// \brief Total number of faces up to a given level of subdivision
    ///
    /// @param level  The number of faces up to 'level' of subdivision
    ///
    /// @return       The summation of the number of faces
    ///
    int GetNumFacesTotal(int level) const {
        return sumList<HbrFace<T> *>(_facesList, level);
    }

    /// \brief Returns the corresponding index of the HbrVertex<T> in the new 
    /// FarMesh
    ///
    /// @param v  the vertex
    ///
    /// @return   the remapped index of the vertex in the FarMesh
    ///
    int GetVertexID( HbrVertex<T> * v );

    /// \brief Returns a the mapping between HbrVertex<T>->GetID() and Far 
    /// vertices indices
    ///
    /// @return the table that maps HbrMesh to FarMesh vertex indices
    ///
    std::vector<int> const & GetRemappingTable( ) const { return _remapTable; }

private:
    friend class FarBilinearSubdivisionTablesFactory<T,U>;
    friend class FarCatmarkSubdivisionTablesFactory<T,U>;
    friend class FarLoopSubdivisionTablesFactory<T,U>;
    friend class FarSubdivisionTablesFactory<T,U>;
    friend class FarVertexEditTablesFactory<T,U>;
    friend class FarPatchTablesFactory<T>;

    template <class X> struct VertCompare {
        bool operator()(HbrVertex<X> const * v1, HbrVertex<X> const * v2 ) const {
            //return v1->GetID() < v2->GetID();
            return (void*)(v1) < (void*)(v2); 
        }
    };
    
    // Non-copyable, so these are not implemented:
    FarMeshFactory( FarMeshFactory const & );
    FarMeshFactory<T,U> & operator=(FarMeshFactory<T,U> const &);

    // True if the HbrMesh applies the bilinear subdivision scheme
    static bool isBilinear(HbrMesh<T> const * mesh);

    // True if the HbrMesh applies the Catmull-Clark subdivision scheme
    static bool isCatmark(HbrMesh<T> const * mesh);

    // True if the HbrMesh applies the Loop subdivision scheme
    static bool isLoop(HbrMesh<T> const * mesh);

    // True if the factory is refining adaptively
    bool isAdaptive() { return _adaptive; }

    // False if v prevents a face from being represented with a BSpline
    static bool vertexIsBSpline( HbrVertex<T> * v, bool next );

    // True if a vertex is a regular boundary
    static bool vertexIsRegularBoundary( HbrVertex<T> * v );

    // Non-const accessor to the remapping table
    std::vector<int> & getRemappingTable( ) { return _remapTable; }

    template <class Type> static int sumList( std::vector<std::vector<Type> > const & list, int level );

    // Calls Hbr to refines the neighbors of v
    static void refineVertexNeighbors(HbrVertex<T> * v);

    // Uniformly refine the Hbr mesh
    static void refine( HbrMesh<T> * mesh, int maxlevel );

    // Adaptively refine the Hbr mesh
    int refineAdaptive( HbrMesh<T> * mesh, int maxIsolate );
    
    typedef std::vector<std::vector< HbrFace<T> *> > FacesList;
    
    // Returns sorted vectors of HbrFace<T> pointers sorted by level
    FacesList const & GetFaceList() const { return _facesList; }
    
private:
    HbrMesh<T> * _hbrMesh;

    bool _adaptive;

    int _maxlevel,
        _firstlevel,
        _numVertices,
        _numCoarseVertices,
        _numFaces,
        _maxValence,
        _numPtexFaces;

    // remapping table to translate vertex ID's between Hbr indices and the
    // order of the same vertices in the tables
    std::vector<int> _remapTable;

    FacesList _facesList;
};

template <class T, class U>
    template <class Type> int
FarMeshFactory<T,U>::sumList( std::vector<std::vector<Type> > const & list, int level) {

    level = std::min(level, (int)list.size()-1);
    int total = 0;
    for (int i=0; i<=level; ++i)
        total += (int)list[i].size();
    return total;
}

// Refines non-adaptively an Hbr mesh
template <class T, class U> void
FarMeshFactory<T,U>::refine( HbrMesh<T> * mesh, int maxlevel ) {

    for (int level=0, firstface=0; level<maxlevel; ++level ) {

        int nfaces = mesh->GetNumFaces();
        
        for (int i=firstface; i<nfaces; ++i) {
        
            HbrFace<T> * f = mesh->GetFace(i);

            if (f->GetDepth()==level) {

                if (not f->IsHole()) {
                    f->Refine();
                } else {                

                    // Hole faces need to maintain the 1-ring of vertices so we
                    // have to create an extra row of children faces around the
                    // hole.
                    HbrHalfedge<T> * e = f->GetFirstEdge();
                    for (int i=0; i<f->GetNumVertices(); ++i) {
                        assert(e);
                        if (e->GetRightFace() and (not e->GetRightFace()->IsHole())) {
                        
                            // RefineFaceAtVertex only creates a single child face
                            // centered on the passed vertex
                            HbrSubdivision<T> * s = mesh->GetSubdivision();
                            s->RefineFaceAtVertex(mesh,f,e->GetOrgVertex());
                            s->RefineFaceAtVertex(mesh,f,e->GetDestVertex());
                        }
                        
                        e = e->GetNext();
                    }
                }
            }
        }
        
        // Hbr allocates faces sequentially, so there is no need to iterate over
        // faces that have already been refined.
        firstface = nfaces;
    }

    mesh->SetSubdivisionMethod(HbrMesh<T>::k_SubdivisionMethodUniform);
}

// Scan the faces of a mesh and compute the max level of subdivision required
template <class T, class U> int 
FarMeshFactory<T,U>::ComputeMinIsolation( HbrMesh<T> const * mesh, int nfaces, int cornerIsolate ) {

    assert(mesh);


    int editmax=0; 
    float sharpmax=0.0f;
    
    
    float cornerSharp=0.0; 
    if (mesh->GetInterpolateBoundaryMethod()<HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner)
        cornerSharp = (float) cornerIsolate;

    // Check vertex sharpness
    int nverts = mesh->GetNumVertices();
    for (int i=0; i<nverts; ++i) {
        HbrVertex<T> * v = mesh->GetVertex(i);
        if (not v->OnBoundary())
            sharpmax = std::max( sharpmax, v->GetSharpness() );
        else {
            sharpmax = std::max( sharpmax, cornerSharp );
        }
    }

    // Check edge sharpness and hierarchical edits
    for (int i=0 ; i<nfaces ; ++i) {
    
        HbrFace<T> * f = mesh->GetFace(i);
        
        // We don't need to check non-coarse faces
        if (not f->IsCoarse())
            continue;

        // Check for edits
        if (f->HasVertexEdits()) {

            HbrVertexEdit<T> ** edits = (HbrVertexEdit<T>**)f->GetHierarchicalEdits();

            while (HbrVertexEdit<T> * edit = *edits++) {
                editmax = std::max( editmax , edit->GetNSubfaces() );
            }
        }

        // Check for sharpness
        int nv = f->GetNumVertices();
        for (int j=0; j<nv; ++j) {
            
            HbrHalfedge<T> * e = f->GetEdge(j);
            if (not e->IsBoundary())
                sharpmax = std::max( sharpmax, f->GetEdge(j)->GetSharpness() );
        }
    }

    int result = std::max( (int)ceil(sharpmax)+1, editmax+1 );
    
    // Cap the result to "infinitely sharp" (10)
    return std::min( result, (int)HbrHalfedge<T>::k_InfinitelySharp );
}

// True if a vertex is a regular boundary
template <class T, class U> bool 
FarMeshFactory<T,U>::vertexIsRegularBoundary( HbrVertex<T> * v ) {
    int valence = v->GetValence();    
    return (v->OnBoundary() and (valence==2 or valence==3));
}

// True if the vertex can be incorporated into a B-spline patch
template <class T, class U> bool 
FarMeshFactory<T,U>::vertexIsBSpline( HbrVertex<T> * v, bool next ) {

    int valence = v->GetValence();    
    
    bool isRegBoundary = v->OnBoundary() and (valence==3);
    
    // Extraordinary vertices that are not on a regular boundary
    if (v->IsExtraordinary() and not isRegBoundary )
        return false;
    
    // Irregular boundary vertices (high valence)
    if (v->OnBoundary() and (valence>3))
        return false;
    
    // Creased vertices that aren't corner / boundaries
    if (v->IsSharp(next) and not v->OnBoundary())
        return false;

    return true;
}

// Calls Hbr to refines the neighbors of v
template <class T, class U> void 
FarMeshFactory<T,U>::refineVertexNeighbors(HbrVertex<T> * v) {
    
    assert(v);

    HbrHalfedge<T> * start = v->GetIncidentEdge(),
                   * next=start;
    do {

        HbrFace<T> * lft = next->GetLeftFace(),
                   * rgt = next->GetRightFace();

        if (not ((lft and lft->IsHole()) and 
                 (rgt and rgt->IsHole()) ) ) {
        
            if (rgt)
                rgt->_adaptiveFlags.isTagged=true;

            if (lft)
                lft->_adaptiveFlags.isTagged=true;

            HbrHalfedge<T> * istart = next, 
                           * inext = istart;
            do {
                if (not inext->IsInsideHole()  )
                    inext->GetOrgVertex()->Refine();
                inext = inext->GetNext();
            } while (istart != inext);
        } 
        next = v->GetNextEdge( next );
    } while (next and next!=start);
}


// Refines an Hbr Catmark mesh adaptively around extraordinary features
template <class T, class U> int
FarMeshFactory<T,U>::refineAdaptive( HbrMesh<T> * mesh, int maxIsolate ) {

    int ncoarsefaces = mesh->GetNumCoarseFaces(),
        ncoarseverts = mesh->GetNumVertices();

    int maxlevel = maxIsolate+1;    
    
    // First pass : tag coarse vertices & faces that need refinement

    typedef std::set<HbrVertex<T> *,VertCompare<T> > VertSet;
    VertSet verts, nextverts;
    
    for (int i=0; i<ncoarseverts; ++i) {
        HbrVertex<T> * v = mesh->GetVertex(i);
        
        // Tag non-BSpline vertices for refinement
        if (not vertexIsBSpline(v, false)) {
            v->_adaptiveFlags.isTagged=true;
            nextverts.insert(v);
        }
    }
    
    for (int i=0; i<ncoarsefaces; ++i) {
        HbrFace<T> * f = mesh->GetFace(i);
        
        if (f->IsHole())
            continue;

        bool extraordinary = mesh->GetSubdivision()->FaceIsExtraordinary(mesh,f);

        int nv = f->GetNumVertices();
        for (int j=0; j<nv; ++j) {
            
            HbrHalfedge<T> * e = f->GetEdge(j);
            assert(e);

            // Tag sharp edges for refinement
            if (e->IsSharp(true) and (not e->IsBoundary())) {
                nextverts.insert(e->GetOrgVertex());
                nextverts.insert(e->GetDestVertex());
                
                e->GetOrgVertex()->_adaptiveFlags.isTagged=true;
                e->GetDestVertex()->_adaptiveFlags.isTagged=true;
            }
            
            // Tag extraordinary (non-quad) faces for refinement
            if (extraordinary or f->HasVertexEdits()) {
                HbrVertex<T> * v = f->GetVertex(j);
                v->_adaptiveFlags.isTagged=true;
                nextverts.insert(v);
            }

            // Quad-faces with 2 non-consecutive boundaries need to be flagged
            // for refinement as boundary patches. 
            //
            //  o ........ o ........ o ........ o
            //  .          |          |          .     ... b.undary edge
            //  .          |   needs  |          .
            //  .          |   flag   |          .     --- regular edge
            //  .          |          |          .
            //  o ........ o ........ o ........ o
            //
            if ( e->IsBoundary() and (not f->_adaptiveFlags.isTagged) and nv==4 ) {
            
                if (e->GetPrev() and (not e->GetPrev()->IsBoundary()) and
                    e->GetNext() and (not e->GetNext()->IsBoundary()) and
                    e->GetNext() and e->GetNext()->GetNext() and e->GetNext()->GetNext()->IsBoundary()) {

                    // Tag the face so that we don't check for this again
                    f->_adaptiveFlags.isTagged=true;

                    // Tag all 4 vertices of the face to make sure 4 boundary
                    // sub-patches are generated
                    for (int k=0; k<4; ++k) {
                        HbrVertex<T> * v = f->GetVertex(j);
                        v->_adaptiveFlags.isTagged=true;
                        nextverts.insert(v);
                    }
                }
            }
        }
        _maxValence = std::max(_maxValence, nv);
    }
    

    // Second pass : refine adaptively around singularities
    
    for (int level=0; level<maxlevel-1; ++level) {

        verts = nextverts;
        nextverts.clear();

        // Refine vertices
        for (typename VertSet::iterator i=verts.begin(); i!=verts.end(); ++i) {

            HbrVertex<T> * v = *i;
            assert(v);

            if (level>0)
                v->_adaptiveFlags.isTagged=true;
            else
                v->_adaptiveFlags.wasTagged=true;

            refineVertexNeighbors(v);

            // Tag non-BSpline vertices for refinement
            if (not vertexIsBSpline(v, true))
                nextverts.insert(v->Subdivide());

            // Refine edges with creases or edits
            int valence = v->GetValence();
            _maxValence = std::max(_maxValence, valence);

            HbrHalfedge<T> * e = v->GetIncidentEdge();
            for (int j=0; j<valence; ++j) {

                // Skip edges that have already been processed (HasChild())
                if ((not e->HasChild()) and e->IsSharp(false) and (not e->IsBoundary())) {
                
                    if (not e->IsInsideHole()) {
                        nextverts.insert( e->Subdivide() );
                        nextverts.insert( e->GetOrgVertex()->Subdivide() );
                        nextverts.insert( e->GetDestVertex()->Subdivide() );
                    }
                }
                HbrHalfedge<T> * next = v->GetNextEdge(e);
                e = next ? next : e->GetPrev();
            }

            // Flag verts with hierarchical edits for neighbor refinement at the next level
            HbrVertex<T> * childvert = v->Subdivide();
            HbrHalfedge<T> * childedge = childvert->GetIncidentEdge();
            assert( childvert->GetValence()==valence);
            for (int j=0; j<valence; ++j) {
                HbrFace<T> * f = childedge->GetFace();
                if (f->HasVertexEdits()) {
                    int nv = f->GetNumVertices();
                    for (int k=0; k<nv; ++k)
                        nextverts.insert( f->GetVertex(k) );
                }
                if ((childedge = childvert->GetNextEdge(childedge)) == NULL)
                    break;
            }
        }

        // Add coarse verts from extraordinary faces
        if (level==0) {
            for (int i=0; i<ncoarsefaces; ++i) {
                HbrFace<T> * f = mesh->GetFace(i);
                assert (f->IsCoarse());
                
                if (mesh->GetSubdivision()->FaceIsExtraordinary(mesh,f))
                    nextverts.insert( f->Subdivide() );
            }
        }
    }
    
    mesh->SetSubdivisionMethod(HbrMesh<T>::k_SubdivisionMethodFeatureAdaptive);
    
    return maxlevel-1;
}

// Assumption : the order of the vertices in the HbrMesh could be set in any
// random order, so the builder runs 2 passes over the entire vertex list to
// gather the counters needed to generate the indexing tables.
template <class T, class U>
FarMeshFactory<T,U>::FarMeshFactory( HbrMesh<T> * mesh, int maxlevel, bool adaptive, int firstlevel ) :
    _hbrMesh(mesh),
    _adaptive(adaptive),
    _maxlevel(maxlevel),
    _firstlevel(firstlevel),
    _numVertices(-1),
    _numCoarseVertices(-1),
    _numFaces(-1),
    _maxValence(3),
    _numPtexFaces(-1),
    _facesList(maxlevel+1)
{
    _numCoarseVertices = mesh->GetNumVertices();
    _numPtexFaces = getNumPtexFaces(mesh);
    
    // Subdivide the Hbr mesh up to maxlevel.
    //
    // Note : using a placeholder vertex class 'T' can greatly speed up the 
    // topological analysis if the interpolation results are not used.
    if (adaptive)
        _maxlevel=refineAdaptive( mesh, maxlevel );
    else
        refine( mesh, maxlevel);
    
    _numFaces = mesh->GetNumFaces();

    _numVertices = mesh->GetNumVertices();
    
    if (not adaptive) {

        // Populate the face lists
        int fsize=0;
        for (int i=0; i<_numFaces; ++i) {
            HbrFace<T> * f = mesh->GetFace(i);
            assert(f);
            if (f->GetDepth()==0 and (not f->IsHole()))
                fsize += mesh->GetSubdivision()->GetFaceChildrenCount( f->GetNumVertices() );
        }

        _facesList[0].reserve(mesh->GetNumCoarseFaces());
        _facesList[1].reserve(fsize);
        for (int l=2; l<=maxlevel; ++l)
            _facesList[l].reserve( _facesList[l-1].capacity()*4 );
        
        for (int i=0; i<_numFaces; ++i) {
            HbrFace<T> * f = mesh->GetFace(i);
            if (f->GetDepth()<=maxlevel and (not f->IsHole()))
                _facesList[ f->GetDepth() ].push_back(f);
        }
    }
}

template <class T, class U> bool
FarMeshFactory<T,U>::isBilinear(HbrMesh<T> const * mesh) {
    return typeid(*(mesh->GetSubdivision()))==typeid(HbrBilinearSubdivision<T>);
}

template <class T, class U> bool
FarMeshFactory<T,U>::isCatmark(HbrMesh<T> const * mesh) {
    return typeid(*(mesh->GetSubdivision()))==typeid(HbrCatmarkSubdivision<T>);
}

template <class T, class U> bool
FarMeshFactory<T,U>::isLoop(HbrMesh<T> const * mesh) {
    return typeid(*(mesh->GetSubdivision()))==typeid(HbrLoopSubdivision<T>);
}

template <class T, class U> void
copyVertex( T & dest, U const & src ) {
}

template <class T> void
copyVertex( T & dest, T const & src ) {
    dest = src;
}

template <class T> int
getNumPtexFaces(HbrMesh<T> const * hmesh) {

    HbrFace<T> * lastface = hmesh->GetFace(hmesh->GetNumFaces()-1);
    assert(lastface);

    int result = lastface->GetPtexIndex();

    result += (hmesh->GetSubdivision()->FaceIsExtraordinary(hmesh, lastface) ?
                  lastface->GetNumVertices() : 1);

    return result;
}

// Computes per-face or per-patch local ptex texture coordinates.
template <class T> FarPatchParam *
computePatchParam(HbrFace<T> const *f, FarPatchParam *coord) {

    short u,v;
    unsigned short ofs = 1;
    unsigned char depth;
    bool nonquad = false;

    if (coord == NULL) return NULL;

    // save the rotation state of the coarse face
    unsigned char rots = f->_adaptiveFlags.rots;

    // track upwards towards coarse parent face, accumulating u,v indices
    HbrFace<T> const * p = f->GetParent();
    for ( u=v=depth=0;  p!=NULL; depth++ ) {

        int nverts = p->GetNumVertices();
        if ( nverts != 4 ) {           // non-quad coarse face : stop accumulating offsets
            nonquad = true;            // set non-quad bit
            break;
        }

        for (unsigned char i=0; i<nverts; ++i) {
            if ( p->GetChild( i )==f ) {
                switch ( i ) {
                    case 0 :                     break;
                    case 1 : { u+=ofs;         } break;
                    case 2 : { u+=ofs; v+=ofs; } break;
                    case 3 : {         v+=ofs; } break;
                }
                break;
            }
        }
        ofs = ofs << 1;
        f = p;
        p = f->GetParent();
    }

    coord->Set( f->GetPtexIndex(), u, v, rots, depth, nonquad );

    return ++coord;
}

template <class T> float *
computeFVarData(HbrFace<T> const *f, const int width, float *coord, bool isAdaptive) {

    if (coord == NULL) return NULL;

    if (isAdaptive) {

        int rots = f->_adaptiveFlags.rots;
        int nverts = f->GetNumVertices();
        assert(nverts==4);

        for ( int j=0; j < nverts; ++j ) {

            HbrVertex<T> *v      = f->GetVertex((j+rots)%4);
            float        *fvdata = v->GetFVarData(f).GetData(0);

            for ( int k=0; k<width; ++k ) {
                (*coord++) = fvdata[k];
            }
        }

    } else {

        // for each face vertex copy face-varying data into coord pointer
        int nverts = f->GetNumVertices();
        for ( int j=0; j < nverts; ++j ) {

            HbrVertex<T> *v      = f->GetVertex(j);
            float        *fvdata = v->GetFVarData(f).GetData(0);

            for ( int k=0; k<width; ++k ) {
                (*coord++) = fvdata[k];
            }
        }
    }

    // pass back pointer to next destination
    return coord;
}

template <class T, class U> FarMesh<U> *
FarMeshFactory<T,U>::Create( bool requireFVarData ) {

    assert( GetHbrMesh() );

    // Note : we cannot create a Far rep of level 0 (coarse mesh)
    if (GetMaxLevel()<1)
        return 0;

    FarMesh<U> * result = new FarMesh<U>();
    
    if ( isBilinear( GetHbrMesh() ) ) {
        result->_subdivisionTables = FarBilinearSubdivisionTablesFactory<T,U>::Create(this, result, &result->_batches);
    } else if ( isCatmark( GetHbrMesh() ) ) {
        result->_subdivisionTables = FarCatmarkSubdivisionTablesFactory<T,U>::Create(this, result, &result->_batches);
    } else if ( isLoop(GetHbrMesh()) ) {
        result->_subdivisionTables = FarLoopSubdivisionTablesFactory<T,U>::Create(this, result, &result->_batches);
    } else
        assert(0);
    assert(result->_subdivisionTables);

    // If the vertex classes aren't place-holders, copy the data of the coarse
    // vertices into the vertex buffer.
    if (sizeof(U)>1) {
        result->_vertices.resize( _numVertices );
        for (int i=0; i<GetNumCoarseVertices(); ++i)
            copyVertex(result->_vertices[i], GetHbrMesh()->GetVertex(i)->GetData());
    }

    // Create the element indices tables (patches for adaptive, quads for non-adaptive)
    if (isAdaptive()) {

        FarPatchTablesFactory<T> factory(GetHbrMesh(), _numFaces, _remapTable);

        // XXXX: currently PatchGregory shader supports up to 29 valence
        result->_patchTables = factory.Create(GetMaxLevel()+1, _maxValence, requireFVarData);

    } else {
        result->_patchTables = FarPatchTablesFactory<T>::Create(GetHbrMesh(), _facesList, _remapTable, _firstlevel, requireFVarData );
    }
    assert( result->_patchTables );

    result->_numPtexFaces = _numPtexFaces;
    
    if (requireFVarData) {
        result->_totalFVarWidth = _hbrMesh->GetTotalFVarWidth();
    }

    // Create VertexEditTables if necessary
    if (GetHbrMesh()->HasVertexEdits()) {
        result->_vertexEditTables = FarVertexEditTablesFactory<T,U>::Create( this, result, &result->_batches, GetMaxLevel() );
        assert(result->_vertexEditTables);
    }
    
    return result;
}

template <class T, class U> int
FarMeshFactory<T,U>::GetVertexID( HbrVertex<T> * v ) {
    assert( v  and (v->GetID() < _remapTable.size()) );
    return _remapTable[ v->GetID() ];
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_MESH_FACTORY_H */
