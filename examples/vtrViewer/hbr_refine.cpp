//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include "hbr_refine.h"

#include <version.h>
#include <far/patchTables.h>

// !!! WARNING !!!
//
// The Far::PatchTablesFactory code duplicated in this file is for debugging
// puproses only ! 
//
// Do *NOT* use, duplicate or rely on this code.

//------------------------------------------------------------------------------
//
// refine the Hbr mesh uniformly
//
void
RefineUniform(Hmesh & mesh, int maxlevel,
    std::vector<Hface const *> & refinedFaces) {

    int nfaces = mesh.GetNumFaces();

    for (int level=0, firstface=0; level<=maxlevel; ++level ) {

        if (level==maxlevel) {

            refinedFaces.resize(nfaces-firstface);
            for (int i=firstface, ofs=0; i<nfaces; ++i) {
                refinedFaces[ofs++]=mesh.GetFace(i);
            }
        } else {

            for (int i=firstface; i<nfaces; ++i) {
                Hface * f = mesh.GetFace(i);
                assert(f->GetDepth()==level);
                if (not f->IsHole()) {
                    f->Refine();
                }
            }
        }

        // Hbr allocates faces sequentially, skip faces that have
        // already been refined.
        firstface = nfaces;
        nfaces = mesh.GetNumFaces();
    }
}


//------------------------------------------------------------------------------

struct VertCompare {
    bool operator() (Hvertex const * v1, Hvertex const * v2 ) const {
        //return v1->GetID() < v2->GetID();
        return (void*)(v1) < (void*)(v2);
    }
};

//------------------------------------------------------------------------------
// True if the vertex can be incorporated into a B-spline patch
bool
vertexIsBSpline(Hvertex * v, bool next) {

    int valence = v->GetValence();

    // Boundary & corner vertices
    if (v->OnBoundary()) {
        if (valence==2) {
            // corner vertex

            Hface * f = v->GetFace();

            // the vertex may not need isolation depending on boundary
            // interpolation rule (sharp vs. rounded corner)
            Hmesh::InterpolateBoundaryMethod method =
                f->GetMesh()->GetInterpolateBoundaryMethod();

            if (method==Hmesh::k_InterpolateBoundaryEdgeAndCorner) {
                if (not next) {
                    // if we are checking coarse vertices (next==false),
                    // count the number of corners in the face, because we
                    // can only have 1 corner vertex in a corner patch.
                    int nsharpboundaries=0;
                    for (int i=0; i<f->GetNumVertices(); ++i) {
                        Hhalfedge * e = f->GetEdge(i);
                        if (e->IsBoundary() and
                            e->GetSharpness()==Hhalfedge::k_InfinitelySharp) {
                            ++nsharpboundaries;
                        }
                    }
                    return nsharpboundaries < 3 ? true: false;
                } else
                    return true;
            } else
                return false;
        } else if (valence>3) {
            // extraordinary boundary vertex (high valence)
            return false;
        }
        // regular boundary vertices have valence 3
        return true;
    }

    // Extraordinary or creased vertices that aren't corner / boundaries
    if (v->IsExtraordinary() or v->IsSharp(next))
        return false;

    return true;
}

//------------------------------------------------------------------------------
void
refineVertexNeighbors(Hvertex * v) {

    assert(v);

    Hhalfedge * start = v->GetIncidentEdge(),
                               * next=start;
    do {

        Hface * lft = next->GetLeftFace(),
                               * rgt = next->GetRightFace();

        if (not ((lft and lft->IsHole()) and
                 (rgt and rgt->IsHole()) ) ) {

            if (rgt)
                rgt->_adaptiveFlags.isTagged=true;

            if (lft)
                lft->_adaptiveFlags.isTagged=true;

            Hhalfedge * istart = next,
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


//------------------------------------------------------------------------------
//
// refine the Hbr mesh adaptively
//
int
RefineAdaptive(Hmesh & mesh, int maxlevel,
    std::vector<Hface const *> & refinedFaces) {

    int ncoarsefaces = mesh.GetNumCoarseFaces(),
        ncoarseverts = mesh.GetNumVertices(),
        maxValence=0;

    // First pass : tag coarse vertices & faces that need refinement

    typedef std::set<Hvertex *, VertCompare> VertSet;
    VertSet verts, nextverts;

    for (int i=0; i<ncoarseverts; ++i) {

        Hvertex * v = mesh.GetVertex(i);

        // Non manifold topology may leave un-connected vertices that need to be skipped
        if (not v->IsConnected()) {
            continue;
        }

        // Tag non-BSpline vertices for refinement
        if (not vertexIsBSpline(v, false)) {
            v->_adaptiveFlags.isTagged=true;
            nextverts.insert(v);
        }
    }

    for (int i=0; i<ncoarsefaces; ++i) {
        Hface * f = mesh.GetFace(i);

        if (f->IsHole())
            continue;

        bool extraordinary = mesh.GetSubdivision()->FaceIsExtraordinary(&mesh,f);

        int nv = f->GetNumVertices();
        for (int j=0; j<nv; ++j) {

            Hhalfedge * e = f->GetEdge(j);
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
                Hvertex * v = f->GetVertex(j);
                v->_adaptiveFlags.isTagged=true;
                nextverts.insert(v);
            }

            // Quad-faces with 2 non-consecutive boundaries need to be flagged
            // for refinement as boundary patches.
            //
            //  o ........ o ........ o ........ o
            //  .          |          |          .     ... boundary edge
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
                        Hvertex * v = f->GetVertex(k);
                        v->_adaptiveFlags.isTagged=true;
                        nextverts.insert(v);
                    }
                }
            }
        }
        maxValence = std::max(maxValence, nv);
    }


    // Second pass : refine adaptively around singularities
    for (int level=0; level<maxlevel; ++level) {

        verts = nextverts;
        nextverts.clear();

        // Refine vertices
        for (VertSet::iterator i=verts.begin(); i!=verts.end(); ++i) {

            Hvertex * v = *i;
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
            maxValence = std::max(maxValence, valence);

            Hhalfedge * e = v->GetIncidentEdge();
            for (int j=0; j<valence; ++j) {

                // Skip edges that have already been processed (HasChild())
                if ((not e->HasChild()) and e->IsSharp(false) and (not e->IsBoundary())) {

                    if (not e->IsInsideHole()) {
                        nextverts.insert( e->Subdivide() );
                        nextverts.insert( e->GetOrgVertex()->Subdivide() );
                        nextverts.insert( e->GetDestVertex()->Subdivide() );
                    }
                }
                Hhalfedge * next = v->GetNextEdge(e);
                e = next ? next : e->GetPrev();
            }

            // Flag verts with hierarchical edits for neighbor refinement at the next level
            Hvertex * childvert = v->Subdivide();
            Hhalfedge * childedge = childvert->GetIncidentEdge();
            assert( childvert->GetValence()==valence);
            for (int j=0; j<valence; ++j) {
                Hface * f = childedge->GetFace();
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
                Hface * f = mesh.GetFace(i);
                assert (f->IsCoarse());

                if (mesh.GetSubdivision()->FaceIsExtraordinary(&mesh,f))
                    nextverts.insert( f->Subdivide() );
            }
        }
    }

    int nfaces = mesh.GetNumFaces();

    // First pass : identify transition / watertight-critical
    for (int i=0; i<nfaces; ++i) {

        Hface * f = mesh.GetFace(i);

        if (f->_adaptiveFlags.isTagged and (not f->IsHole())) {
            Hvertex * v = f->Subdivide();
            assert(v);
            v->_adaptiveFlags.wasTagged=true;
        }

        int nv = f->GetNumVertices();
        for (int j=0; j<nv; ++j) {

            if (f->IsCoarse())
                f->GetVertex(j)->_adaptiveFlags.wasTagged=true;

            Hhalfedge * e = f->GetEdge(j);

            // Flag transition edge that require a triangulated transition
            if (f->_adaptiveFlags.isTagged) {
                e->_adaptiveFlags.isTriangleHead=true;

                // Both half-edges need to be tagged if an opposite exists
                if (e->GetOpposite())
                    e->GetOpposite()->_adaptiveFlags.isTriangleHead=true;
            }

            Hface * left = e->GetLeftFace(),
                                   * right = e->GetRightFace();

            if (not (left and right))
                continue;

            // a tagged edge w/ no children is inside a hole
            if (e->HasChild() and (left->_adaptiveFlags.isTagged ^ right->_adaptiveFlags.isTagged)) {

                e->_adaptiveFlags.isTransition = true;

                Hvertex * child = e->Subdivide();
                assert(child);

                // These edges will require extra rows of CVs to maintain water-tightness
                // Note : vertices inside holes have no children
                if (e->GetOrgVertex()->HasChild()) {
                    Hhalfedge * org = child->GetEdge(e->GetOrgVertex()->Subdivide());
                    if (org)
                        org->_adaptiveFlags.isWatertightCritical=true;
                }

                if (e->GetDestVertex()->HasChild()) {
                    Hhalfedge * dst = child->GetEdge(e->GetDestVertex()->Subdivide());
                    if (dst)
                        dst->_adaptiveFlags.isWatertightCritical=true;
                }
            }
        }
    }

    refinedFaces.reserve(nfaces - ncoarsefaces);
    for (int i=ncoarsefaces; i<nfaces; ++i) {

        Hface const * f = mesh.GetFace(i);

        if (f->_adaptiveFlags.isTagged) {
            continue;
        }

        refinedFaces.push_back(mesh.GetFace(i));
    }

    return maxValence;
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class Far::PatchTablesFactory {

public:

    static Far::PatchTables const * Create(Hmesh & mesh, int maxvalence);

private:

    typedef Far::PatchDescriptor Descriptor;

    // Returns true if one of v's neighboring faces has vertices carrying the tag "wasTagged"
    static bool vertexHasTaggedNeighbors(Hvertex * v);

    // Returns the rotation for a boundary patch
    static unsigned char computeBoundaryPatchRotation( Hface * f );

    // Returns the rotation for a corner patch
    static unsigned char computeCornerPatchRotation( Hface * f );

    // Populates the patch parametrization descriptor 'coord' for the given face
    // returns a pointer to the next descriptor
    static OpenSubdiv::Far::PatchParam * computePatchParam(Hface const *f, OpenSubdiv::Far::PatchParam *coord);

    // Populates an array of indices with the "one-ring" vertices for the given face
    static Far::Index * getOneRing( Hface const * f, int ringsize, Far::Index const * remap, Far::Index * result );

    // Populates the Gregory patch quad offsets table
    static void getQuadOffsets( Hface const * f, unsigned int * result );

    // The number of patches in the mesh
    static int getNumPatches( Far::PatchTables::PatchArrayVector const & parrays );

    // Reserves tables based on the contents of the PatchArrayVector
    static void allocateTables( Far::PatchTables * tables, int nlevels, int fvarwidth );

    // A convenience container for the different types of feature adaptive patches
    template<class TYPE> struct PatchTypes {

        static const int NUM_TRANSITIONS=6,
                         NUM_ROTATIONS=4;

        TYPE R[NUM_TRANSITIONS],                   // regular patch
             B[NUM_TRANSITIONS][NUM_ROTATIONS],    // boundary patch (4 rotations)
             C[NUM_TRANSITIONS][NUM_ROTATIONS],    // corner patch (4 rotations)
             G,                                    // gregory patch
             GB,                                   // gregory boundary patch
             GP;                                   // gregory basis

        PatchTypes() { memset(this, 0, sizeof(PatchTypes<TYPE>)); }

        // Returns the number of patches based on the patch type in the descriptor
        TYPE & getValue( Descriptor desc );

        // Counts the number of arrays required to store each type of patch used
        // in the primitive
        int getNumPatchArrays() const;
    };

    typedef PatchTypes<OpenSubdiv::Far::PatchParam *> ParamPointers;
    typedef PatchTypes<Far::Index*>                 CVPointers;
    typedef PatchTypes<float *>                     FVarPointers;
    typedef PatchTypes<int>                         Counter;

};

//------------------------------------------------------------------------------
template <class TYPE> TYPE &
Far::PatchTablesFactory::PatchTypes<TYPE>::getValue( Far::PatchDescriptor desc ) {

    switch (desc.GetType()) {
        case Far::PatchDescriptor::REGULAR          : return R[desc.GetPattern()];
        case Far::PatchDescriptor::SINGLE_CREASE    : break;
        case Far::PatchDescriptor::BOUNDARY         : return B[desc.GetPattern()][desc.GetRotation()];
        case Far::PatchDescriptor::CORNER           : return C[desc.GetPattern()][desc.GetRotation()];
        case Far::PatchDescriptor::GREGORY          : return G;
        case Far::PatchDescriptor::GREGORY_BOUNDARY : return GB;
        case Far::PatchDescriptor::GREGORY_BASIS    : return GP;
        default : assert(0);
    }
    // can't be reached (suppress compiler warning)
    return R[0];
}

template <class TYPE> int
Far::PatchTablesFactory::PatchTypes<TYPE>::getNumPatchArrays() const {

    int result=0;

    for (int i=0; i<6; ++i) {

        if (R[i]) ++result;

        for (int j=0; j<4; ++j) {
            if (B[i][j]) ++result;
            if (C[i][j]) ++result;

        }
    }

    if (G) ++result;
    if (GB) ++result;

    return result;
}

//------------------------------------------------------------------------------
// True if the surrounding faces are "tagged" (unsupported feature : watertight
// critical patches)
bool
Far::PatchTablesFactory::vertexHasTaggedNeighbors(Hvertex * v) {

    assert(v);

    Hhalfedge * start = v->GetIncidentEdge(),
                               * next=start;
    do {
        Hface * right = next->GetRightFace(),
                               * left = next->GetLeftFace();

        if (right and (not right->hasTaggedVertices()))
            return true;

        if (left and (not left->hasTaggedVertices()))
            return true;

        next = v->GetNextEdge(next);

    } while (next and next!=start);
    return false;
}

// Returns a rotation index for boundary patches (range [0-3])
unsigned char
Far::PatchTablesFactory::computeBoundaryPatchRotation( Hface * f ) {
    unsigned char rot=0;
    for (unsigned char i=0; i<4;++i) {
        if (f->GetVertex(i)->OnBoundary() and
            f->GetVertex((i+1)%4)->OnBoundary())
            break;
        ++rot;
    }
    return rot;
}

// Returns a rotation index for corner patches (range [0-3])
unsigned char
Far::PatchTablesFactory::computeCornerPatchRotation( Hface * f ) {
    unsigned char rot=0;
    for (unsigned char i=0; i<4; ++i) {
        if (not f->GetVertex((i+3)%4)->OnBoundary())
            break;
        ++rot;
    }
    return rot;
}
/*
int
Far::PatchTablesFactory::getNumPatches( Far::PatchTables::PatchArrayVector const & parrays ) {

    int result=0;
    for (int i=0; i<(int)parrays.size(); ++i) {
        result += parrays[i].GetNumPatches();
    }

    return result;
}
*/
//------------------------------------------------------------------------------
void
Far::PatchTablesFactory::allocateTables( Far::PatchTables * tables, int /* nlevels */, int fvarwidth ) {

    int nverts = 0, npatches = 0;
    for (int i=0; i<tables->GetNumPatchArrays(); ++i) {
        npatches += tables->GetNumPatches(i);
        nverts += tables->GetNumControlVertices(i);
    }

    if (nverts==0 or npatches==0)
        return;

    tables->_patchVerts.resize( nverts );

    tables->_paramTable.resize( npatches );

    if (fvarwidth>0) {
        //Far::PatchTables::PatchArrayVector const & parrays = tables->GetPatchArrayVector();
        //int nfvarverts = 0;
        //for (int i=0; i<(int)parrays.size(); ++i) {
        //    nfvarverts += parrays[i].GetNumPatches() *
        //                  (parrays[i].GetDescriptor().GetType() == Far::PatchTables::TRIANGLES ? 3 : 4);
        //}

        //tables->_fvarData._data.resize( nfvarverts * fvarwidth );

        //if (nlevels >1) {
        //    tables->_fvarData._offsets.resize( nlevels );
        //}
    }
}

//------------------------------------------------------------------------------
Far::PatchTables const *
Far::PatchTablesFactory::Create(Hmesh & mesh, int maxvalence) {

    int nfaces = mesh.GetNumFaces();

    Counter patchCtr;  // counters for full and transition patches

    // Second pass : count boundaries / identify transition constellation
    for (int i=0; i<nfaces; ++i) {

        Hface * f = mesh.GetFace(i);

        if (mesh.GetSubdivision()->FaceIsExtraordinary(&mesh,f))
            continue;

        if (f->IsHole())
            continue;

        bool isTagged=0, wasTagged=0, isConnected=0, isWatertightCritical=0, isExtraordinary=0;
        int  triangleHeads=0, boundaryVerts=0;

        int nv = f->GetNumVertices();
        for (int j=0; j<nv; ++j) {
            Hvertex * v = f->GetVertex(j);

            if (v->OnBoundary()) {
                boundaryVerts++;

                // Boundary vertices with valence higher than 3 aren't Full Boundary
                // patches, they are Gregory Boundary patches.
                if (v->IsSingular() or v->GetValence()>3)
                    isExtraordinary=true;

            } else if (v->IsExtraordinary())
                isExtraordinary=true;

            if (f->GetParent() and (not isWatertightCritical))
                isWatertightCritical = vertexHasTaggedNeighbors(v);

            if (v->_adaptiveFlags.isTagged)
                isTagged=1;

            if (v->_adaptiveFlags.wasTagged)
                wasTagged=1;

            // Count the number of triangle heads to find which transition
            // pattern to use.
            Hhalfedge * e = f->GetEdge(j);
            if (e->_adaptiveFlags.isTriangleHead) {

                ++triangleHeads;
                if (f->GetEdge((j+1)%4)->_adaptiveFlags.isTriangleHead)
                    isConnected=true;
            }
        }

        f->_adaptiveFlags.bverts=boundaryVerts;
        f->_adaptiveFlags.isCritical=isWatertightCritical;

        // Regular Boundary Patch
        if (wasTagged)
            // XXXX manuelk - need to implement end patches
            f->_adaptiveFlags.patchType = Hface::kEnd;

        if (f->_adaptiveFlags.isTagged)
            continue;

        assert(f->_adaptiveFlags.rots==0 and nv==4);

        if (not isTagged and wasTagged) {

            if (triangleHeads==0) {

                 if (not isExtraordinary and boundaryVerts!=1) {

                    // Full Patches
                    f->_adaptiveFlags.patchType = Hface::kFull;

                    switch (boundaryVerts) {

                        case 0 : {   // Regular patch
                                     patchCtr.R[Far::PatchDescriptor::NON_TRANSITION]++;
                                 } break;

                        case 2 : {   // Boundary patch
                                     f->_adaptiveFlags.rots=computeBoundaryPatchRotation(f);
                                     patchCtr.B[Far::PatchDescriptor::NON_TRANSITION][0]++;
                                 } break;

                        case 3 : {   // Corner patch
                                     f->_adaptiveFlags.rots=computeCornerPatchRotation(f);
                                     patchCtr.C[Far::PatchDescriptor::NON_TRANSITION][0]++;
                                 } break;

                        default : break;
                    }
                } else {

                    // Default to Gregory Patch
                    f->_adaptiveFlags.patchType = Hface::kGregory;

                    switch (boundaryVerts) {

                        case 0 : {   // Regular Gregory patch
                                     patchCtr.G++;
                                 } break;


                        default : { // Boundary Gregory patch
                                     patchCtr.GB++;
                                  } break;
                    }
                }

            } else {

                // Transition Patch

                // Resolve transition constellation : 5 types (see p.5 fig. 7)
                switch (triangleHeads) {

                    case 1 : {   for (unsigned char j=0; j<4; ++j) {
                                     if (f->GetEdge(j)->IsTriangleHead())
                                         break;
                                     f->_adaptiveFlags.rots++;
                                 }
                                 f->_adaptiveFlags.transitionType = Hface::kTransition0;
                            } break;

                    case 2 : {   for (unsigned char j=0; j<4; ++j) {
                                     if (isConnected) {
                                         if (f->GetEdge(j)->IsTriangleHead() and
                                             f->GetEdge((j+3)%4)->IsTriangleHead())
                                             break;
                                     } else {
                                         if (f->GetEdge(j)->IsTriangleHead())
                                             break;
                                     }
                                     f->_adaptiveFlags.rots++;
                                 }

                                 if (isConnected)
                                    f->_adaptiveFlags.transitionType = Hface::kTransition1;
                                 else
                                    f->_adaptiveFlags.transitionType = Hface::kTransition4;
                             } break;

                    case 3 : {   for (unsigned char j=0; j<4; ++j) {
                                     if (not f->GetEdge(j)->IsTriangleHead())
                                         break;
                                     f->_adaptiveFlags.rots++;
                                 }
                                 f->_adaptiveFlags.transitionType = Hface::kTransition2;
                             } break;

                    case 4 : f->_adaptiveFlags.transitionType = Hface::kTransition3;
                             break;

                    default: break;
                }

                int pattern = f->_adaptiveFlags.transitionType;
                assert(pattern>=0);

                // Correct rotations for corners & boundaries
                if (not isExtraordinary and boundaryVerts!=1) {

                    switch (boundaryVerts) {

                        case 0 : {   // regular patch
                                     patchCtr.R[pattern+1]++;
                                 } break;

                        case 2 : {   // boundary patch
                                     unsigned char rot=computeBoundaryPatchRotation(f);

                                     f->_adaptiveFlags.brots=(4-f->_adaptiveFlags.rots+rot)%4;

                                     f->_adaptiveFlags.rots=rot; // override the transition rotation

                                     patchCtr.B[pattern+1][f->_adaptiveFlags.brots]++;
                                 } break;

                        case 3 : {   // corner patch
                                     unsigned char rot=computeCornerPatchRotation(f);

                                     f->_adaptiveFlags.brots=(4-f->_adaptiveFlags.rots+rot)%4;

                                     f->_adaptiveFlags.rots=rot; // override the transition rotation

                                     patchCtr.C[pattern+1][f->_adaptiveFlags.brots]++;
                                 } break;

                        default : assert(0); break;
                    }
                } else {
                    // Use Gregory Patch transition ?
                }
            }
        }
    }


    static const Far::Index remapRegular        [16] = {5,6,10,9,4,0,1,2,3,7,11,15,14,13,12,8};
    static const Far::Index remapRegularBoundary[12] = {1,2,6,5,0,3,7,11,10,9,8,4};
    static const Far::Index remapRegularCorner  [ 9] = {1,2,5,4,0,8,7,6,3};

    int fvarwidth=0;

    Far::PatchTables * result = new Far::PatchTables(maxvalence);

    // Populate the patch array descriptors
    result->reservePatchArrays(patchCtr.getNumPatchArrays());

    typedef Far::PatchDescriptorVector DescVec;

    DescVec const & catmarkDescs = Far::PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SCHEME_CATMARK);

    int voffset=0, poffset=0, qoffset=0;
    for (DescVec::const_iterator it=catmarkDescs.begin(); it!=catmarkDescs.end(); ++it) {
        result->pushPatchArray(*it, patchCtr.getValue(*it), &voffset, &poffset, &qoffset );
    }

    //result->_fvarData._fvarWidth = fvarwidth;
    result->_numPtexFaces = 0;

    // Allocate various tables
    allocateTables( result, 0, fvarwidth );

    if ((patchCtr.G > 0) or (patchCtr.GB > 0)) { // Quad-offsets tables (for Gregory patches)
        result->_quadOffsetsTable.resize( patchCtr.G*4 + patchCtr.GB*4 );
    }

    // Setup convenience pointers at the beginning of each patch array for each
    // table (patches, ptex, fvar)
    CVPointers    iptrs;
    ParamPointers pptrs;
    FVarPointers  fptrs;

    for (DescVec::const_iterator it=catmarkDescs.begin(); it!=catmarkDescs.end(); ++it) {

        Index arrayIndex = result->findPatchArray(*it);
        if (arrayIndex==Vtr::INDEX_INVALID) {
            continue;
        }

        iptrs.getValue( *it ) = result->getPatchArrayVertices(arrayIndex).begin();
        pptrs.getValue( *it ) = result->getPatchParams(arrayIndex).begin();
    }

    unsigned int * quad_G_C0_P = patchCtr.G>0 ? &result->_quadOffsetsTable[0] : 0,
                 * quad_G_C1_P = patchCtr.GB>0 ? &result->_quadOffsetsTable[patchCtr.G*4] : 0;

    // Populate patch index tables with vertex indices
    for (int i=0; i<nfaces; ++i) {

        Hface * f = mesh.GetFace(i);

        if (not f->isTransitionPatch() ) {

            // Full / End patches

            if (f->_adaptiveFlags.patchType==Hface::kFull) {
                if (not f->_adaptiveFlags.isExtraordinary and f->_adaptiveFlags.bverts!=1) {

                    int pattern = Far::PatchDescriptor::NON_TRANSITION,
                        rot = 0;

                    switch (f->_adaptiveFlags.bverts) {
                        case 0 : {   // Regular Patch (16 CVs)
                                     iptrs.R[pattern] = getOneRing(f, 16, remapRegular, iptrs.R[0]);
                                     pptrs.R[pattern] = computePatchParam(f, pptrs.R[0]);
                                     //fptrs.R[pattern] = computeFVarData(f, fvarwidth, fptrs.R[0], /*isAdaptive=*/true);
                                 } break;

                        case 2 : {   // Boundary Patch (12 CVs)
                                     f->_adaptiveFlags.brots = (f->_adaptiveFlags.rots+1)%4;
                                     iptrs.B[pattern][rot] = getOneRing(f, 12, remapRegularBoundary, iptrs.B[0][0]);
                                     pptrs.B[pattern][rot] = computePatchParam(f, pptrs.B[0][0]);
                                     //fptrs.B[pattern][rot] = computeFVarData(f, fvarwidth, fptrs.B[0][0], /*isAdaptive=*/true);
                                 } break;

                        case 3 : {   // Corner Patch (9 CVs)
                                     f->_adaptiveFlags.brots = (f->_adaptiveFlags.rots+1)%4;
                                     iptrs.C[pattern][rot] = getOneRing(f, 9, remapRegularCorner, iptrs.C[0][0]);
                                     pptrs.C[pattern][rot] = computePatchParam(f, pptrs.C[0][0]);
                                     //fptrs.C[pattern][rot] = computeFVarData(f, fvarwidth, fptrs.C[0][0], /*isAdaptive=*/true);
                                 } break;

                        default : assert(0);
                    }
                }
            } else if (f->_adaptiveFlags.patchType==Hface::kGregory) {

                if (f->_adaptiveFlags.bverts==0) {

                    // Gregory Regular Patch (4 CVs + quad-offsets / valence tables)
                    for (int j=0; j<4; ++j)
                        iptrs.G[j] = f->GetVertex(j)->GetID();
                    iptrs.G+=4;
                    getQuadOffsets(f, quad_G_C0_P);
                    quad_G_C0_P += 4;
                    pptrs.G = computePatchParam(f, pptrs.G);
                    //fptrs.G = computeFVarData(f, fvarwidth, fptrs.G, /*isAdaptive=*/true);
                } else {

                    // Gregory Boundary Patch (4 CVs + quad-offsets / valence tables)
                    for (int j=0; j<4; ++j)
                        iptrs.GB[j] = f->GetVertex(j)->GetID();
                    iptrs.GB+=4;
                    getQuadOffsets(f, quad_G_C1_P);
                    quad_G_C1_P += 4;
                    pptrs.GB = computePatchParam(f, pptrs.GB);
                    //fptrs.GB = computeFVarData(f, fvarwidth, fptrs.GB, /*isAdaptive=*/true);
                }
            } else {
                // XXXX manuelk - end patches here
            }
        } else {

            // Transition patches

            int pattern = f->_adaptiveFlags.transitionType;
            assert( pattern>=Hface::kTransition0 and pattern<=Hface::kTransition4 );
            ++pattern;  // TransitionPattern begin with NON_TRANSITION

            if (not f->_adaptiveFlags.isExtraordinary and f->_adaptiveFlags.bverts!=1) {

                switch (f->_adaptiveFlags.bverts) {
                    case 0 : {   // Regular Transition Patch (16 CVs)
                                 iptrs.R[pattern] = getOneRing(f, 16, remapRegular, iptrs.R[pattern]);
                                 pptrs.R[pattern] = computePatchParam(f, pptrs.R[pattern]);
                                 //fptrs.R[pattern] = computeFVarData(f, fvarwidth, fptrs.R[pattern], /*isAdaptive=*/true);
                             } break;

                    case 2 : {   // Boundary Transition Patch (12 CVs)
                                 unsigned rot = f->_adaptiveFlags.brots;
                                 iptrs.B[pattern][rot] = getOneRing(f, 12, remapRegularBoundary, iptrs.B[pattern][rot]);
                                 pptrs.B[pattern][rot] = computePatchParam(f, pptrs.B[pattern][rot]);
                                 //fptrs.B[pattern][rot] = computeFVarData(f, fvarwidth, fptrs.B[pattern][rot], /*isAdaptive=*/true);
                             } break;

                    case 3 : {   // Corner Transition Patch (9 CVs)
                                 unsigned rot = f->_adaptiveFlags.brots;
                                 iptrs.C[pattern][rot] = getOneRing(f, 9, remapRegularCorner, iptrs.C[pattern][rot]);
                                 pptrs.C[pattern][rot] = computePatchParam(f, pptrs.C[pattern][rot]);
                                 //fptrs.C[pattern][rot] = computeFVarData(f, fvarwidth, fptrs.C[pattern][rot], /*isAdaptive=*/true);
                             } break;
                }
            } else
                // No transition Gregory patches
                assert(false);
        }
    }
    
    // Build Gregory patches vertex valence indices table
    if ((patchCtr.G > 0) or (patchCtr.GB > 0)) {

        // MAX_VALENCE is a property of hardware shaders and needs to be matched in OSD
        const int perVertexValenceSize = 2*maxvalence + 1;

        const int nverts = mesh.GetNumVertices();

        Far::PatchTables::VertexValenceTable & table = result->_vertexValenceTable;
        table.resize(nverts * perVertexValenceSize);

        class GatherNeighborsOperator : public OpenSubdiv::HbrVertexOperator<Vertex> {
        public:
            Hvertex * center;
            Far::PatchTables::VertexValenceTable & table;
            int offset, valence;

            GatherNeighborsOperator(Far::PatchTables::VertexValenceTable & itable, int ioffset, Hvertex * v) :
                center(v), table(itable), offset(ioffset), valence(0) { }

            ~GatherNeighborsOperator() { }

            // Operator iterates over neighbor vertices of v and accumulates
            // pairs of indices the neighbor and diagonal vertices
            //
            //          Regular case
            //                                           Boundary case
            //      o ------- o      D3 o
            //   D0        N0 |         |
            //                |         |             o ------- o      D2 o
            //                |         |          D0        N0 |         |
            //                |         |                       |         |
            //      o ------- o ------- o                       |         |
            //   N1 |       V |      N3                         |         |
            //      |         |                       o ------- o ------- o
            //      |         |                    N1          V       N2
            //      |         |
            //      o         o ------- o
            //   D1         N2        D2
            //
            virtual void operator() (Hvertex &v) {

                table[offset++] = v.GetID();

                Hvertex * diagonal=&v;

                Hhalfedge * e = center->GetEdge(&v);
                if ( e ) {
                    // If v is on a boundary, there may not be a diagonal vertex
                    diagonal = e->GetNext()->GetDestVertex();
                }
                //else {
                //    diagonal = v.GetQEONext( center );
                //}

                table[offset++] = diagonal->GetID();

                ++valence;
            }
        };

        for (int i=0; i<nverts; ++i) {
            Hvertex * v = mesh.GetVertex(i);

            int outputVertexID = v->GetID();
            int offset = outputVertexID * perVertexValenceSize;

            // feature adaptive refinement can generate un-connected face-vertices
            // that have a valence of 0
            if (not v->IsConnected()) {
                //assert( v->GetParentFace() );
                table[offset] = 0;
                continue;
            }

            // "offset+1" : the first table entry is the vertex valence, which
            // is gathered by the operator (see note below)
            GatherNeighborsOperator op( table, offset+1, v );
            v->ApplyOperatorSurroundingVertices( op );

            // Valence sign bit used to mark boundary vertices
            table[offset] = v->OnBoundary() ? -op.valence : op.valence;

            // Note : some topologies can cause v to be singular at certain
            // levels of adaptive refinement, which prevents us from using
            // the GetValence() function. Fortunately, the GatherNeighbors
            // operator above just performed a similar traversal, so it is
            // very convenient to use it to accumulate the actionable valence.
        }
    } else {
        result->_vertexValenceTable.clear();
    }

    return result;
}

//------------------------------------------------------------------------------
// The One Ring vertices to rule them all !
Far::Index *
Far::PatchTablesFactory::getOneRing(Hface const * f,
    int ringsize, Far::Index const * remap, Far::Index * result) {

    assert( f and f->GetNumVertices()==4 and ringsize >=4 );

    int idx=0;

    for (unsigned char i=0; i<4; ++i) {
        result[remap[idx++ % ringsize]] =
            f->GetVertex( (i+f->_adaptiveFlags.rots)%4 )->GetID();
    }

    if (ringsize==16) {

        // Regular case
        //
        //       |      |      |      |
        //       | 4    | 15   | 14   | 13
        //  ---- o ---- o ---- o ---- o ----
        //       |      |      |      |
        //       | 5    | 0    | 3    | 12
        //  ---- o ---- o ---- o ---- o ----
        //       |      |      |      |
        //       | 6    | 1    | 2    | 11
        //  ---- o ---- o ---- o ---- o ----
        //       |      |      |      |
        //       | 7    | 8    | 9    | 10
        //  ---- o ---- o ---- o ---- o ----
        //       |      |      |      |
        //       |      |      |      |

        for (int i=0; i<4; ++i) {
            int rot = i+f->_adaptiveFlags.rots;
            Hvertex * v0 = f->GetVertex( rot % 4 ),
                                     * v1 = f->GetVertex( (rot+1) % 4 );

            Hhalfedge * e =
                v0->GetNextEdge( v0->GetNextEdge( v0->GetEdge(v1) ) );

            for (int j=0; j<3; ++j) {
                e = e->GetNext();
                result[remap[idx++ % ringsize]] = e->GetOrgVertex()->GetID();
            }
        }

        result += 16;

    } else if (ringsize==12) {

        // Boundary case
        //
        //         4      0      3      5
        //  ---- o ---- o ---- o ---- o ----
        //       |      |      |      |
        //       | 11   | 1    | 2    | 6
        //  ---- o ---- o ---- o ---- o ----
        //       |      |      |      |
        //       | 10   | 9    | 8    | 7
        //  ---- o ---- o ---- o ---- o ----
        //       |      |      |      |
        //       |      |      |      |

        Hvertex * v[4];
        for (int i=0; i<4; ++i)
            v[i] = f->GetVertex( (i+f->_adaptiveFlags.rots)%4 );

        Hhalfedge * e;

        e = v[0]->GetIncidentEdge()->GetPrev()->GetOpposite()->GetPrev();
        result[remap[idx++ % ringsize]] = e->GetOrgVertex()->GetID();

        e = v[1]->GetIncidentEdge();
        result[remap[idx++ % ringsize]] = e->GetDestVertex()->GetID();

        e = v[2]->GetNextEdge( v[2]->GetEdge(v[1]) );
        for (int i=0; i<3; ++i) {
            e = e->GetNext();
            result[remap[idx++ % ringsize]] = e->GetOrgVertex()->GetID();
        }

        e = v[3]->GetNextEdge( v[3]->GetEdge(v[2]) );
        for (int i=0; i<3; ++i) {
            e = e->GetNext();
            result[remap[idx++ % ringsize]] = e->GetOrgVertex()->GetID();
        }

        result += 12;

    } else if (ringsize==9) {

        // Corner case
        //
        //     0      1      4
        //   o ---- o ---- o ----
        //   |      |      |
        //   | 3    | 2    | 5
        //   o ---- o ---- o ----
        //   |      |      |
        //   | 8    | 7    | 6
        //   o ---- o ---- o ----
        //   |      |      |
        //   |      |      |

        Hvertex * v0 = f->GetVertex( (0+f->_adaptiveFlags.rots)%4 ),
                                 * v2 = f->GetVertex( (2+f->_adaptiveFlags.rots)%4 ),
                                 * v3 = f->GetVertex( (3+f->_adaptiveFlags.rots)%4 );

        Hhalfedge * e;

        e = v0->GetIncidentEdge()->GetPrev()->GetOpposite()->GetPrev();
        result[remap[idx++ % ringsize]] = e->GetOrgVertex()->GetID();

        e = v2->GetIncidentEdge();
        result[remap[idx++ % ringsize]] = e->GetDestVertex()->GetID();

        e = v3->GetNextEdge( v3->GetEdge(v2) );
        for (int i=0; i<3; ++i) {
            e = e->GetNext();
            result[remap[idx++ % ringsize]] = e->GetOrgVertex()->GetID();
        }

        result += 9;

    }
    assert(idx==ringsize);
    return result;
}

//------------------------------------------------------------------------------
// Populate the quad-offsets table used by Gregory patches
void
Far::PatchTablesFactory::getQuadOffsets(Hface const * f, unsigned int * result) {

    assert(result and f and f->GetNumVertices()==4);

    // Builds a table of value pairs for each vertex of the patch.
    //
    //            o
    //         N0 |
    //            |
    //            |
    //   o ------ o ------ o
    // N1       V | .... M3
    //            | .......
    //            | .......
    //            o .......
    //          N2
    //
    // [...] [N2 - N3] [...]
    //
    // Each value pair is composed of 2 index values in range [0-4[ pointing
    // to the 2 neighbor vertices to the vertex that belong to the Gregory patch.
    // Neighbor ordering is valence counter-clockwise and must match the winding
    // used to build the vertexValenceTable.
    //

    class GatherOffsetsOperator : public OpenSubdiv::HbrVertexOperator<Vertex> {
    public:
        Hvertex ** verts; int offsets[2]; int index; int count;

        GatherOffsetsOperator(Hvertex ** iverts) : verts(iverts) { }

        ~GatherOffsetsOperator() { }

        void reset() {
            index=count=offsets[0]=offsets[1]=0;
        }

        virtual void operator() (Hvertex &v) {
             // Resolve which 2 neighbor vertices of v belong to the Gregory patch
             for (unsigned char i=0; i<4; ++i)
                if (&v==verts[i]) {
                    assert(count<3);
                    offsets[count++]=index;
                    break;
                }
            ++index;
        }
    };

    // 4 central CVs of the Gregory patch
    Hvertex * fvs[4] = { f->GetVertex(0),
                                          f->GetVertex(1),
                                          f->GetVertex(2),
                                          f->GetVertex(3) };


    // Hbr vertex operator that iterates over neighbor vertices
    GatherOffsetsOperator op( fvs );

    for (unsigned char i=0; i<4; ++i) {

        op.reset();

        fvs[i]->ApplyOperatorSurroundingVertices( op );

        if (op.offsets[1] - op.offsets[0] != 1)
            std::swap(op.offsets[0], op.offsets[1]);

        // Pack the 2 indices in 16 bits
        result[i] = (op.offsets[0] | (op.offsets[1] << 8));
    }
}

//------------------------------------------------------------------------------
// Computes per-face or per-patch local ptex texture coordinates.
OpenSubdiv::Far::PatchParam *
Far::PatchTablesFactory::computePatchParam(Hface const * f, OpenSubdiv::Far::PatchParam *coord) {

    unsigned short u, v, ofs = 1;
    unsigned char depth;
    bool nonquad = false;

    if (coord == NULL) return NULL;

    // save the rotation state of the coarse face
    unsigned char rots = (unsigned char)f->_adaptiveFlags.rots;

    // track upwards towards coarse parent face, accumulating u,v indices
    Hface const * p = f->GetParent();
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
        ofs = (unsigned short)(ofs << 1);
        f = p;
        p = f->GetParent();
    }

    coord->Set( f->GetPtexIndex(), u, v, rots, depth, nonquad );

    return ++coord;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv


OpenSubdiv::Far::PatchTables const *
CreatePatchTables(Hmesh & mesh, int maxvalence) {

    return OpenSubdiv::Far::PatchTablesFactory::Create(mesh, maxvalence);
}

//------------------------------------------------------------------------------
