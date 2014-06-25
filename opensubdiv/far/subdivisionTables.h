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

#ifndef FAR_SUBDIVISION_TABLES_H
#define FAR_SUBDIVISION_TABLES_H

#include "../version.h"

#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief FarSubdivisionTables are a serialized topological data representation.
///
/// Subdivision tables store the indexing tables required in order to compute
/// the refined positions of a mesh without the help of a hierarchical data
/// structure. The advantage of this representation is its ability to be executed
/// in a massively parallel environment without data dependencies.
///
/// The vertex indexing tables require the vertex buffer to be sorted based on the
/// nature of the parent of a given vertex : either a face, an edge, or a vertex.
/// (note : the Loop subdivision scheme does not create vertices as a child of a
/// face).
///
/// Each type of vertex in the buffer is associated the following tables :
/// - _<T>_IT : indices of all the adjacent vertices required by the compute kernels
/// - _<T>_W : fractional weight of the vertex (based on sharpness & topology)
/// - _<T>_ITa : codex for the two previous tables
/// (where T denotes a face-vertex / edge-vertex / vertex-vertex)
///
///
/// Because each subdivision scheme (Catmark / Loop / Bilinear) introduces variations
/// in the subdivision rules, a derived class specialization is associated with
/// each scheme.
///
/// For more details see : "Feature Adaptive GPU Rendering of Catmull-Clark
/// Subdivision Surfaces"  (p.3 - par. 3.2)
///
class FarSubdivisionTables {

public:

    enum Scheme {
        UNDEFINED=0,
        BILINEAR,
        CATMARK,
        LOOP
    };

    enum TableType {
        E_IT,  ///< edge-vertices adjacency indexing table
        E_W,   ///< edge-vertices weights
        
        V_ITa, ///< vertex-vertices adjacency indexing table
        V_IT,  ///< vertex-vertices indexing table
        V_W,   ///< vertex-vertices weights
        
        F_ITa, ///< face-vertices adjacency indexing table
        F_IT,  ///< face-vertices indexing table
        
        TABLE_TYPES_COUNT  // number of different types of tables
    };

    /// \brief Destructor
    ~FarSubdivisionTables() {}

    /// \brief Return the highest level of subdivision possible with these tables
    int GetMaxLevel() const { return (int)(_vertsOffsets.size()-1); }

    /// \brief Memory required to store the indexing tables
    int GetMemoryUsed() const;

    /// \brief The index of the first vertex that belongs to the level of subdivision
    /// represented by this set of FarSubdivisionTables
    int GetFirstVertexOffset( int level ) const;

    /// \brief Returns the total number of vertex adressed by the tables (this is the
    /// length that a vertex buffer object should be allocating
    int GetNumVertices( ) const;

    /// \brief Returns the number of vertices at a given level
    int GetNumVertices( int level ) const;

    /// \brief Returns the summation of the number of vertices up to a given level
    int GetNumVerticesTotal( int level ) const;

    // Indexing tables accessors

    /// \brief Returns the face vertices codex table
    std::vector<int> const &          Get_F_ITa( ) const { return _F_ITa; }

    /// \brief Returns the face vertices indexing table
    std::vector<unsigned int> const & Get_F_IT( ) const { return _F_IT; }

    /// \brief Returns the edge vertices indexing table
    std::vector<int> const &          Get_E_IT() const { return _E_IT; }

    /// \brief Returns the edge vertices weights table
    std::vector<float> const &        Get_E_W() const { return _E_W; }

    /// \brief Returns the vertex vertices codex table
    std::vector<int> const &          Get_V_ITa() const { return _V_ITa; }

    /// \brief Returns the vertex vertices indexing table
    std::vector<unsigned int> const & Get_V_IT() const { return _V_IT; }

    /// \brief Returns the vertex vertices weights table
    std::vector<float> const &        Get_V_W() const { return _V_W; }

    /// \brief  Returns the subdivision scheme of the tables 
    /// (sidesteps typeinfo dependency)
    Scheme GetScheme() const { return _scheme; }

    /// \brief Returns the number of indexing tables needed to represent this particular
    /// subdivision scheme.
    int GetNumTables() const;

    // -------------------------------------------------------------------------
    // Bilinear scheme

    // Compute-kernel applied to vertices resulting from the refinement of a face.
    template <class U>
    void computeBilinearFacePoints(int vertexOffset, int tableOffset, int start, int end, U *vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of an edge.
    template <class U>
    void computeBilinearEdgePoints(int vertexOffset, int tableOffset, int start, int end, U *vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    template <class U>
    void computeBilinearVertexPoints(int vertexOffset, int tableOffset, int start, int end, U *vsrc) const;


    // -------------------------------------------------------------------------
    // Catmark scheme

    // Compute-kernel applied to vertices resulting from the refinement of a face.
    template <class U>
    void computeCatmarkFacePoints(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a quad face.
    template <class U>
    void computeCatmarkQuadFacePoints(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a tri or quad face.
    template <class U>
    void computeCatmarkTriQuadFacePoints(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of an edge.
    template <class U>
    void computeCatmarkEdgePoints(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a smooth or sharp edge.
    template <class U>
    void computeCatmarkRestrictedEdgePoints(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "A" Handles the k_Crease and k_Corner rules
    template <class U>
    void computeCatmarkVertexPointsA(int vertexOffset, bool pass, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "B" Handles the k_Smooth and k_Dart rules
    template <class U>
    void computeCatmarkVertexPointsB(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a smooth or sharp vertex
    // Kernel "A" handles the k_Crease and k_Corner rules
    template <class U>
    void computeCatmarkRestrictedVertexPointsA(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a smooth or sharp vertex
    // Kernel "B1" handles the regular k_Smooth and k_Dart rules
    template <class U>
    void computeCatmarkRestrictedVertexPointsB1(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a smooth or sharp vertex
    // Kernel "B2" handles the irregular k_Smooth and k_Dart rules
    template <class U>
    void computeCatmarkRestrictedVertexPointsB2(int vertexOffset, int tableOffset, int start, int end, U * vsrc) const;


    // -------------------------------------------------------------------------
    // Loop scheme

    // Compute-kernel applied to vertices resulting from the refinement of an edge.
    template <class U>
    void computeLoopEdgePoints(int offset, int level, int start, int end, U *vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "A" Handles the k_Smooth and k_Dart rules
    template <class U>
    void computeLoopVertexPointsA(int offset, bool pass, int level, int start, int end, U *vsrc) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "B" Handles the k_Crease and k_Corner rules
    template <class U>
    void computeLoopVertexPointsB(int offset,int level, int start, int end, U *vsrc) const;

protected:
    template <class X, class Y> friend class FarBilinearSubdivisionTablesFactory;
    template <class X, class Y> friend class FarCatmarkSubdivisionTablesFactory;
    template <class X, class Y> friend class FarLoopSubdivisionTablesFactory;
    template <class X, class Y> friend class FarSubdivisionTablesFactory;

    FarSubdivisionTables( int maxlevel, Scheme scheme );

    std::vector<int>          _F_ITa; // vertices from face refinement
    std::vector<unsigned int> _F_IT;  // indices of face vertices

    std::vector<int>          _E_IT;  // vertices from edge refinement
    std::vector<float>        _E_W;   // weigths

    std::vector<int>          _V_ITa; // vertices from vertex refinement
    std::vector<unsigned int> _V_IT;  // indices of adjacent vertices
    std::vector<float>        _V_W;   // weights

    std::vector<int> _vertsOffsets;   // offset to the first vertex of each level

    Scheme _scheme;                   // subdivision scheme
};

inline
FarSubdivisionTables::FarSubdivisionTables( int maxlevel, Scheme scheme ) :
    _vertsOffsets(maxlevel+2, 0), _scheme(scheme)
{
    assert( maxlevel > 0 );
}

inline int
FarSubdivisionTables::GetFirstVertexOffset( int level ) const {
    assert(level>=0 and level<(int)_vertsOffsets.size());
    return _vertsOffsets[level];
}

inline int
FarSubdivisionTables::GetNumVertices( ) const {
    if (_vertsOffsets.empty()) {
        return 0;
    } else {
        // _vertsOffsets contains an extra offset at the end that is the position
        // of the first vertex 1 level above that of the tables
        return *_vertsOffsets.rbegin();
    }
}

inline int
FarSubdivisionTables::GetNumVertices( int level ) const {
    assert(level>=0 and level<((int)_vertsOffsets.size()-1));
    return _vertsOffsets[level+1] - _vertsOffsets[level];
}

inline int
FarSubdivisionTables::GetNumVerticesTotal( int level ) const {
    assert(level>=0 and level<((int)_vertsOffsets.size()-1));
    return _vertsOffsets[level+1];
}

inline int
FarSubdivisionTables::GetNumTables() const {
    switch (_scheme) {
        case BILINEAR: return 7;
        case CATMARK: return 7;
        case LOOP: return 5;
        default: return 0;
    }
}

inline int
FarSubdivisionTables::GetMemoryUsed() const {
    return (int)(_F_ITa.size() * sizeof(int) +
                 _F_IT.size() * sizeof(unsigned int) +
                 _E_IT.size() * sizeof(int) +
                 _E_W.size() * sizeof(float) +
                 _V_ITa.size() * sizeof(int) +
                 _V_IT.size() * sizeof(unsigned int) +
                 _V_W.size() * sizeof(float));
}


//
// Face-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeBilinearFacePoints( int offset, int tableOffset, int start, int end, U *vsrc ) const {

    U * vdst = vsrc + offset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int h = this->_F_ITa[2*i  ],
            n = this->_F_ITa[2*i+1];
        float weight = 1.0f/n;

        for (int j=0; j<n; ++j) {
             vdst->AddWithWeight( vsrc[ this->_F_IT[h+j] ], weight );
             vdst->AddVaryingWithWeight( vsrc[ this->_F_IT[h+j] ], weight );
        }
    }
}

//
// Edge-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeBilinearEdgePoints( int offset,  int tableOffset, int start, int end, U *vsrc ) const {

    U * vdst = vsrc + offset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int eidx0 = this->_E_IT[2*i+0],
            eidx1 = this->_E_IT[2*i+1];

        vdst->AddWithWeight( vsrc[eidx0], 0.5f );
        vdst->AddWithWeight( vsrc[eidx1], 0.5f );

        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f );
    }
}

//
// Vertex-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeBilinearVertexPoints( int offset, int tableOffset, int start, int end, U *vsrc ) const {

    U * vdst = vsrc + offset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int p = this->_V_ITa[i];   // index of the parent vertex

        vdst->AddWithWeight( vsrc[p], 1.0f );
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

//
// Face-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeCatmarkFacePoints( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int h = this->_F_ITa[2*i  ],
            n = this->_F_ITa[2*i+1];
        float weight = 1.0f/n;

        for (int j=0; j<n; ++j) {
             vdst->AddWithWeight( vsrc[ this->_F_IT[h+j] ], weight );
             vdst->AddVaryingWithWeight( vsrc[ this->_F_IT[h+j] ], weight );
        }
    }
}

//
// Quad face-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeCatmarkQuadFacePoints( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start; i<end; ++i, ++vdst ) {
        int fidx0 = _F_IT[tableOffset + 4 * i + 0];
        int fidx1 = _F_IT[tableOffset + 4 * i + 1];
        int fidx2 = _F_IT[tableOffset + 4 * i + 2];
        int fidx3 = _F_IT[tableOffset + 4 * i + 3];

        vdst->Clear();
        vdst->AddWithWeight(vsrc[fidx0], 0.25f);
        vdst->AddVaryingWithWeight(vsrc[fidx0], 0.25f);
        vdst->AddWithWeight(vsrc[fidx1], 0.25f);
        vdst->AddVaryingWithWeight(vsrc[fidx1], 0.25f);
        vdst->AddWithWeight(vsrc[fidx2], 0.25f);
        vdst->AddVaryingWithWeight(vsrc[fidx2], 0.25f);
        vdst->AddWithWeight(vsrc[fidx3], 0.25f);
        vdst->AddVaryingWithWeight(vsrc[fidx3], 0.25f);
    }
}

//
// Tri/quad face-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeCatmarkTriQuadFacePoints( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start; i<end; ++i, ++vdst ) {
        int fidx0 = _F_IT[tableOffset + 4 * i + 0];
        int fidx1 = _F_IT[tableOffset + 4 * i + 1];
        int fidx2 = _F_IT[tableOffset + 4 * i + 2];
        int fidx3 = _F_IT[tableOffset + 4 * i + 3];

        bool isTriangle = (fidx3 == fidx2);
        float weight = isTriangle ? 1.0f / 3.0f : 1.0f / 4.0f;

        vdst->Clear();
        vdst->AddWithWeight(vsrc[fidx0], weight);
        vdst->AddVaryingWithWeight(vsrc[fidx0], weight);
        vdst->AddWithWeight(vsrc[fidx1], weight);
        vdst->AddVaryingWithWeight(vsrc[fidx1], weight);
        vdst->AddWithWeight(vsrc[fidx2], weight);
        vdst->AddVaryingWithWeight(vsrc[fidx2], weight);
        if (not isTriangle) {
            vdst->AddWithWeight(vsrc[fidx3], weight);
            vdst->AddVaryingWithWeight(vsrc[fidx3], weight);
        }
    }
}

//
// Edge-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeCatmarkEdgePoints( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int eidx0 = this->_E_IT[4*i+0],
            eidx1 = this->_E_IT[4*i+1],
            eidx2 = this->_E_IT[4*i+2],
            eidx3 = this->_E_IT[4*i+3];

        float vertWeight = this->_E_W[i*2+0];

        // Fully sharp edge : vertWeight = 0.5f
        vdst->AddWithWeight( vsrc[eidx0], vertWeight );
        vdst->AddWithWeight( vsrc[eidx1], vertWeight );

        if (eidx2!=-1) {
            // Apply fractional sharpness
            float faceWeight = this->_E_W[i*2+1];

            vdst->AddWithWeight( vsrc[eidx2], faceWeight );
            vdst->AddWithWeight( vsrc[eidx3], faceWeight );
        }

        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f );
    }
}

//
// Restricted edge-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeCatmarkRestrictedEdgePoints( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int eidx0 = this->_E_IT[4*i+0],
            eidx1 = this->_E_IT[4*i+1],
            eidx2 = this->_E_IT[4*i+2],
            eidx3 = this->_E_IT[4*i+3];

        vdst->AddWithWeight( vsrc[eidx0], 0.25f );
        vdst->AddWithWeight( vsrc[eidx1], 0.25f );
        vdst->AddWithWeight( vsrc[eidx2], 0.25f );
        vdst->AddWithWeight( vsrc[eidx3], 0.25f );
        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f );
    }
}

//
// Vertex-vertices compute Kernels "A" and "B" - completely re-entrant
//

// multi-pass kernel handling k_Crease and k_Corner rules
template <class U> void
FarSubdivisionTables::computeCatmarkVertexPointsA( int vertexOffset, bool pass, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        if (not pass)
            vdst->Clear();

        int     n=this->_V_ITa[5*i+1],   // number of vertices in the _V_IT array (valence)
                p=this->_V_ITa[5*i+2],   // index of the parent vertex
            eidx0=this->_V_ITa[5*i+3],   // index of the first crease rule edge
            eidx1=this->_V_ITa[5*i+4];   // index of the second crease rule edge

        float weight = pass ? this->_V_W[i] : 1.0f - this->_V_W[i];

        // In the case of fractional weight, the weight must be inverted since
        // the value is shared with the k_Smooth kernel (statistically the
        // k_Smooth kernel runs much more often than this one)
        if (weight>0.0f and weight<1.0f and n>0)
            weight=1.0f-weight;

        // In the case of a k_Corner / k_Crease combination, the edge indices
        // won't be null,  so we use a -1 valence to detect that particular case
        if (eidx0==-1 or (pass==false and (n==-1)) ) {
            // k_Corner case
            vdst->AddWithWeight( vsrc[p], weight );
        } else {
            // k_Crease case
            vdst->AddWithWeight( vsrc[p], weight * 0.75f );
            vdst->AddWithWeight( vsrc[eidx0], weight * 0.125f );
            vdst->AddWithWeight( vsrc[eidx1], weight * 0.125f );
        }
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

// multi-pass kernel handling k_Dart and k_Smooth rules
template <class U> void
FarSubdivisionTables::computeCatmarkVertexPointsB( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int h = this->_V_ITa[5*i  ],     // offset of the vertices in the _V_IT array
            n = this->_V_ITa[5*i+1],     // number of vertices in the _V_IT array (valence)
            p = this->_V_ITa[5*i+2];     // index of the parent vertex

        float weight = this->_V_W[i],
                  wp = 1.0f/(n*n),
                  wv = (n-2.0f)*n*wp;

        vdst->AddWithWeight( vsrc[p], weight * wv );

        for (int j=0; j<n; ++j) {
            vdst->AddWithWeight( vsrc[this->_V_IT[h+j*2  ]], weight * wp );
            vdst->AddWithWeight( vsrc[this->_V_IT[h+j*2+1]], weight * wp );
        }
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

// single-pass kernel handling k_Crease and k_Corner rules
template <class U> void
FarSubdivisionTables::computeCatmarkRestrictedVertexPointsA( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int     p=this->_V_ITa[5*i+2],   // index of the parent vertex
            eidx0=this->_V_ITa[5*i+3],   // index of the first crease rule edge
            eidx1=this->_V_ITa[5*i+4];   // index of the second crease rule edge

        vdst->AddWithWeight( vsrc[p], 0.75f );
        vdst->AddWithWeight( vsrc[eidx0], 0.125f );
        vdst->AddWithWeight( vsrc[eidx1], 0.125f );
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

// single-pass kernel handling regular k_Smooth and k_Dart rules
template <class U> void
FarSubdivisionTables::computeCatmarkRestrictedVertexPointsB1( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int h = this->_V_ITa[5*i  ],     // offset of the vertices in the _V_IT array
            p = this->_V_ITa[5*i+2];     // index of the parent vertex

        vdst->AddWithWeight( vsrc[p], 0.5f );

        for (int j=0; j<8; ++j, ++h)
            vdst->AddWithWeight( vsrc[this->_V_IT[h]], 0.0625f );

        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

// single-pass kernel handling irregular k_Smooth and k_Dart rules
template <class U> void
FarSubdivisionTables::computeCatmarkRestrictedVertexPointsB2( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int h = this->_V_ITa[5*i  ],     // offset of the vertices in the _V_IT array
            n = this->_V_ITa[5*i+1],     // number of vertices in the _V_IT array (valence)
            p = this->_V_ITa[5*i+2];     // index of the parent vertex

        float wp = 1.0f/(n*n),
              wv = (n-2.0f)*n*wp;

        vdst->AddWithWeight( vsrc[p], wv );

        for (int j=0; j<n; ++j) {
            vdst->AddWithWeight( vsrc[this->_V_IT[h+j*2  ]], wp );
            vdst->AddWithWeight( vsrc[this->_V_IT[h+j*2+1]], wp );
        }
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

//
// Edge-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarSubdivisionTables::computeLoopEdgePoints( int vertexOffset, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int eidx0 = this->_E_IT[4*i+0],
            eidx1 = this->_E_IT[4*i+1],
            eidx2 = this->_E_IT[4*i+2],
            eidx3 = this->_E_IT[4*i+3];

        float endPtWeight = this->_E_W[i*2+0];

        // Fully sharp edge : endPtWeight = 0.5f
        vdst->AddWithWeight( vsrc[eidx0], endPtWeight );
        vdst->AddWithWeight( vsrc[eidx1], endPtWeight );

        if (eidx2!=-1) {
            // Apply fractional sharpness
            float oppPtWeight = this->_E_W[i*2+1];

            vdst->AddWithWeight( vsrc[eidx2], oppPtWeight );
            vdst->AddWithWeight( vsrc[eidx3], oppPtWeight );
        }

        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f );
    }
}

//
// Vertex-vertices compute Kernels "A" and "B" - completely re-entrant
//

// multi-pass kernel handling k_Crease and k_Corner rules
template <class U> void
FarSubdivisionTables::computeLoopVertexPointsA( int vertexOffset, bool pass, int tableOffset, int start, int end, U * vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        if (not pass)
            vdst->Clear();

        int     n=this->_V_ITa[5*i+1], // number of vertices in the _V_IT array (valence)
                p=this->_V_ITa[5*i+2], // index of the parent vertex
            eidx0=this->_V_ITa[5*i+3], // index of the first crease rule edge
            eidx1=this->_V_ITa[5*i+4]; // index of the second crease rule edge

        float weight = pass ? this->_V_W[i] : 1.0f - this->_V_W[i];

        // In the case of fractional weight, the weight must be inverted since
        // the value is shared with the k_Smooth kernel (statistically the
        // k_Smooth kernel runs much more often than this one)
        if (weight>0.0f and weight<1.0f and n>0)
            weight=1.0f-weight;

        // In the case of a k_Corner / k_Crease combination, the edge indices
        // won't be null,  so we use a -1 valence to detect that particular case
        if (eidx0==-1 or (pass==false and (n==-1)) ) {
            // k_Corner case
            vdst->AddWithWeight( vsrc[p], weight );
        } else {
            // k_Crease case
            vdst->AddWithWeight( vsrc[p], weight * 0.75f );
            vdst->AddWithWeight( vsrc[eidx0], weight * 0.125f );
            vdst->AddWithWeight( vsrc[eidx1], weight * 0.125f );
        }
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

// multi-pass kernel handling k_Dart and k_Smooth rules
template <class U> void
FarSubdivisionTables::computeLoopVertexPointsB( int vertexOffset, int tableOffset, int start, int end, U *vsrc ) const {

    U * vdst = vsrc + vertexOffset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear();

        int h = this->_V_ITa[5*i  ], // offset of the vertices in the _V_IT array
            n = this->_V_ITa[5*i+1], // number of vertices in the _V_IT array (valence)
            p = this->_V_ITa[5*i+2]; // index of the parent vertex

        float weight = this->_V_W[i],
                  wp = 1.0f/n,
                beta = 0.25f * cosf((float)M_PI * 2.0f * wp) + 0.375f;
        beta = beta*beta;
        beta = (0.625f-beta)*wp;

        vdst->AddWithWeight( vsrc[p], weight * (1.0f-(beta*n)));

        for (int j=0; j<n; ++j)
            vdst->AddWithWeight( vsrc[this->_V_IT[h+j]], weight * beta );

        vdst->AddVaryingWithWeight( vsrc[p], 1.0f );
    }
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_SUBDIVISION_TABLES_H */
