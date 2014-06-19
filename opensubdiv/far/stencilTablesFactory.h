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

#ifndef FAR_STENCILTABLE_FACTORY_H
#define FAR_STENCILTABLE_FACTORY_H

#include "../version.h"

#include "../hbr/allocator.h"
#include "../hbr/mesh.h"
#include "../hbr/catmark.h"

#include "../far/stencilTables.h"

#include <string.h>
#include <list>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class FarVertexStencil;

class FarStencilFactoryVertex;

/// \brief A factory for FarStencilTables
///
/// The FarStencilTablesFactory is used to generate FarStencilTables. Currently
/// this factory can only accumulate stencil tables for the Catmark subdivision
/// scheme.
///
/// Client-code must first select a face index in the source Hbr mesh (and
/// possibly a sub-face quadrant if the coarse face is not a quad). If the call
/// is successful, then an arbitrary number of stencils corresponding to (u,v)
/// sample locations can be computed using the AppendStencils method. These
/// stencils are self-contained and can be appended to any FarStencilTables.
///
/// Each stencil can be pushed to an selected level of feature isolation.
/// Surface approximation beyond this level of isolation is obtained by pushing
/// volatile vertices to their limits and bilinearly interpolating the resulting
/// quad sub-face containing the sample location (no Gregory patch approximation).
///
/// The current stencils accumulated are for limit positions only. Additional
/// methods to provide limit stencils on vertex locations should be fairly
/// easy to implement.
///
/// \note Hierarchical edits are not supported yet (XXXX)
///
/// \note Face-varying boundaries are not implemented yet (XXXX)
///
/// \note The FarStencilTablesFactory processes are currently not re-entrant:
/// every invokation of SetCurrentFace causes the HbrMesh to be unrefined to
/// a default starting position.
///
template <class T=FarStencilFactoryVertex> class FarStencilTablesFactory {

public:

    /// \brief Constructor
    ///
    /// \note this factory *will* modify the state of the HbrMesh
    ///
    /// \note hierarchical vertex edits are not supported yet
    ///
    FarStencilTablesFactory( HbrMesh<T> * mesh );

    /// \brief Returns the HbrMesh used by the factory
    HbrMesh<T> const * GetMesh() {
        return _mesh;
    }

    /// \brief Sets the current Hbr face where stencils will be added
    ///
    /// @param id        A valid Hbr coarse face ID
    ///
    /// @param quadrant  If the face is extraordinary, specify which (u,v)
    ///                  coordinates quadrant (aka sub-face) to use.
    ///
    /// @return          True upon success
    ///
    bool SetCurrentFace( int id, unsigned int quadrant=0 );

    /// \brief Append stencils for the given UV's to the FarStencilTables
    ///
    /// @param stencilTables  The table of stencils to add the results to
    ///
    /// @param nsamples       The number of uv locations
    ///
    /// @param u              Array of u-paramater locations
    ///
    /// @param v              Array of v-paramater locations
    ///
    /// @param reflevel       Max level of feature isolation
    ///
    /// @return               The number of stencils added to the array
    ///
    int AppendStencils( FarStencilTables * stencilTables,
                        int nsamples,
                        float const * u,
                        float const * v,
                        int reflevel );

    /// \brief Returns the maximum valence of a vertex allowed in the coarse
    /// mesh topology. Higher valences will generate incorrect limit tangents.
    int GetMaxValenceSupported();

private:

    friend class FarVertexStencil;
    friend class FarStencilFactoryVertex;

    // Reserve space for stencils of a set size at the end of a stencil table
    void _AddNewStencils( FarStencilTables * tables, int nstencils, int stencilsize);

    HbrMesh<T> * _mesh;

    int _numCoarseVertices;

    // A private helper class that caches computations for the current face
    class Patch;

    Patch _patch; // Cache for the current face
};


// Constructor
template <class T>
FarStencilTablesFactory<T>::FarStencilTablesFactory( HbrMesh<T> * mesh )
    : _mesh(mesh) {

    _numCoarseVertices = mesh->GetNumVertices();
}

// Sets the current Hbr face where stencils will be added
template <class T> bool
FarStencilTablesFactory<T>::SetCurrentFace(int id, unsigned int quadrant) {

    assert(_mesh);

    HbrFace<T> * f = GetMesh()->GetFace(id);

    // XXXX Vertex edits are not supported yet
    if ((f and (not f->IsCoarse())) or GetMesh()->HasVertexEdits())
        return false;

    // Extraordinary faces don't have simple (u,v) parameterization so we need
    // to specify the parametric quadrant
    if (GetMesh()->GetSubdivision()->FaceIsExtraordinary(GetMesh(), f)) {

        if ( quadrant > (unsigned int)(f->GetNumVertices()-1) )
            return false;

    } else {
        // face is a quad: lock quadrant to 0
        quadrant = 0;
    }

    _mesh->Unrefine(_numCoarseVertices, GetMesh()->GetNumCoarseFaces());

    _patch.SetupControlStencils(f, quadrant);

    return true;
}


// Append stencils for the given UV's to the FarStencilTables
template <class T> int
FarStencilTablesFactory<T>::AppendStencils( FarStencilTables * stencilTables,
                                            int nsamples,
                                            float const * u,
                                            float const * v,
                                            int reflevel  ) {

    assert(stencilTables);

    int result=0;

    if ((not nsamples) or (not _patch.GetCurrentFace()))
        return result;

    int stencilsize = _patch.GetStencilSize();

    _AddNewStencils( stencilTables, nsamples, stencilsize );

    FarStencil stencil = stencilTables->GetStencil( stencilTables->GetNumStencils()-nsamples );

    std::vector<int> const & indices = _patch.GetControlVertexIndices();

    for (int i=0; i<nsamples; ++i) {

        // Copy control vertices indices
        for (int j=0; j<stencilsize; ++j)
            stencil._indices[j] = indices[j];

        HbrHalfedge<T> * edge =
            _patch.GetCurrentFace()->GetEdge(_patch.GetCurrentQuadrant());

        // Copy stencil weights for sample
        result += _patch.GetStencilsAtUV( edge, u[i], v[i], reflevel,
                                          stencil._point,
                                          stencil._uderiv,
                                          stencil._vderiv );

        // Advance to the next stencil
        stencil.Increment();
    }

    return result;
}

template <class T> void
FarStencilTablesFactory<T>::_AddNewStencils( FarStencilTables * tables,
                                             int nstencils, int stencilsize) {

    int size = tables->GetNumStencils();

    // Append sizes
    tables->_sizes.insert(tables->_sizes.end(), nstencils, stencilsize);

    // Append offsets
    tables->_offsets.resize(size+nstencils);

    int * offsPtr = &(tables->_offsets[size]),
                h = (int)tables->_point.size();

    for (int i=0; i<nstencils; ++i, h+=stencilsize)
        *offsPtr++ = h;

    // Extend data space for vertex indices and weights
    int newsize = (int)tables->_point.size() + (nstencils*stencilsize);
    tables->_indices.resize(newsize);
    tables->_point.resize(newsize);
    tables->_uderiv.resize(newsize);
    tables->_vderiv.resize(newsize);
}


class FarVertexStencilAllocator;

/// \brief A container for vertex stencil weight coefficients
///
class FarVertexStencil {
public:

    /// \brief Constructor
    FarVertexStencil() { }

    /// \brief Destructor
    ~FarVertexStencil() { }

    /// \brief Next available instance in the pool allocator
    FarVertexStencil * & GetNext() {
        return _next;
    }

    /// \brief Returns a pointer to the vertex stencils pool allocator
    FarVertexStencilAllocator * GetAllocator() const {
        return _allocator;
    }

    /// \brief Returns a pointer to the first stencil weight coefficient
    float const * GetData() const {
        return _data;
    }

    /// \brief Resets the values of the weight coefficients: dst = val
    ///
    /// @param dst          Pointer to the weight coefficients that will be reset
    ///
    /// @param val          Value the weights will be reset to
    ///
    /// @param stencilsize  Number of weight coefficients in the stencil
    ///
    static void Reset( float * dst, float val, int stencilsize );

    /// \brief Resets the values of the weight coefficients: this = val
    ///
    /// @param val          Value the weights will be reset to
    ///
    void Reset( float val );

    /// \brief Copies the values of the weight coefficients: dst = other
    ///
    /// @param dst          Pointer to the destination weight coefficients
    ///
    /// @param other        Source stencil to copy the weights from
    ///
    static void Copy( float * dst , FarVertexStencil const * other );

    /// \brief Adds the coefficients from a stencil
    ///
    /// @param dst          Pointer to the destination weight coefficients
    ///
    /// @param other        Source stencil to add the weights from
    ///
    static void Add( float * dst, FarVertexStencil const * other );

    /// \brief Subtracts the coefficients from a stencil dst = (a - b)
    ///
    /// @param dst          Pointer to the destination weight coefficients
    ///
    /// @param sa           'A' source stencil
    ///
    /// @param sb           'B' source stencil
    ///
    static void Subtract( float * dst, FarVertexStencil const * sa, FarVertexStencil const * sb );

    /// \brief Scales the coefficients from a stencil: dst *= val
    ///
    /// @param dst          Pointer to the destination weight coefficients
    ///
    /// @param val          Value the weights will be scaled with
    ///
    /// @param stencilsize  Number of weight coefficients in the stencil
    ///
    static void Scale( float * dst , float val, int stencilsize );

    /// \brief Adds and scales the coefficients of the stencil: dst += other * val
    ///
    /// @param dst          Pointer to the destination weight coefficients
    ///
    /// @param other        Stencil to be added
    ///
    /// @param val          Value to weigh 'other' with
    ///
    static void AddScaled( float * dst, FarVertexStencil const * other, float val );

    /// \brief Adds and scales the coefficients of the stencil: this += other * val
    ///
    /// @param other        Stencil to be added
    ///
    /// @param val          Value to weigh 'other' with
    ///
    void AddScaled( FarVertexStencil const & other, float val );

    /// \brief Combines coefficients: this = a*sa + b*sb
    ///
    /// @param a            'A' weight
    ///
    /// @param sa           'A' stencil
    ///
    /// @param b            'B' weight
    ///
    /// @param sb           'B' stencil
    ///
    void Combine( float a, FarVertexStencil const & sa, float b, FarVertexStencil const & sb );

    /// \brief Prints the weight coefficients
    void Print() const;

private:

    template <class T> friend class FarStencilTablesFactory<T>::Patch;

    friend class FarVertexStencilAllocator;

    float * _GetData() {
        return _data;
    }

    FarVertexStencilAllocator * _allocator;

    FarVertexStencil * _next;

    float _data[1];
};


/// \brief A pool allocator for FarVertexStencil
///
class FarVertexStencilAllocator {

public:

    FarVertexStencilAllocator(int stencilsize) :
        _stencilSize(stencilsize),
        _allocator(&_memory, 256, 0, 0, sizeof(FarVertexStencil)+(stencilsize-1)*sizeof(float)) {
        assert(stencilsize>=1);
    }

    // Returns the number of coefficients in the the stencil
    int GetStencilSize() const {
        return _stencilSize;
    }

    // Returns the total memory used by the allocator
    size_t GetMemoryUsed( ) const {
        return _memory;
    }

    // Allocates a FarVertexStencil from the pool
    FarVertexStencil * Allocate( ) {

        FarVertexStencil * result = _allocator.Allocate();

        result->_allocator = this;

        return result;
    }

    // Deallocates a FarVertexStencil from the pool
    void Deallocate( FarVertexStencil * v) {
        _allocator.Deallocate(v);
    }

private:

    int _stencilSize;

    size_t _memory;
    HbrAllocator<FarVertexStencil> _allocator;
};

inline void
FarVertexStencil::Reset( float val ) {
    Reset( _GetData(), val, GetAllocator()->GetStencilSize() );
}

inline void
FarVertexStencil::Reset( float * dst, float val, int stencilsize ) {
    float * end = dst + stencilsize;
    while (dst < end)
        (*dst++) = val;
}

inline void
FarVertexStencil::Copy( float * dst, FarVertexStencil const * other ) {
    assert(other and other->GetAllocator());
    memcpy( dst, other->GetData(), other->GetAllocator()->GetStencilSize()*sizeof(float) );
}

inline void
FarVertexStencil::Add( float * dst, FarVertexStencil const * other ) {
    assert(other and other->GetAllocator());
    float * end = dst + other->GetAllocator()->GetStencilSize();
    float const * src = other->GetData();
    while (dst < end)
        (*dst++) += (*src++);
}

inline void
FarVertexStencil::Subtract( float * dst, FarVertexStencil const * sa,
                                         FarVertexStencil const * sb ) {
    assert(sa and sb and sa->GetAllocator());
    float * end = dst + sa->GetAllocator()->GetStencilSize();
    float const * aptr = sa->GetData(),
                * bptr = sb->GetData();
    while (dst < end)
        (*dst++) = (*aptr++) - (*bptr++);
}

inline void
FarVertexStencil::Scale( float * dst , float val, int stencilsize ) {
    float * end = dst + stencilsize;
    while (dst < end)
        (*dst++) *= val;
}

inline void
FarVertexStencil::AddScaled( FarVertexStencil const & other, float val ) {
    AddScaled( _GetData(), &other, val );
}

inline void
FarVertexStencil::AddScaled( float * dst, FarVertexStencil const * other, float val ) {
    assert(other and other->GetAllocator());
    float * end = dst + other->GetAllocator()->GetStencilSize();
    float const * src = other->GetData();
    while (dst < end)
        (*dst++) += (*src++) * val;
}

inline void
FarVertexStencil::Combine( float a, FarVertexStencil const & sa,
                           float b, FarVertexStencil const & sb  ) {
    assert(GetAllocator());
    float * dst = _data,
          * end = dst + GetAllocator()->GetStencilSize();
    float const * aptr = sa.GetData(),
                * bptr = sb.GetData();
    while (dst < end)
        (*dst++) = a*(*aptr++) + b*(*bptr++);
}

inline void
FarVertexStencil::Print() const {
    printf("{ ");
    for (int i=0; i<_allocator->GetStencilSize(); ++i)
        printf("%.5f ", _data[i] );
    printf("}");
}

/// \brief A dedicated vertex class for the FarStencilTablesFactory
///
class FarStencilFactoryVertex {
public:

    /// \brief Constructor
    FarStencilFactoryVertex( ) : _stencil(0) { }

    FarStencilFactoryVertex(int) : _stencil(0) { }

    /// \brief Destructor
    ~FarStencilFactoryVertex( ) { }

    FarVertexStencil const * GetStencil() const {
        return _stencil;
    }

    /// \brief Hbr template vertex class API : resets the vertex
    void Clear() {
        if (_stencil)
            _stencil->Reset(0.0f);
    }

    /// \brief Hbr template vertex class API: interpolate vertex data with 'src'
    ///        vertex (this += src * weight)
    ///
    /// @param src     The source vertex
    ///
    /// @param weight  The interpolation weight
    ///
    void AddWithWeight(FarStencilFactoryVertex const & src, float weight, void * =0 ) {

        if (not _stencil) {

            // Allocation is lazy, but 'src' will have a valid stencil unless that
            // vertex is singular
            if (src.GetStencil()) {
                _stencil = src.GetStencil()->GetAllocator()->Allocate();
                _stencil->Reset(0.0f);
            } else
                return;
        }
        _stencil->AddScaled( *src.GetStencil(), weight );
    }

    /// \brief Hbr template vertex class API: interpolate varying data with 'src'
    ///        vertex (this += src * weight)
    ///
    void AddVaryingWithWeight(FarStencilFactoryVertex const & /* src */, float /* weight */, void * =0 ) { }

    /// \brief Hbr template vertex class API: edits are not supported yet
    ///
    void ApplyVertexEdit(OpenSubdiv::HbrVertexEdit<FarStencilFactoryVertex> const & /* edit */) { }

    /// \brief Hbr template vertex class API: edits are not supported yet
    ///
    void ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<FarStencilFactoryVertex> &) { }

private:

    template <class T> friend class FarStencilTablesFactory<T>::Patch;

    FarVertexStencil * _GetStencil() {
        return _stencil;
    }

    void _SetStencil( FarVertexStencil * stencil) {
        _stencil = stencil;
    }

    FarVertexStencil * _stencil;
};



// A private helper class that caches computations for the current face
template <class T>
class FarStencilTablesFactory<T>::Patch {

public:

    // Constructor
    Patch() : _face(0), _quadrant(0), _allocator(0) { }

    // Returns the current coarse face
    HbrFace<T> const * GetCurrentFace() const {
        return _face;
    }

    // Sub-quadrant of the current face if it's not a quad
    int GetCurrentQuadrant() const {
        return _quadrant;
    }

    // Returns the number of weights in the stencils for this face
    int GetStencilSize() const {
        return (int)_allocator->GetStencilSize();
    }

    // Return the indices of the control vertices in the HbrMesh
    std::vector<int> const & GetControlVertexIndices() const {
        return _vertIndices;
    }

    // Gathers all the coarse control vertices for an arbitrary patch
    void SetupControlStencils( HbrFace<T> * f, int quadrant );

    // Appends stencil weight coefficients for the given u & v
    bool GetStencilsAtUV( HbrHalfedge<T> * e,
                          float u,
                          float v,
                          int reflevel,
                          float *point,
                          float *deriv1,
                          float *deriv2 );

private:

    // True if the vertex has a BSpline limit
    static bool _IsaBSpline( HbrVertex<T> * v );

    // True if the edge has a BSpline limit
    static bool _IsaBSpline( HbrHalfedge<T> * e );

    // True if the face has a BSpline limit
    static bool _IsaBSpline( HbrFace<T> * f );

    // Computes the limit stencils
    void _GetLimitStencils( HbrVertex<T> * v,
                            float *point );

    // Computes derivative limit stencils
    void _GetTangentLimitStencils( HbrHalfedge<T> * e,
                                   float *uderiv,
                                   float *vderiv );

    // Computes BSpline weights
    static void _GetBSplineWeights( float t,
                                    float *cubicWeights,
                                    float *quadraticWeights );

    // Updates the cached BSpline stencils
    void _UpdateBSplineStencils( HbrFace<T> const * f );

    // Scale tangent stencils so that magnitudes are consistent across
    // isolation levels.
    static void _ScaleTangentStencil( HbrFace<T> const * f,
                                      int stencilsize,
                                      float *uderiv,
                                      float *vderiv );

    // Computes BSpline stencil weights at (u,v)
    void _GetBSplineStencilsAtUV( float u,
                                  float v,
                                  float *point,
                                  float *deriv1,
                                  float *deriv2 );

    HbrFace<T> * _face; // current face

    int _quadrant,      // current quadrant (if _face is not a quad)
        _sharpedge;     // index of the sharp edge (if any)

    std::vector<int> _vertIndices; // indices of the control vertices

    HbrFace<T> const * _bsplineFace; // current b-spline sub-face

    FarVertexStencil * _bsplineStencils[16],  // control stencils for current
                     * _reflectedStencils[4]; // b-spline sub-face

    FarVertexStencilAllocator * _allocator;
};

// Gathers all the coarse control vertices for an arbitrary patch
template <class T> void
FarStencilTablesFactory<T>::Patch::SetupControlStencils( HbrFace<T> * f,
                                                         int quadrant ) {

    assert(f and f->IsCoarse());

    // same face and same quadrant: control stencil and cached bspline patch
    // stay the same
    if (quadrant==GetCurrentQuadrant() and f==GetCurrentFace())
        return;

    // new face or new quadrant: control stencil may still be the same, but
    // cached bspline patch must be invalidated
    _quadrant = quadrant;
    _bsplineFace = NULL;
    if (f==GetCurrentFace())
        return;

    // new coarse face: control stencil is invalidated and must be recomputed
    _face = f;

    HbrMesh<T> * mesh = f->GetMesh();

    // reset stencil pointers in coarse vertices (if any)
    for (int i=0; i<(int)_vertIndices.size(); ++i) {
        mesh->GetVertex(_vertIndices[i])->GetData()._SetStencil(0);
    }

    // new face will require a new set of coarse vertices
    _vertIndices.clear();

    // most stencils should use 16 CVs : 64 should be enough for most faces
    _vertIndices.reserve( 16*4 );

    // Gather operator that collects all the 1-ring control vertices around a
    // face's corner
    //
    //         R2
    //           o
    //          /  .
    //         /     .
    //        /        .  R1           R0
    //    R3 o          o ----------- o ----
    //        \         |             |
    //         \        |             |
    //          \       | Corner      |
    //           o ---- o ----------- o ----
    //        R4 |      | ........... |
    //           |      | ........... |
    //           |      | .. Face  .. |
    //           |      | ........... |
    //           o ---- o ----------- o ----
    //        R5 |      |             |
    //           |      |             |
    //
    class GatherOperator : public HbrFaceOperator<T> {
        HbrFace<T> const * _face;
        HbrVertex<T> const * _corner;
        std::vector<int> & _verts;
    public:

        GatherOperator( HbrFace<T> const * face,
            HbrVertex<T> const * corner, std::vector<int> & verts  ) :
                _face(face), _corner(corner), _verts(verts) { }

        ~GatherOperator() { }

        virtual void operator() (HbrFace<T> &face) {

            HbrMesh<T> * mesh = face.GetMesh();
            assert(mesh);

            if (&face==_face)
                return;

            int nv = face.GetNumVertices();

            for (int i=0; i<nv; ++i) {

                HbrHalfedge<T> * e = face.GetEdge(i);

                // skip edges along original 0-ring face
                if (e and (e->GetRightFace()==_face or e->GetLeftFace()==_face))
                    continue;

                // skip original corner vertex
                if (e->GetDestVertex(mesh)!=_corner)
                    _verts.push_back(e->GetDestVertexID());
            }
        }
    };

    for (int i=0; i<f->GetNumVertices(); ++i) {

        HbrVertex<T> * corner = f->GetVertex(i);

        // save the the 0-ring corner vertices
        _vertIndices.push_back( corner->GetID() );

        // gather the 1-ring vertices
        GatherOperator op( f, corner, _vertIndices );
        corner->ApplyOperatorSurroundingFaces( op );
    }

    std::sort( _vertIndices.begin(), _vertIndices.end());

    std::vector<int>::const_iterator last = std::unique(_vertIndices.begin(), _vertIndices.end());

    int ssize = (int)(last-_vertIndices.begin());

    _vertIndices.resize(ssize);

    // At this point _vertIndices contains the IDs of all base vertices that
    // control the limit surface patch corresponding to the given face.
    // Now we need to create and cache a base stencil for each of these
    // control vertices.

    if (_allocator) {
        delete _allocator;
    }

    for (int i=0; i<4; ++i) {
        _reflectedStencils[i] = NULL;
    }

    _allocator = new FarVertexStencilAllocator(ssize);

    for (int i=0; i<ssize; ++i) {

        // Allocate a clear vertex stencil (all coefs set to 0.0)
        FarVertexStencil * stencil = _allocator->Allocate();

        stencil->Reset(0.0f);

        // Set the i-th coefficient of this stencil to 1.0
        stencil->_GetData()[i] = 1.0f;

        // Assign the stencil to the control vertex
        mesh->GetVertex(_vertIndices[i])->GetData()._SetStencil(stencil);
    }
}

// Evaluate the surface for the quad face left of edge at local coordinates
// (u,v), computing point and tangent stencils for non-null corresponding
// stencil pointers.
//
// In most cases only approximate evaluation is done by linearly interpolating
// between limit values after refLevel subdivision steps.
//
template <class T> bool
FarStencilTablesFactory<T>::Patch::GetStencilsAtUV( HbrHalfedge<T> * e,
                                                    float u,
                                                    float v,
                                                    int reflevel,
                                                    float *point,
                                                    float *uderiv,
                                                    float *vderiv ) {

    assert( _allocator );

    HbrFace<T> * f = e->GetLeftFace();

    if (f->IsHole())
        return false;

    // non-quad ? get corresponding quadrant sub-face
    if (f->GetMesh()->GetSubdivision()->FaceIsExtraordinary(f->GetMesh(), f)) {

        f->Refine();
        f = f->GetChild(GetCurrentQuadrant());
    }

    // We cache the B-spline stencils that we compute for a face,
    // so the first thing we do is check whether the face that was given
    // to us is the same one for which we last computed the B-spline
    // stencils. If this is so, we can simply return the pointer.
    if (f==_bsplineFace) {

        _GetBSplineStencilsAtUV( u, v, point, uderiv, vderiv );
        return true;

    } else if (_IsaBSpline(f)) {

        _UpdateBSplineStencils( f );
        _GetBSplineStencilsAtUV( u, v, point, uderiv, vderiv );
        return true;
    }

    // We reached the maximum recursion : give up !
    if (reflevel==0) {

        assert(not f->IsCoarse());

        FarVertexStencil * pt = _allocator->Allocate(),
                         * utan = _allocator->Allocate(),
                         * vtan = _allocator->Allocate();

        // Compute the bi-linear weights corresponding to the 4 face corners:
        float weights[4];
        weights[0] = (1.0f - u)*(1.0f - v);
        weights[1] = u * (1.0f - v);
        weights[2] = u * v;
        weights[3] = (1.0f - u) * v;

        // Iterate over the 4 corners
        for(int i = 0; i < 4; ++i) {

            if(weights[i] == 0)
                continue;

            HbrHalfedge<T> * elimit = f->GetEdge(i);
            HbrVertex<T> * vlimit = f->GetVertex(i);

            _GetLimitStencils( vlimit, pt->_GetData() );

            FarVertexStencil::AddScaled( point, pt, weights[i] );

            if (uderiv and vderiv) {

                _GetTangentLimitStencils( elimit, utan->_GetData(),
                                                  vtan->_GetData() );

                // Tangent vectors must compensate for the CCW rotation of elimit
                switch (i) {
                    case 0: {
                        FarVertexStencil::AddScaled( uderiv, utan,  weights[i] );
                        FarVertexStencil::AddScaled( vderiv, vtan,  weights[i] );
                    } break;

                    case 1: {
                        FarVertexStencil::AddScaled( uderiv, vtan, -weights[i] );
                        FarVertexStencil::AddScaled( vderiv, utan,  weights[i] );
                    } break;

                    case 2: {
                        FarVertexStencil::AddScaled( uderiv, utan, -weights[i] );
                        FarVertexStencil::AddScaled( vderiv, vtan, -weights[i] );
                    } break;

                    case 3: {
                        FarVertexStencil::AddScaled( uderiv, vtan,  weights[i] );
                        FarVertexStencil::AddScaled( vderiv, utan, -weights[i] );
                    } break;
                }
            }
        }

        _allocator->Deallocate(pt);
        _allocator->Deallocate(utan);
        _allocator->Deallocate(vtan);

        return true;
    } else {

        f->Refine();

        int quadrant=-1;

             if (u<=0.5f and v<=0.5f) { quadrant = 0; }
        else if (u> 0.5f and v<=0.5f) { quadrant = 1; u-=0.5f; }
        else if (u> 0.5f and v> 0.5f) { quadrant = 2; u-=0.5f; v-=0.5f; }
        else if (u<=0.5f and v> 0.5f) { quadrant = 3; v-=0.5f; }
        else
            assert(0);

        assert(quadrant>-1 and quadrant<4);

        HbrVertex<T> * a = f->GetVertex(quadrant)->Subdivide(),
                     * b = f->GetEdge(quadrant)->Subdivide();

        HbrHalfedge<T> * subedge = a->GetEdge(b);

        GetStencilsAtUV( subedge, 2.0f*u, 2.0f*v, reflevel-1, point, uderiv, vderiv );
    }
    return true;
}

// True if the vertex has a BSpline limit
template <class T> bool
FarStencilTablesFactory<T>::Patch::_IsaBSpline( HbrVertex<T> * v ) {

    if (v->IsExtraordinary() or v->GetMask(true))
        return false;

    HbrHalfedge<T> const * start = v->GetIncidentEdge(),
                         * e = start;
    if (e) do {

        HbrFace<T> * f = e->GetFace();

        // Adjacent face must be regular
        if (f->GetNumVertices() != 4)
            return false;

        // If the adjacent face has hierarchical edits which don't
        // terminate at this level, we can't subdivide
        if (HbrHierarchicalEdit<T> ** edits = f->GetHierarchicalEdits()) {
            while (HbrHierarchicalEdit<T> const * edit = *edits) {
                if (not edit->IsRelevantToFace(f))
                    break;
                if (edit->GetNSubfaces() > f->GetDepth())
                    return false;
                edits++;
            }
        }

        HbrHalfedge<T> const * next = v->GetNextEdge(e);
        if (!next) {
            // We encountered a break in the cycle. This means the
            // vertex is on a boundary!
            return false;
        } else {
            e = next;
        }
    } while (e != start);

    // If there is facevarying data, we check whether this is a
    // facevarying boundary vertex as well.
    if (v->GetMesh()->GetFVarCount() and
        v->GetMesh()->GetFVarInterpolateBoundaryMethod() !=  HbrMesh<T>::k_InterpolateBoundaryNone) {
        if (not v->IsFVarAllSmooth()) return false;
    }
    return true;
}


// True if the edge has a BSpline limit
template <class T> bool
FarStencilTablesFactory<T>::Patch::_IsaBSpline( HbrHalfedge<T> * e ) {

    // Edge operator checking sharpness of next edge in cycle
    struct _SharpOnext {
        HbrHalfedge<T> const *
        operator () (HbrHalfedge<T> const * e0) const {

            HbrVertex<T> const * v = e0->GetOrgVertex();

            HbrHalfedge<T> const * e = v->GetNextEdge(e0);

            if (e) do {
                if (e->GetSharpness())
                    break;
                e = v->GetNextEdge(e);
            } while (e and e != e0);
            return (e == e0 ? 0 : e);
        }
    };

    e->GetOrgVertex()->GuaranteeNeighbors();

    if (e->GetSharpness() and e->GetSharpness() < HbrHalfedge<T>::k_InfinitelySharp)
        return false;

    HbrMesh<T> const * mesh = e->GetLeftFace() ? e->GetLeftFace()->GetMesh() :
                                          e->GetRightFace()->GetMesh();

    // currently the B-spline code does not handle facevarying sharp edges correctly
    if (mesh->GetFVarCount() and
        mesh->GetFVarInterpolateBoundaryMethod() != HbrHalfedge<T>::k_InterpolateBoundaryNone) {

        for(int j = 0; j < mesh->GetFVarCount(); ++j)
            if(e->GetFVarSharpness(j))
                return false;
    }

    if (e->GetSharpness()) {
        if (e->GetOrgVertex()->GetMask(true) != HbrVertex<T>::k_Crease or
            e->GetDestVertex()->GetMask(true) != HbrVertex<T>::k_Crease)
            return false;

        if (!e->GetOpposite())
            return false;

        HbrHalfedge<T> const * e1 = _SharpOnext(e),
                             * e2 = _SharpOnext(e->GetOpposite());
        if (    (not e1)
             or (not e2)
             or e1->GetSharpness() < HbrHalfedge<T>::k_InfinitelySharp
             or e2->GetSharpness() < HbrHalfedge<T>::k_InfinitelySharp )
            return false;
    } else {
        if ((not _IsaBSpline(e->GetOrgVertex())) or
            (not _IsaBSpline(e->GetDestVertex()))) {
            return false;
        }
    }
    return true;
}


// True if the face has a BSpline limit
template <class T> bool
FarStencilTablesFactory<T>::Patch::_IsaBSpline( HbrFace<T> * f ) {

    int nv = f->GetNumVertices();

    if (nv != 4)
        return false;

    if (f->IsHole())
        return false;

    HbrMesh<T> const * mesh = f->GetMesh();

    HbrHalfedge<T> * sharpEdge = 0,
                   * edge = f->GetFirstEdge();

    for (int i = 0; i < 4; ++i) {

        edge->GetOrgVertex()->GuaranteeNeighbors();

        // currently the BSpline code does not handle facevarying
        // sharp edges correctly
        if (mesh->GetFVarCount() and mesh->GetFVarInterpolateBoundaryMethod() !=
            HbrMesh<T>::k_InterpolateBoundaryNone) {

            for(int j = 0; j < mesh->GetFVarCount(); ++j)
                if(edge->GetFVarSharpness(j))
                    return false;
        }

        // ensure that the face isn't adjacent to more than one sharp edge
        if (edge->GetSharpness() >= HbrHalfedge<T>::k_InfinitelySharp) {
            if (sharpEdge) {
                return false;
            } else {
                sharpEdge = edge;
            }
        }
        edge = edge->GetNext();
    }

    if (sharpEdge) {

        // Make sure the two vertices on the sharp edge are creases
        // and the other two vertices are bspline verts
        if ( sharpEdge->GetOrgVertex()->GetMask(true)!=HbrVertex<T>::k_Crease or
             sharpEdge->GetDestVertex()->GetMask(true)!=HbrVertex<T>::k_Crease or
            (not _IsaBSpline(sharpEdge->GetNext()->GetDestVertex())) or
            (not _IsaBSpline(sharpEdge->GetPrev()->GetOrgVertex()))) {
            return false;
        }

        // Make sure the hard edge continues being a hard edge on adjacent
        // two faces
        edge = sharpEdge->GetNext()->GetOpposite();
        if (!edge) {
            return false;
        }

        if (edge->GetNext()->GetSharpness() !=
            HbrHalfedge<T>::k_InfinitelySharp) {
            return false;
        }

        edge = sharpEdge->GetPrev()->GetOpposite();
        if ((not edge) or edge->GetPrev()->GetSharpness() !=
            HbrHalfedge<T>::k_InfinitelySharp) {
            return false;
        }
    } else {

        // All four vertices must be spline verts
        edge = f->GetFirstEdge();
        for (int i = 0; i < 4; ++i) {
            if (!_IsaBSpline(edge->GetOrgVertex()))
                return false;
            edge = edge->GetNext();
        }
    }
    return true;
}


// Updates the cached BSpline stencils
template <class T> void
FarStencilTablesFactory<T>::Patch::_UpdateBSplineStencils( HbrFace<T> const * f ) {

    assert( f->GetNumVertices()==4 );

    _bsplineFace = f;

    // Fill out the array of stencils assuming this is a simple B-Spline
    // case, and no stencil reflection is necessary.
    static size_t indices[16] = {5, 4, 0, 1, 6, 2, 3, 7, 10, 11, 15, 14, 9, 13, 12, 8};

    int sharpedge = -1;

    for (int i=0; i<4; ++i) {
        HbrHalfedge<T> const * e = f->GetEdge(i);

        // First check if this edge is sharp, and if so remember it.
        // Also, if it's sharp we don't want all of the four stencils
        // whose numbers are given by indices[4*i]..indices[4*i+3].
        // Rather, we need just the first two (the remaining two
        // will be computed by reflection later in this function).
        //
        //   |           |           |
        //   |           |           |         --- : regular edge
        //   |           |           |
        //   | (4*i+1)   | (4*i)     |         === : sharp edge
        //   X --------- X ========= o ------
        //        e1           e                X  : accumulated vertices
        //
        if (e->GetSharpness() >= HbrHalfedge<T>::k_InfinitelySharp) {

            sharpedge=i;

            HbrHalfedge<T> * e1 = e->GetPrev()->GetOpposite()->GetPrev();

            _bsplineStencils[indices[4*i  ]] = e->GetOrgVertex()->GetData()._GetStencil();
            _bsplineStencils[indices[4*i+1]] = e1->GetOrgVertex()->GetData()._GetStencil();
            continue;
        }

        // Also must check if the previous edge was sharp. If it was, then
        // once again, we only need two stencils instead of four. In this
        // case those are the stencils numbered indices[4*i] and
        // indices[4*i+3].
        //
        //   |           ||          |
        //   |           ||          |         --- : regular edge
        //   |           ||          |
        //   |           || (4*i)    |          || : sharp edge
        //   X --------- X --------- o ------
        //               |     e     |          X  : accumulated vertices
        //               |           |
        //               |
        //               | (4*i+3)
        //          ---- X ----
        //
        if (e->GetPrev()->GetSharpness() >= HbrHalfedge<T>::k_InfinitelySharp) {

            _bsplineStencils[indices[4*i  ]] = e->GetOrgVertex()->GetData()._GetStencil();
            _bsplineStencils[indices[4*i+3]] = e->GetOpposite()->GetNext()->GetDestVertex()->GetData()._GetStencil();
            continue;
        }

        // Both the current edge, and the previous one are smooth,
        // so we should get all four stencils in this case.
        HbrHalfedge<T> * e2 = e->GetOpposite()->GetNext()->GetOpposite()->GetNext();
        for (int j=0; j<4; ++j) {
            assert(e2);
            _bsplineStencils[indices[4*i+j]] = e2->GetOrgVertex()->GetData()._GetStencil();
            e2 = e2->GetNext();
        }
    }

    if (sharpedge<0)
        return;

    if (not _reflectedStencils[0]) {
        _reflectedStencils[0] = _allocator->Allocate();
        _reflectedStencils[1] = _allocator->Allocate();
        _reflectedStencils[2] = _allocator->Allocate();
        _reflectedStencils[3] = _allocator->Allocate();
    }

    // There is a sharp edge around the face, pointed to by sharpEdge.
    // We must override the corresponding 4 outer control stencils by
    // reflecting 4 internal control stencils across this edge.
    // We'll make use of the following three static index arrays.

    // reflect[k] lists the indices of those stencils who must be reflected
    // if the sharp edge is the k-th edge ccw around the face.
    static int reflect[4][4] = {{8, 9, 10, 11},{1, 5, 9, 13},{7, 6, 5, 4},{14, 10, 6, 2}};

    // via[k] lists the indices of those stencils in the row or column
    // corresponding to the k-th sharp edge (ccw around the face).
    static int via[4][4] = {{4, 5, 6, 7}, {2, 6, 10, 14}, {11, 10, 9, 8}, {13, 9, 5, 1}};

    // dest[k] lists the destination of the reflected stencils
    // corresponding to the k-th sharp edge (ccw around the face).
    static int dest[4][4] = {{0, 1, 2, 3}, {3, 7, 11, 15}, {15, 14, 13, 12}, {12, 8, 4, 0}};

    // We must reflect the stencil reflect[sharpEdgeIndex][j] via
    // stencil via[sharpEdgeIndex][j], overriding the stencil
    // dest[sharpEdgeIndex][j]
    for (int i=0; i<4; ++i) {
        _reflectedStencils[i]->Combine(
             2.0, *(_bsplineStencils[    via[sharpedge][i]]),
            -1.0, *(_bsplineStencils[reflect[sharpedge][i]])  );

        _bsplineStencils[dest[sharpedge][i]] = _reflectedStencils[i];
    }

    return;
}

// Computes BSpline weights
template <class T> void
FarStencilTablesFactory<T>::Patch::_GetBSplineWeights( float t,
                                                       float *cubicWeights,
                                                       float *quadraticWeights) {

    // The weights for the four uniform cubic B-Spline basis functions are:
    // (1/6)(1 - t)^3
    // (1/6)(3t^3 - 6t^2 + 4)
    // (1/6)(-3t^3 + 3t^2 + 3t + 1)
    // (1/6)t^3
    float t2 = t*t;
    float t3 = 3*t2*t;
    float w0 = 1 - t;

    cubicWeights[0] = (w0*w0*w0) / 6.0f;
    cubicWeights[1] = (t3 - 6.0f*t2 + 4.0f) / 6.0f;
    cubicWeights[2] = (3.0f*t2 - t3 + 3.0f*t + 1.0f) / 6.0f;
    cubicWeights[3] = t3 / 18.0f;

    // The weights for the three uniform quadratic basis functions are:
    // (1/2)(1-t)^2
    // (1/2)(1 + 2t - 2t^2)
    // (1/2)t^2
    quadraticWeights[0] = 0.5f * w0 * w0;
    quadraticWeights[1] = 0.5f + t - t2;
    quadraticWeights[2] = 0.5f * t2;
}


// Computes BSpline stencil weights at (u,v)
template <class T> void
FarStencilTablesFactory<T>::Patch::_GetBSplineStencilsAtUV( float u,
                                                            float v,
                                                            float *point,
                                                            float *deriv1,
                                                            float *deriv2 ) {


    float uWeights[4], vWeights[4], duWeights[3], dvWeights[3];
    _GetBSplineWeights(u, uWeights, duWeights);
    _GetBSplineWeights(v, vWeights, dvWeights);

    if (point) {

        FarVertexStencil::Reset(point, 0.0f, GetStencilSize());

        // Compute the tensor product weight corresponding to each control
        // vertex, and accumulate the weighted control stencils.
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float weight = uWeights[j] * vWeights[i];
                FarVertexStencil::AddScaled(point, _bsplineStencils[4*i+j], weight);
            }
        }
    }

    if (deriv1 and deriv2) {

        // Compute the du tangent stencil. This is done by taking the tensor
        // product between the quadratic weights computed for u and the cubic
        // weights computed for v. The stencil is constructed using
        // differences between consecutive vertices in each row (i.e.
        // in the u direction).

        FarVertexStencil::Reset(deriv1, 0.0f, GetStencilSize());

        for (int i = 0; i < 4; ++i) {
            float prevWeight = 0.0f;
            for (int j = 0; j < 3; ++j) {
                float weight = duWeights[j]*vWeights[i];
                FarVertexStencil::AddScaled(deriv1, _bsplineStencils[4*i+j], prevWeight - weight);
                prevWeight = weight;
            }
            FarVertexStencil::AddScaled(deriv1, _bsplineStencils[4*i+3], prevWeight);
        }

        FarVertexStencil::Reset(deriv2, 0.0f, GetStencilSize());

        for (int j = 0; j < 4; ++j) {
            float prevWeight = 0.0f;
            for (int i = 0; i < 3; ++i) {
                float weight = uWeights[j]*dvWeights[i];
                FarVertexStencil::AddScaled(deriv2, _bsplineStencils[4*i+j], prevWeight - weight);
                prevWeight = weight;
            }
            FarVertexStencil::AddScaled(deriv2, _bsplineStencils[12+j], prevWeight);
        }
        _ScaleTangentStencil(_bsplineFace, GetStencilSize(), deriv1, deriv2);
    }
}

// Computes the limit stencils
template <class T> void
FarStencilTablesFactory<T>::Patch::_GetLimitStencils( HbrVertex<T> * v,
                                                      float *point ) {
    v->GuaranteeNeighbors();

    // Refine vertex until stable
    while (v->IsVolatile()) {

        v->Refine();
        HbrVertex<T> * vfine = v->Subdivide();
        assert(vfine);
        v=vfine;
    }

    assert(v->GetMask(false) == v->GetMask(true));
    switch(v->GetMask(false)) {

        case HbrVertex<T>::k_Smooth:
        case HbrVertex<T>::k_Dart: {

            assert(not v->OnBoundary());

            FarVertexStencil::Reset(point, 0.0f, GetStencilSize());

            int n = v->GetValence();

            //
            //          o ------- o         o
            //     b0*1           | a0*4    | b3*1
            //                    |         |
            //                    |         |
            //                    |         |
            //          o ------- o ------- o
            //     a1*4 |   point |           a3*4
            //          |         |
            //          |         |
            //          |         |
            //          o         o ------- o
            //     b1*1      a2*4             b2*1
            //
            //  point = (a0+a1+a2+a3)*4 + (b0+b1+b2+b3)
            //
            class SmoothVertexOperator : public HbrHalfedgeOperator<T> {
            public:
                SmoothVertexOperator(HbrVertex<T> *v, float * point) :
                    _vertex(v), _point(point) { }

                ~SmoothVertexOperator() { }

                virtual void operator() (HbrHalfedge<T> &e) {

                    HbrVertex<T> * a = e.GetDestVertex();
                    HbrHalfedge<T> * next = e.GetNext();
                    if (a==_vertex) {
                        a = e.GetOrgVertex();
                        next = e.GetPrev();
                    }
                    HbrVertex<T> * b = next->GetDestVertex();

                    FarVertexStencil::AddScaled( _point, a->GetData().GetStencil(), 4.0f );
                    FarVertexStencil::Add( _point, b->GetData().GetStencil() );
                }
            private:
                HbrVertex<T> * _vertex;
                float * _point;
            };

            SmoothVertexOperator op( v, point );
            v->ApplyOperatorSurroundingEdges(op);

            float s = 1.0f / float(n*(n+5.0f));
            FarVertexStencil::Scale(point, s, GetStencilSize());
            FarVertexStencil::AddScaled(point, v->GetData().GetStencil(), s*n*n );
        } break;

        case HbrVertex<T>::k_Crease: {

            FarVertexStencil::Reset(point, 0.0f, GetStencilSize());

            //
            //        |         |         |
            //        |         |         |
            //        |         |         |         --- : regular edge
            //   ---- o ======= o ======= o ---
            //     a1 |   point |         | a0      === : sharp edge
            //        |         |         |
            //        |         |         |
            //
            //   point = a0 + a1
            //
            class CreaseEdgeOperator : public HbrHalfedgeOperator<T> {
            public:

                CreaseEdgeOperator(HbrVertex<T> *v, float * point) :
                    _vertex(v), _point(point), _count(0) { }

                ~CreaseEdgeOperator() { }

                virtual void operator() (HbrHalfedge<T> &e) {

                    if (_count<2 and e.IsSharp(false)) {

                        HbrVertex<T> * a = e.GetDestVertex();
                        if (a==_vertex)
                            a = e.GetOrgVertex();

                        FarVertexStencil::Add( _point, a->GetData().GetStencil() );

                        ++_count;
                    }
                }

            private:
                HbrVertex<T> * _vertex;
                float * _point;
                int _count;
            };

            CreaseEdgeOperator op( v, point );
            v->ApplyOperatorSurroundingEdges(op);

            FarVertexStencil::Scale( point, 1.0f/6.0f, GetStencilSize() );
            FarVertexStencil::AddScaled( point, v->GetData().GetStencil(), 2.0f/3.0f );

        } break;

        case HbrVertex<T>::k_Corner: {
            FarVertexStencil::Copy( point, v->GetData().GetStencil() );
        } break;
    }
}

// Scale tangent stencils so that magnitudes are consistent across
// isolation levels.
template <class T> void
FarStencilTablesFactory<T>::Patch::_ScaleTangentStencil( HbrFace<T> const * f,
                                                         int stencilsize,
                                                         float *uderiv,
                                                         float *vderiv ) {

    float scale = float (1 << f->GetDepth());
    FarVertexStencil::Scale(uderiv, scale, stencilsize);
    FarVertexStencil::Scale(vderiv, scale, stencilsize);
}


// Computes derivative limit stencils
template <class T> void
FarStencilTablesFactory<T>::Patch::_GetTangentLimitStencils( HbrHalfedge<T> * e,
                                                             float * uderiv,
                                                             float * vderiv ) {

    // Boundary vertex tangent stencil table generated using the python script:
    // opensubdiv/tools/tangentStencils/tangentStencils.py

#define FAR_LIMITTANGENT_MAXVALENCE 20

    static float creaseK[][40] =
        { { -0.662085f,  -0.082761f,   0.662085f,  -0.248282f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.165521f,   0.165521f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.430450f,   0.385279f,  -0.430450f,  -0.430450f,   0.475622f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,  -0.107613f,  -0.215225f,  -0.107613f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          { -0.263966f,  -0.557805f,   0.263966f,   0.373304f,   0.263966f,  -0.530084f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.065992f,   0.159318f,   0.159318f,   0.065992f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.173071f,   0.574415f,  -0.173071f,  -0.280034f,  -0.280034f,  -0.173071f,   0.611831f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,  -0.043268f,  -0.113276f,  -0.140017f,  -0.113276f,  -0.043268f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.110310f,   0.858370f,  -0.110310f,  -0.191063f,  -0.220620f,  -0.191063f,  -0.110310f,   0.266369f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,  -0.027578f,  -0.075343f,  -0.102921f,  -0.102921f,  -0.075343f,  -0.027578f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.059499f,  -0.108652f,  -0.059499f,  -0.107213f,  -0.133693f,  -0.133693f,  -0.107213f,  -0.059499f,   0.950367f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,  -0.014875f,  -0.041678f,  -0.060226f,  -0.066846f,  -0.060226f,  -0.041678f,  -0.014875f,   0.000000f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          { -0.069034f,  -0.643889f,   0.069034f,   0.127558f,   0.166663f,   0.180394f,   0.166663f,   0.127558f,   0.069034f,  -0.647432f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.017258f,   0.049148f,   0.073555f,   0.086764f,   0.086764f,   0.073555f,   0.049148f,   0.017258f,   0.000000f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.054360f,   0.565883f,  -0.054360f,  -0.102164f,  -0.137645f,  -0.156524f,  -0.156524f,  -0.137645f,  -0.102164f,  -0.054360f,
             0.731835f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,  -0.013590f,  -0.039131f,  -0.059952f,  -0.073542f,  -0.078262f,  -0.073542f,  -0.059952f,  -0.039131f,  -0.013590f,
             0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          { -0.044000f,  -0.755024f,   0.044000f,   0.083693f,   0.115193f,   0.135418f,   0.142386f,   0.135418f,   0.115193f,   0.083693f,
             0.044000f,  -0.549465f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.011000f,   0.031923f,   0.049721f,   0.062653f,   0.069451f,   0.069451f,   0.062653f,   0.049721f,   0.031923f,
             0.011000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.010145f,  -0.497027f,  -0.010145f,  -0.019467f,  -0.027213f,  -0.032754f,  -0.035642f,  -0.035642f,  -0.032754f,  -0.027213f,
            -0.019467f,  -0.010145f,   0.862545f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,  -0.002536f,  -0.007403f,  -0.011670f,  -0.014992f,  -0.017099f,  -0.017821f,  -0.017099f,  -0.014992f,  -0.011670f,
            -0.007403f,  -0.002536f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          { -0.029547f,  -0.419056f,   0.029547f,   0.057081f,   0.080725f,   0.098867f,   0.110272f,   0.114162f,   0.110272f,   0.098867f,
             0.080725f,   0.057081f,   0.029547f,  -0.852117f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.007387f,   0.021657f,   0.034451f,   0.044898f,   0.052285f,   0.056109f,   0.056109f,   0.052285f,   0.044898f,
             0.034451f,   0.021657f,   0.007387f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.022520f,   0.197157f,  -0.022520f,  -0.043731f,  -0.062400f,  -0.077443f,  -0.087986f,  -0.093415f,  -0.093415f,  -0.087986f,
            -0.077443f,  -0.062400f,  -0.043731f,  -0.022520f,   0.942807f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,  -0.005630f,  -0.016563f,  -0.026533f,  -0.034961f,  -0.041357f,  -0.045350f,  -0.046707f,  -0.045350f,  -0.041357f,
            -0.034961f,  -0.026533f,  -0.016563f,  -0.005630f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          { -0.022403f,  -0.513093f,   0.022403f,   0.043683f,   0.062773f,   0.078714f,   0.090709f,   0.098155f,   0.100680f,   0.098155f,
             0.090709f,   0.078714f,   0.062773f,   0.043683f,   0.022403f,  -0.804837f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.005601f,   0.016522f,   0.026614f,   0.035372f,   0.042356f,   0.047216f,   0.049709f,   0.049709f,   0.047216f,
             0.042356f,   0.035372f,   0.026614f,   0.016522f,   0.005601f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          { -0.019877f,  -0.743782f,   0.019877f,   0.038885f,   0.056194f,   0.071047f,   0.082795f,   0.090924f,   0.095079f,   0.095079f,
             0.090924f,   0.082795f,   0.071047f,   0.056194f,   0.038885f,   0.019877f,  -0.600744f,   0.000000f,   0.000000f,   0.000000f,
             0.000000f,   0.004969f,   0.014691f,   0.023770f,   0.031810f,   0.038460f,   0.043430f,   0.046501f,   0.047540f,   0.046501f,
             0.043430f,   0.038460f,   0.031810f,   0.023770f,   0.014691f,   0.004969f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.015725f,   0.289165f,  -0.015725f,  -0.030845f,  -0.044780f,  -0.056994f,  -0.067018f,  -0.074466f,  -0.079053f,  -0.080602f,
            -0.079053f,  -0.074466f,  -0.067018f,  -0.056994f,  -0.044780f,  -0.030845f,  -0.015725f,   0.922656f,   0.000000f,   0.000000f,
             0.000000f,  -0.003931f,  -0.011642f,  -0.018906f,  -0.025444f,  -0.031003f,  -0.035371f,  -0.038380f,  -0.039914f,  -0.039914f,
            -0.038380f,  -0.035371f,  -0.031003f,  -0.025444f,  -0.018906f,  -0.011642f,  -0.003931f,   0.000000f,   0.000000f,   0.000000f, },
          {  0.015399f,   0.556833f,  -0.015399f,  -0.030273f,  -0.044117f,  -0.056458f,  -0.066877f,  -0.075018f,  -0.080605f,  -0.083446f,
            -0.083446f,  -0.080605f,  -0.075018f,  -0.066877f,  -0.056458f,  -0.044117f,  -0.030273f,  -0.015399f,   0.784351f,   0.000000f,
             0.000000f,  -0.003850f,  -0.011418f,  -0.018598f,  -0.025144f,  -0.030834f,  -0.035474f,  -0.038906f,  -0.041013f,  -0.041723f,
            -0.041013f,  -0.038906f,  -0.035474f,  -0.030834f,  -0.025144f,  -0.018598f,  -0.011418f,  -0.003850f,   0.000000f,   0.000000f, },
          { -0.006900f,  -0.951532f,   0.006900f,   0.013591f,   0.019869f,   0.025543f,   0.030441f,   0.034414f,   0.037341f,   0.039134f,
             0.039737f,   0.039134f,   0.037341f,   0.034414f,   0.030441f,   0.025543f,   0.019869f,   0.013591f,   0.006900f,   0.277130f,
             0.000000f,   0.001725f,   0.005123f,   0.008365f,   0.011353f,   0.013996f,   0.016214f,   0.017939f,   0.019119f,   0.019718f,
             0.019718f,   0.019119f,   0.017939f,   0.016214f,   0.013996f,   0.011353f,   0.008365f,   0.005123f,   0.001725f,   0.000000f, },
          {  0.010264f,   0.154214f,  -0.010264f,  -0.020248f,  -0.029680f,  -0.038302f,  -0.045879f,  -0.052205f,  -0.057107f,  -0.060451f,
            -0.062146f,  -0.062146f,  -0.060451f,  -0.057107f,  -0.052205f,  -0.045879f,  -0.038302f,  -0.029680f,  -0.020248f,  -0.010264f,
             0.964364f,  -0.002566f,  -0.007628f,  -0.012482f,  -0.016995f,  -0.021045f,  -0.024521f,  -0.027328f,  -0.029389f,  -0.030649f,
            -0.031073f,  -0.030649f,  -0.029389f,  -0.027328f,  -0.024521f,  -0.021045f,  -0.016995f,  -0.012482f,  -0.007628f,  -0.002566f,
          },
        };

    HbrVertex<T> * v = e->GetOrgVertex();

    while (v->IsVolatile()) {
        v->Refine();
        v = v->Subdivide();
        e = v->GetEdge(e->Subdivide());
        assert(e);
    }

    assert(v->GetMask(false) == v->GetMask(true));
    switch(v->GetMask(false)) {

        case HbrVertex<T>::k_Smooth:
        case HbrVertex<T>::k_Dart: {

            FarVertexStencil::Reset(uderiv, 0.0f, GetStencilSize());
            FarVertexStencil::Reset(vderiv, 0.0f, GetStencilSize());

            int n = v->GetValence();

            float alpha = 2.0f * static_cast<float>(M_PI) / (float) n,
                  c0 = 2.0f * cosf(alpha),
                  c1 = 1.0f,
                  A  = 1.0f + c0 + sqrtf(18.0f + c0) * cosf(0.5f * alpha);

            int i = 0;
            float d = 0.0f;
            HbrHalfedge<T> * e0=e;
            if (e0) do {

                HbrHalfedge<T> * onext = v->GetNextEdge(e0);
                assert(onext);

                HbrVertex<T> * w1 = e0->GetDestVertex(),
                             * f1 = e0->GetNext()->GetDestVertex(),
                             * w2 = onext->GetDestVertex(),
                             * f2 = onext->GetNext()->GetDestVertex();

                c0 = c1;
                c1 = cosf((i+1.0f) * alpha);

                float K1 = A * c0,
                      K2 = c0 + c1;

                d += fabsf(K1) + fabsf(K2);

                 // Accumulate
                FarVertexStencil::AddScaled(uderiv, w1->GetData().GetStencil(), K1);
                FarVertexStencil::AddScaled(uderiv, f1->GetData().GetStencil(), K2);

                FarVertexStencil::AddScaled(vderiv, w2->GetData().GetStencil(), K1);
                FarVertexStencil::AddScaled(vderiv, f2->GetData().GetStencil(), K2);

                e0 = onext;
                i++;

            } while (e0 != e);

            // XXXX prman scales deriv1 and deriv2 by 1.0/d
            // Why? Do we need to do this as well?
            float invd = 1.0f/d;
            FarVertexStencil::Scale(uderiv, invd, GetStencilSize());
            FarVertexStencil::Scale(vderiv, invd, GetStencilSize());
        } break;

        case HbrVertex<T>::k_Crease: {

            std::list<HbrHalfedge<T> *> edges;
            v->GetSurroundingEdges(std::back_inserter(edges));

            std::list<HbrVertex<T> *> vertices;
            v->GetSurroundingVertices(std::back_inserter(vertices));

            assert(edges.size()==vertices.size());

            // Circle the lists around so that we start processing at the edge
            // after 'e'
            while (*edges.rbegin() != e) {
                edges.push_back(edges.front()); edges.pop_front();
                vertices.push_back(vertices.front()); vertices.pop_front();
            }

            // Look for the two sharp edges
            int idx=0, e1i=-1, e2i=-1;
            typename std::list<HbrHalfedge<T> *>::iterator ei;
            typename std::list<HbrVertex<T> *>::iterator vi, v1i, v2i;
            for (ei=edges.begin(), vi=vertices.begin(); ei!=edges.end(); ++ei, ++vi, ++idx) {
                if ((*ei)->IsSharp(false)) {
                    if (e2i<0) {
                        e2i = idx;
                        v2i = vi;
                    } else {
                        e1i = idx;
                        v1i = vi;
                        break;
                    }
                }
            }

            // We expected (at least) two edges to be on a crease. Instead, there
            // are zero or 1. We're not really sure what to do in that case, but
            // the easiest thing to do is to ignore the crease, which fixes the
            // coredump at the very least.  The code here is exactly the same as
            // the default case.
            if (e1i<0 or e2i<0) {

                e = e->GetPrev();

                HbrVertex<T> * v1 = e->GetDestVertex(),
                             * v2 = e->GetNext()->GetDestVertex();

                FarVertexStencil::Subtract(uderiv, v1->GetData().GetStencil(), v->GetData().GetStencil());
                FarVertexStencil::Subtract(vderiv, v2->GetData().GetStencil(), v->GetData().GetStencil());
                break;
            }

            // Count the number of edges between e1 and e2 going clockwise.
            // Since e1 is AFTER e2 (see above), this just requires some math on
            // the edge indices
            int n = (int)edges.size() - e1i + e2i + 1;
            assert(n >= 2);

            // creaseK table has 11 entries : max valence is 10
            // XXXX error should be reported
            if (n >= FAR_LIMITTANGENT_MAXVALENCE) {
                break;
            }

            // Math on the two crease vertices
            HbrVertex<T> * v1 = *v1i,
                         * v2 = *v2i;

            FarVertexStencil::Subtract(uderiv, v1->GetData().GetStencil(), v2->GetData().GetStencil());
            FarVertexStencil::Scale(uderiv, 0.5f, GetStencilSize());

            FarVertexStencil::Reset(vderiv, 0.0f, GetStencilSize());
            FarVertexStencil::AddScaled(vderiv,  v->GetData().GetStencil(), creaseK[n][0]);
            FarVertexStencil::AddScaled(vderiv, v1->GetData().GetStencil(), creaseK[n][1]);
            FarVertexStencil::AddScaled(vderiv, v2->GetData().GetStencil(), creaseK[n][2]);

            // Math on vertices between the two creases
            float d = fabsf(creaseK[n][0]) + fabsf(creaseK[n][1]) + fabsf(creaseK[n][2]);
            idx = 3;
            vi = v1i;
            if ((++vi)==vertices.end()) {
                vi = vertices.begin();
            }
            while (vi!=v2i) {
                FarVertexStencil::AddScaled(vderiv, (*vi)->GetData().GetStencil(), creaseK[n][idx]);
                d += fabsf(creaseK[n][idx]);
                ++idx;
                if (++vi==vertices.end()) {
                    vi = vertices.begin();
                }
            }

            FarVertexStencil::Scale(vderiv, -2.0f/d, GetStencilSize());

        } break;

        case HbrVertex<T>::k_Corner: {

            HbrVertex<T> * v1 = e->GetDestVertex(),
                         * v2 = v->GetQEONext(v1);

            FarVertexStencil::Subtract(uderiv, v1->GetData().GetStencil(),  v->GetData().GetStencil());
            FarVertexStencil::Subtract(vderiv, v2->GetData().GetStencil(),  v->GetData().GetStencil());
        } break;
    }

    _ScaleTangentStencil(e->GetFace(), GetStencilSize(), uderiv, vderiv);
}

template <class T> int
FarStencilTablesFactory<T>::GetMaxValenceSupported() {
    return FAR_LIMITTANGENT_MAXVALENCE;
}


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_STENCILTABLE_FACTORY_H
