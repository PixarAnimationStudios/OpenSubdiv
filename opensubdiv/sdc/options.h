//
//   Copyright 2014 DreamWorks Animation LLC.
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
#ifndef SDC_OPTIONS_H
#define SDC_OPTIONS_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Sdc {

//
//  This header contains all supported options that can be applied to a subdivision
//  scheme to affect the shape of the limit surface.  These differ from approximations
//  that may be applied at a higher level, i.e. options to limit the level of feature
//  adaptive subdivision, options to ignore fractional creasing, or creasing entirely,
//  etc.  These options define a particular limit surface.
//
//  The intent is that these sets of options be defined at a high-level and propagated
//  into the lowest-level computation in support of each subdivision scheme.  Ideally
//  it remains a set of bit-fields (essentially an int) and so remains light weight and
//  easily passed down by value.
//
//  Questions:
//      Should the individual enum's be nested within the class or independent?
//
//  Note:
//      A case can be made that the CreaseMethod enum is better defined as part of the
//  Crease class, but the goal is to try and put them all in one place.  We could define
//  it there and aggregate it into Options here, but we need to be careful about the
//  possibility of circular dependencies (nesting types in classes inhibits forward
//  declaration).
//
class Options {
public:

    //  XXXX
    //  Manuel suggested "VertexBoundaryInterpolation" here, but when used, that sounded
    //  too much like boundary interpolation specific to a vertex -- I went with the VVar
    //  and FVar naming here instead (abbreviating the FaceVaryingBoundaryInterpolation
    //  that was suggested)...
    //
    enum VVarBoundaryInterpolation {
        VVAR_BOUNDARY_NONE = 0,
        VVAR_BOUNDARY_EDGE_ONLY,
        VVAR_BOUNDARY_EDGE_AND_CORNER
    };

    enum FVarBoundaryInterpolation {
        FVAR_BOUNDARY_BILINEAR = 0,
        FVAR_BOUNDARY_EDGE_ONLY,
        FVAR_BOUNDARY_EDGE_AND_CORNER,
        FVAR_BOUNDARY_ALWAYS_SHARP
    };

    //
    //  Tony has expressed a preference of UNIFORM vs NORMAL here, which diverges from
    //  Hbr/RenderMan, but makes a lot more sense as it allows us to distinguish between
    //  uniform and non-uniform creasing computations (with uniform being trivial).
    //
    enum CreasingMethod {
        CREASE_UNIFORM = 0,
        CREASE_CHAIKIN
    };

    //
    //  Is it possible to get rid of this entirely?  It is specific to Catmark, seems to
    //  be little used and only applies to the first level of subdivision.  Getting rid
    //  of the code that supports this (though it is localized) would be a relief...
    //
    enum TriangleSubdivision {
        TRI_SUB_NORMAL = 0,
        TRI_SUB_OLD,
        TRI_SUB_NEW
    };

    //
    //  This is speculative for now and included for illustration purposes -- the simplest
    //  set of interpolation rules for non-manifold features is to make them infinitely
    //  sharp, which fits into existing evaluation schemes.  Allowing them to be smooth is
    //  less well-defined and requires additional cases in the masks to properly support.
    //
    enum NonManifoldInterpolation {
        NON_MANIFOLD_NONE = 0,
        NON_MANIFOLD_SMOOTH,
        NON_MANIFOLD_SHARP
    };

public:

    //  Trivial constructor and destructor:
    Options() : _vvarBoundInterp(VVAR_BOUNDARY_NONE),
                   _fvarBoundInterp(FVAR_BOUNDARY_BILINEAR),
                   _nonManInterp(NON_MANIFOLD_NONE),
                   _creasingMethod(CREASE_UNIFORM),
                   _triangleSub(TRI_SUB_NORMAL),
                   _hbrCompatible(false) { }
    ~Options() { }

    //
    //  Trivial get/set methods:
    //
    VVarBoundaryInterpolation GetVVarBoundaryInterpolation() const { return (VVarBoundaryInterpolation) _vvarBoundInterp; }
    void SetVVarBoundaryInterpolation(VVarBoundaryInterpolation b) { _vvarBoundInterp = b; }

    FVarBoundaryInterpolation GetFVarBoundaryInterpolation() const { return (FVarBoundaryInterpolation) _fvarBoundInterp; }
    void SetFVarBoundaryInterpolation(FVarBoundaryInterpolation b) { _fvarBoundInterp = b; }

    CreasingMethod GetCreasingMethod() const { return (CreasingMethod) _creasingMethod; }
    void SetCreasingMethod(CreasingMethod c) { _creasingMethod = c; }

    NonManifoldInterpolation GetNonManifoldInterpolation() const { return (NonManifoldInterpolation) _nonManInterp; }
    void SetNonManifoldInterpolation(NonManifoldInterpolation n) { _nonManInterp = n; }

    TriangleSubdivision GetTriangleSubdivision() const { return (TriangleSubdivision) _triangleSub; }
    void SetTriangleSubdivision(TriangleSubdivision t) { _triangleSub = t; }

    //
    //  This may be premature, but it is useful to have some kind of flag so that users can be assured
    //  the options and meshes they specify are compliant with Hbr, RenderMan, etc.  How to measure that
    //  is still ill-defined given versions of Hbr, prMan will evolve...
    //
    bool GetHbrCompatibility() const { return _hbrCompatible; }
    void SetHbrCompatibility(bool onOrOff) { _hbrCompatible = onOrOff; }

private:
    //  Bitfield members:
    unsigned int _vvarBoundInterp : 2;
    unsigned int _fvarBoundInterp : 2;
    unsigned int _nonManInterp    : 2;
    unsigned int _creasingMethod  : 2;
    unsigned int _triangleSub     : 2;
    unsigned int _hbrCompatible   : 1;
};

} // end namespace sdc

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* SDC_OPTIONS_H */
