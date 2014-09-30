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
//  etc.  These options define the shape of a particular limit surface, including the
//  "shape" of primitive variable data associated with it.
//
//  The intent is that these sets of options be defined at a high-level and propagated
//  into the lowest-level computation in support of each subdivision scheme.  Ideally
//  it remains a set of bit-fields (essentially an int) and so remains light weight and
//  easily passed down by value.
//
//  ALPHA NOTES:
//      Several of these options are being reconsidered in light of the divergence of
//  OSD 3.0 from Hbr.  In some cases the options can be expressed more clearly and free
//  of any RenderMan legacy for future use.  Details are noted below:
//      "CreasingMethod"
//          - note the change from the default "Normal" method to "Uniform"
//      "VVarBoundaryInterpolation"
//          - both name and enumerations being reconsidered
//          - the "VVar" prefix was misguided on my part (barfowl)
//          - "boundary interpolation" is a potential misnomer as it only affects corners
//          - its effect is to sharpen edges/corners, but edges are always sharpened
//          - the "None" case serves no purpose (and would be discouraged)
//      "FVarBoundaryInterpolation" and the bool "propagate corners"
//          - as above, both name and enumerations being reconsidered
//          - again "boundary interpolation" a misnomer interior interpolation affected
//          - consider "FVarInterpolation" or "FVarLinearInterpolation" which by default
//            (0) is linear throughout and specifies where FVar interpolation is to be
//            Linear instead of following the assigned scheme.  Since the default (0) is
//            completely (bi)linear, each option successively removes linear features
//            making the last fully smooth:
//              FVAR_LINEAR_ALL,
//              FVAR_LINEAR_EDGE_AND_CORNER,
//              FVAR_LINEAR_CORNER,
//              FVAR_LINEAR_CORNER_PROPAGATE (incorporates "propagate corners")
//              FVAR_LINEAR_NONE,
//          - the "propgate corners" option only applied to one "interpolation option"
//            (formerly EDGE_AND_CORNER, now LINEAR_CORNER) and so has been added as a
//            new fifth choice.
//      "TriangleSubdivision":
//          - hoping we can get rid of this due to lack of interest/use
//          - specific to Catmark and only at level 0
//      "NonManifoldInterpolation":
//          - hoping we can get rid of this due to lack of interest/use
//
class Options {
public:
    enum VVarBoundaryInterpolation {
        VVAR_BOUNDARY_NONE = 0,
        VVAR_BOUNDARY_EDGE_ONLY,
        VVAR_BOUNDARY_EDGE_AND_CORNER
    };
    enum FVarBoundaryInterpolation {
        FVAR_BOUNDARY_BILINEAR = 0,
        FVAR_BOUNDARY_EDGE_ONLY,
        FVAR_BOUNDARY_EDGE_AND_CORNER,
        FVAR_BOUNDARY_ALWAYS_SHARP,
        FVAR_BOUNDARY_EDGE_AND_CORNER_PROP
    };
    enum CreasingMethod {
        CREASE_UNIFORM = 0,
        CREASE_CHAIKIN
    };
    enum TriangleSubdivision {
        TRI_SUB_NORMAL = 0,
        TRI_SUB_OLD,
        TRI_SUB_NEW
    };
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
    unsigned int _fvarBoundInterp : 3;
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
