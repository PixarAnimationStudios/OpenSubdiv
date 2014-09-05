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
#ifndef SDC_CREASE_H
#define SDC_CREASE_H

#include "../version.h"

#include "../sdc/options.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Sdc {

//
//  Types, constants and utilities related to semi-sharp creasing -- whose implementation is
//  independent of the subdivision scheme.
//
//  Crease is intended to be a light-weight, trivially constructed class that computes
//  crease-related properties.  An instance of an Crease is defined with a set of options
//  that include current and future variations that will impact computations involving
//  sharpness values.
//
//  We do not to use Neighborhoods here as input.  Since their sharpness values are potentially
//  not specified (and gathered on demand), and the methods here rely more on the sharpness
//  values and less on the topology, we choose to work directly with the sharpness values for
//  more flexibility.  We also follow the trend of using primitive arrays in the interface.
//
//  Note on the need for and use of sharpness values:
//      In general, mask queries rely on the sharpness values.  The common case of a smooth
//  vertex, when known, avoids the need to inspect them, but unless the rules are well understood,
//  users will be expected to provided them -- particularly when they expect the mask queries
//  to do all of the work (just determining if a vertex is smooth will require inspection of
//  incident edge sharpness).
//      Mask queries will occassionally require the subdivided sharpness values around the
//  child vertex.  So users will be expected to either provide them up front when known, or to be
//  gathered on demand.  Any implementation of subdivision with creasing cannot avoid subdividing
//  the sharpness values first, so keeping them available for re-use is a worthwhile consideration.
//

class Crease {
public:
    //
    //  Constants and related queries of sharpness values:
    //
    static float const SHARPNESS_SMOOTH; // =  0.0f, do we really need this?
    static float const SHARPNESS_INFINITE;  // = 10.0f;

    static bool IsSmooth(float sharpness)    { return sharpness <= SHARPNESS_SMOOTH; }
    static bool IsSharp(float sharpness)     { return sharpness > SHARPNESS_SMOOTH; }
    static bool IsInfinite(float sharpness)  { return sharpness >= SHARPNESS_INFINITE; }
    static bool IsSemiSharp(float sharpness) { return (SHARPNESS_SMOOTH < sharpness) && (sharpness < SHARPNESS_INFINITE); }

    //
    //  Enum for the types of subdivision rules applied based on sharpness values (note these
    //  correspond to Hbr's vertex "mask").  The values are assigned to bit positions as it is
    //  useful to OR the corners of faces to quickly inspect its applicable rules.
    //
    enum Rule {
        RULE_UNKNOWN = 0,
        RULE_SMOOTH  = (1 << 0),
        RULE_DART    = (1 << 1),
        RULE_CREASE  = (1 << 2),
        RULE_CORNER  = (1 << 3)
    };

public:
    Crease() : _options() { }
    Crease(Options const& options) : _options(options) { }
    ~Crease() { }

    //
    //  Considering labeling the current/default/normal creasing method as "simple" in contrast
    //  to all others that are "complex".  The idea is that code can make certain assumptions
    //  and take some "simple" action in some cases to avoid the higher costs of dealing with
    //  more complex implementations.
    //
    bool IsUniform() const { return _options.GetCreasingMethod() == Options::CREASE_UNIFORM; }

    //
    //  Optional sharp features:
    //      Since options treat certain topological features as infinitely sharp -- boundaries
    //  or nonmanifold features -- sharpness values should be adjust before use.  The following
    //  methods will adjust specific  according to the options applied.
    //
    float SharpenBoundaryEdge(float edgeSharpness) const;
    float SharpenBoundaryVertex(float edgeSharpness) const;

    float SharpenNonManifoldEdge(float edgeSharpness) const;
    float SharpenNonManifoldVertex(float edgeSharpness) const;

    //
    //  Sharpness subdivision:
    //      The simple case for computing a subdivided sharpness value is as follows:
    //        - Smooth edges or verts stay Smooth
    //        - Sharp edges or verts stay Sharp
    //        - semi-sharp edges or verts are decremented by 1.0
    //  but for Chaikin (and potentially future creasing schemes that improve upon it) the
    //  computation is more involved.  In the case of edges in particular, the sharpness of a
    //  child edge is determined by the sharpness in the neighborhood of the end vertex
    //  corresponding to the child.  For this reason, an alternative to subdividing sharpness
    //  that computes all child edges around a vertex is given.
    //
    float SubdivideUniformSharpness(float vertexOrEdgeSharpness) const;

    float SubdivideVertexSharpness(float vertexSharpness) const;

    float SubdivideEdgeSharpnessAtVertex(float        edgeSharpness,
                                         int          incidentEdgeCountAtEndVertex,
                                         float const* edgeSharpnessAroundEndVertex) const;

    void SubdivideEdgeSharpnessesAroundVertex(int          incidentEdgeCountAtVertex,
                                              float const* incidentEdgeSharpnessAroundVertex,
                                              float*       childEdgesSharpnessAroundVertex) const;

    //
    //  Rule determination:
    //      Mask queries do not require the Rule to be known, it can be determined from
    //  the information provided, but it is generally more efficient when the Rule is known
    //  and provided.  In particular, the Smooth case dominates and is known to be applicable
    //  based on the origin of the vertex without inspection of sharpness.
    //
    Rule DetermineVertexVertexRule(float        vertexSharpness,
                                   int          incidentEdgeCount,
                                   float const* incidentEdgeSharpness) const;
    Rule DetermineVertexVertexRule(float        vertexSharpness,
                                   int          sharpEdgeCount) const;

    //
    //  Transitional weighting:
    //      When the rules applicable to a parent vertex and its child differ, one or more
    //  sharpness values has "decayed" to zero.  Both rules are then applicable and blended
    //  by a weight between 0 and 1 that reflects the transition.  Most often this will be
    //  a single sharpness value that decays from within the interval [0,1] to zero -- and
    //  the weight to apply is exactly that sharpness value -- but more than one may decay,
    //  and values > 1 may also decay to 0 in a single step while others within [0,1] may
    //  remain > 0.
    //      So to properly determine a transitional weight, sharpness values for both the
    //  parent and child must be inspected, combined and clamped accordingly.
    //
    //  Open questions:
    //      - does this method need to be public, or can it reside within the mask
    //        query classes? (though it would be the same for anything non-linear, so
    //        may be worth making a protected method somewhere)
    //      - does this need further consideration at an edge-vertex?
    //          - no, the edge-vertex case is far more trivial:  one non-zero sharpness
    //            for the edge that decays to zero for one or both child edges -- the
    //            transitional weight is simply the edge sharpness (clamped to 1)
    //      ? why pass only the parent vertex sharpness...
    //          - because it is so trivial to compute the child vertex sharpness?
    //          - may be better off passing both parent and child for both vertex and edge
    //            just to be clear here.
    //
    float ComputeFractionalWeightAtVertex(float        vertexSharpness,
                                          float        childVertexSharpness,
                                          int          incidentEdgeCount,
                                          float const* incidentEdgeSharpness,
                                          float const* childEdgesSharpness) const;

    //  Would these really help?  Maybe only need Rules for the vertex-vertex case...
    //
    //  Rule DetermineEdgeVertexRule(float parentEdgeSharpness) const;
    //  Rule DetermineEdgeVertexRule(float childEdge1Sharpness, float childEdge2Sharpness) const;

protected:
    float decrementSharpness(float sharpness) const;

private:
    Options _options;
};


//
//  Non-trivial inline declarations:
//
inline float
Crease::SharpenBoundaryEdge(float edgeSharpness) const {

    return (_options.GetVVarBoundaryInterpolation() != Options::VVAR_BOUNDARY_NONE) ?
            SHARPNESS_INFINITE : edgeSharpness;
}

inline float
Crease::SharpenBoundaryVertex(float vertexSharpness) const {

    return (_options.GetVVarBoundaryInterpolation() == Options::VVAR_BOUNDARY_EDGE_AND_CORNER) ?
            SHARPNESS_INFINITE : vertexSharpness;
}

inline float
Crease::SharpenNonManifoldEdge(float edgeSharpness) const {

    //  Shouldn't we error/assert somehow if indicated that non-manifold not supported?
    //  assert(_options.GetNonManifoldInterpolation() != Options::NON_MANIFOLD_NONE);

    return (_options.GetNonManifoldInterpolation() == Options::NON_MANIFOLD_SHARP) ?
            SHARPNESS_INFINITE : edgeSharpness;
}
inline float
Crease::SharpenNonManifoldVertex(float vertexSharpness) const {

    //  Shouldn't we error/assert somehow if indicated that non-manifold not supported?
    //  assert(_options.GetNonManifoldInterpolation() != Options::NON_MANIFOLD_NONE);

    return (_options.GetNonManifoldInterpolation() == Options::NON_MANIFOLD_SHARP) ?
            SHARPNESS_INFINITE : vertexSharpness;
}


inline float
Crease::decrementSharpness(float sharpness) const {

    if (IsSmooth(sharpness)) return Crease::SHARPNESS_SMOOTH;  // redundant but most common
    if (IsInfinite(sharpness)) return Crease::SHARPNESS_INFINITE;
    if (sharpness > 1.0f) return (sharpness - 1.0f);
    return Crease::SHARPNESS_SMOOTH;
}

inline float
Crease::SubdivideUniformSharpness(float vertexOrEdgeSharpness) const {

    return decrementSharpness(vertexOrEdgeSharpness);
}

inline float
Crease::SubdivideVertexSharpness(float vertexSharpness) const {

    return decrementSharpness(vertexSharpness);
}

} // end namespace sdc

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* SDC_CREASE_H */
