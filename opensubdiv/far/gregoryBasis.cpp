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

#include "../far/gregoryBasis.h"
#include "../far/error.h"
#include "../far/stencilTableFactory.h"
#include "../far/topologyRefiner.h"
#include "../vtr/stackBuffer.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

inline float computeCoefficient(int valence) {
    // precomputed coefficient table up to valence 29
    static float efTable[] = {
        0, 0, 0,
        0.812816f, 0.500000f, 0.363644f, 0.287514f,
        0.238688f, 0.204544f, 0.179229f, 0.159657f,
        0.144042f, 0.131276f, 0.120632f, 0.111614f,
        0.103872f, 0.09715f, 0.0912559f, 0.0860444f,
        0.0814022f, 0.0772401f, 0.0734867f, 0.0700842f,
        0.0669851f, 0.0641504f, 0.0615475f, 0.0591488f,
        0.0569311f, 0.0548745f, 0.0529621f
    };
    assert(valence > 0);
    if (valence < 30) return efTable[valence];

    float t = 2.0f * float(M_PI) / float(valence);
    return 1.0f / (valence * (cosf(t) + 5.0f +
                              sqrtf((cosf(t) + 9) * (cosf(t) + 1)))/16.0f);
}

//
//  There is a long and unclear history to the details of the patch conversion here...
//
//  The formulae for computing the Gregory patch points do not follow the more widely
//  accepted work of Loop, Shaefer et al or Myles et al.  The formulae for the limit
//  points and tangents also ultimately need to be retrieved from Sdc::Scheme to
//  ensure they conform, so future factoring of the formulae is still necessary.
//
//  This implementation is in the process of iterative refactoring to adapt it for
//  more general use.  The method is currently divided into four stages -- some of
//  which will eventually be moved externally and/or made into methods of their own:
//
//      - gather complete topology information for all four corners of the patch
//      - compute the vertex-points and intermediate values used below
//      - compute the edge-points
//      - compute the face-points (which depend on multiple edge-points)
//
GregoryBasis::ProtoBasis::ProtoBasis(
    Vtr::internal::Level const & level, Index faceIndex,
    Vtr::internal::Level::VSpan const cornerSpans[],
    int levelVertOffset, int fvarChannel) {

    //
    //  The first stage -- gather topology information for the entire patch:
    //
    //  This stage is intentionally separated from any computation as the information
    //  gathered here for each corner vertex (one-ring, valence, etc.) will eventually
    //  be passed to this function in a more general and compact form.  We have to
    //  be careful with face-varying channels to query the topology from the vertices
    //  of the level, while computing the patch basis from the points (fvar values).
    //
    Vtr::ConstIndexArray faceVerts = level.getFaceVertices(faceIndex);
    Vtr::ConstIndexArray facePoints = (fvarChannel < 0)
        ? faceVerts
        : level.getFaceFVarValues(faceIndex, fvarChannel);

    //  Should be use a "local" max valence here in future
    //  A discontinuous edge in the fvar topology can increase the valence by one.
    int maxvalence = level.getMaxValence() + int(fvarChannel>=0);

    Vtr::internal::StackBuffer<Index, 40> manifoldRings[4];
    manifoldRings[0].SetSize(maxvalence*2);
    manifoldRings[1].SetSize(maxvalence*2);
    manifoldRings[2].SetSize(maxvalence*2);
    manifoldRings[3].SetSize(maxvalence*2);

    bool  cornerBoundary[4];
    int   cornerValences[4];
    int   cornerNumFaces[4];
    int   cornerPatchFace[4];
    float cornerFaceAngle[4];

    //  Sum the number of source vertices contributing to the patch, which define the
    //  size of the stencil for each "point" involved.  We just want an upper bound
    //  here for now, so sum the vertices from the neighboring rings at each corner,
    //  but don't count the shared face points multiple times. 
    int stencilCapacity = 4;

    for (int corner = 0; corner < 4; ++corner) {

        // save for varying stencils
        varyingIndex[corner] = faceVerts[corner] + levelVertOffset;

        //  Gather the (partial) one-ring around the corner vertex:
        int ringSize = 0;
        if (!cornerSpans[corner].isAssigned()) {
            ringSize = level.gatherQuadRegularRingAroundVertex( faceVerts[corner],
                    manifoldRings[corner], fvarChannel);
        } else {
            ringSize = level.gatherQuadRegularPartialRingAroundVertex( faceVerts[corner],
                    cornerSpans[corner],
                    manifoldRings[corner], fvarChannel);
        }
        stencilCapacity += ringSize - 3;

        //  Cache topology information about the corner for ease of use later:
        if (ringSize & 1) {
            cornerBoundary[corner] = true;
            cornerNumFaces[corner] = (ringSize - 1) / 2;
            cornerValences[corner] = cornerNumFaces[corner] + 1;

            cornerFaceAngle[corner] = float(M_PI) / float(cornerNumFaces[corner]);

            //  Necessary to pad the ring to even size for the f[] and r[] computations...
            manifoldRings[corner][ringSize] = manifoldRings[corner][ringSize-1];
        } else {
            cornerBoundary[corner] = false;
            cornerNumFaces[corner] = ringSize / 2;
            cornerValences[corner] = cornerNumFaces[corner];

            cornerFaceAngle[corner] = 2.0f * float(M_PI) / float(cornerNumFaces[corner]);
        }

        //  Identify the patch-face within the ring of faces for the corner (which
        //  will later be identified externally and specified directly):
        int nEdgeVerts = cornerValences[corner];

        Index vNext = facePoints[(corner + 1) % 4];
        Index vPrev = facePoints[(corner + 3) % 4];

        cornerPatchFace[corner] = -1;
        for (int i = 0; i < nEdgeVerts; ++i) {
            int iPrev = (i + 1) % nEdgeVerts;
            if ((manifoldRings[corner][2*i] == vNext) && (manifoldRings[corner][2*iPrev] == vPrev)) {
                cornerPatchFace[corner] = i;
                break;
            }
        }
        assert(cornerPatchFace[corner] != -1);
    }

    //
    //  The first computation pass...
    //
    //  Compute vertex-point (P) and intermediate values (f[] and r[]) for each corner
    //
    Point e0[4], e1[4];

    Vtr::internal::StackBuffer<Point, 10> f(maxvalence);
    Vtr::internal::StackBuffer<Point, 40> r(maxvalence*4);

    for (int corner = 0; corner < 4; ++corner) {
        Index vCorner = facePoints[corner];

        int cornerValence = cornerValences[corner];

        //
        //  Compute intermediate f[] and r[] vectors:
        //
        //  The f[] are used to compute position and limit tangents for the interior case,
        //  which should eventually be computed directly with Sdc::Scheme methods -- so
        //  these f[] will ultimately be made obsolete.
        //
        //  The r[] are only used in computing face points Fp and Fm, and of the r[] that
        //  are allocated and computed for every edge of every corner vertex, only two are
        //  used for each corner vertex.  Aside from only computing the subset of r[] needed,
        //  these can be deferred to direct computation as part of Fp and Fm as they serve
        //  no other purpose.
        //
        //  Note also that the computations of each f[] and r[] do not take into account
        //  boundaries and relies on padding of the rings to provide an indexable value in
        //  these cases.
        //
        for (int i = 0; i < cornerValence; ++i) {

            int iPrev = (i+cornerValence-1)%cornerValence;
            int iNext = (i+1)%cornerValence;

            //  Identify the vertex at the end of each edge along with the previous and
            //  next face- and edge-vertex in the ring:
            Index vEdge     = (manifoldRings[corner][2*i]);
            Index vFaceNext = (manifoldRings[corner][2*i + 1]);
            Index vEdgeNext = (manifoldRings[corner][2*iNext]);
            Index vEdgePrev = (manifoldRings[corner][2*iPrev]);
            Index vFacePrev = (manifoldRings[corner][2*iPrev + 1]);

            float denom = 1.0f / (float(cornerValence) + 5.0f);
            f[i].Clear(4);
            f[i].AddWithWeight(vCorner,   float(cornerValence) * denom);
            f[i].AddWithWeight(vEdgeNext, 2.0f * denom);
            f[i].AddWithWeight(vEdge,     2.0f * denom);
            f[i].AddWithWeight(vFaceNext, denom);

            int rid = corner * maxvalence + i;
            r[rid].Clear(4);
            r[rid].AddWithWeight(vEdgeNext,  1.0f / 3.0f);
            r[rid].AddWithWeight(vEdgePrev, -1.0f / 3.0f);
            r[rid].AddWithWeight(vFaceNext,  1.0f / 6.0f);
            r[rid].AddWithWeight(vFacePrev, -1.0f / 6.0f);
        }

        //
        //  Compute the vertex point P[] and intermediate limit tangents e0 and e1:
        //
        //  The limit tangents e0 and e1 should be computed from Sdc::Scheme methods.
        //  But these explicit limit tangents vectors are not needed as intermediate
        //  results as the Ep and Em can be computed more directly from the limit
        //  masks for the tangent vectors.
        //
        if (cornerSpans[corner]._sharp) {
            P[corner].Clear(stencilCapacity);
            P[corner].AddWithWeight(vCorner, 1.0f);

            // Approximating these for now, pending future investigation...
            e0[corner].Clear(stencilCapacity);
            e0[corner].AddWithWeight(facePoints[corner],       2.0f / 3.0f);
            e0[corner].AddWithWeight(facePoints[(corner+1)%4], 1.0f / 3.0f);

            e1[corner].Clear(stencilCapacity);
            e1[corner].AddWithWeight(facePoints[corner],       2.0f / 3.0f);
            e1[corner].AddWithWeight(facePoints[(corner+3)%4], 1.0f / 3.0f);
        } else if (! cornerBoundary[corner]) {
            float theta    = cornerFaceAngle[corner];
            float posScale = 1.0f / float(cornerValence);
            float tanScale = computeCoefficient(cornerValence);

            P[corner].Clear(stencilCapacity);
            e0[corner].Clear(stencilCapacity);
            e1[corner].Clear(stencilCapacity);

            for (int i=0; i<cornerValence; ++i) {
                int iPrev = (i+cornerValence-1) % cornerValence;

                P[corner].AddWithWeight(f[i], posScale);

                float c0 = tanScale * 0.5f * cosf(float(i) * theta);
                e0[corner].AddWithWeight(f[i],     c0);
                e0[corner].AddWithWeight(f[iPrev], c0);

                float c1 = tanScale * 0.5f * sinf(float(i) * theta);
                e1[corner].AddWithWeight(f[i],     c1);
                e1[corner].AddWithWeight(f[iPrev], c1);
            }
        } else {
            Index vEdgeLeading  = manifoldRings[corner][0];
            Index vEdgeTrailing = manifoldRings[corner][2*cornerValence-1];

            P[corner].Clear(stencilCapacity);
            P[corner].AddWithWeight(vEdgeLeading,  1.0f / 6.0f);
            P[corner].AddWithWeight(vEdgeTrailing, 1.0f / 6.0f);
            P[corner].AddWithWeight(vCorner,       4.0f / 6.0f);

            float k = float(cornerNumFaces[corner]);
            float theta = cornerFaceAngle[corner];
            float c = cosf(theta);
            float s = sinf(theta);
            float div3kc = 1.0f / (3.0f*k+c);
            float gamma = -4.0f * s * div3kc;
            float alpha_0k = -((1.0f+2.0f*c) * sqrtf(1.0f+c)) * div3kc / sqrtf(1.0f-c);
            float beta_0 = s * div3kc;

            Index vEdge = manifoldRings[corner][0];
            Index vFace = manifoldRings[corner][1];

            e0[corner].Clear(stencilCapacity);
            e0[corner].AddWithWeight(vEdgeLeading,   1.0f / 6.0f);
            e0[corner].AddWithWeight(vEdgeTrailing, -1.0f / 6.0f);

            e1[corner].Clear(stencilCapacity);
            e1[corner].AddWithWeight(vCorner,       gamma);
            e1[corner].AddWithWeight(vEdgeLeading,  alpha_0k);
            e1[corner].AddWithWeight(vFace,         beta_0);
            e1[corner].AddWithWeight(vEdgeTrailing, alpha_0k);

            for (int i = 1; i < cornerValence - 1; ++i) {
                float alpha = 4.0f * sinf(float(i)*theta) * div3kc;
                float beta = (sinf(float(i)*theta) + sinf(float(i+1)*theta)) * div3kc;

                vEdge = manifoldRings[corner][2*i + 0];
                vFace = manifoldRings[corner][2*i + 1];

                e1[corner].AddWithWeight(vEdge, alpha);
                e1[corner].AddWithWeight(vFace, beta);
            }
            e1[corner] *= 1.0f / 3.0f;
        }
    }

    //
    //  The second computation pass...
    //
    //  Compute the edge points Ep and Em first.  These can be computed local to the corner,
    //  unlike the face points, whose computation requires edge points from adjacent corners
    //  and so are computed in a final pass after all edge points are available.
    //
    //  Consider merging this pass with the previous, now that face points have been deferred
    //  to a separate third pass.
    //
    //  Note that computation of Ep and Em here use intermediate limit tangents e0 and e1 and
    //  compute rotations of these for Ep and Em.  The masks for the limit tangents can be
    //  rotated topologically to avoid the explicit rotation here (at least for the interior
    //  case -- boundary case still warrants it until there is more flexibility in limit
    //  tangent masks orientation in Sdc)
    //
    for (int corner = 0; corner < 4; ++corner) {

        //  Identify edges in the ring pointing to the next and previous corner of the patch:
        int iEdgeNext = cornerPatchFace[corner];
        int iEdgePrev = (cornerPatchFace[corner] + 1) % cornerValences[corner];

        float faceAngle = cornerFaceAngle[corner];

        float faceAngleNext = faceAngle * float(iEdgeNext);
        float faceAnglePrev = faceAngle * float(iEdgePrev);

        if (cornerSpans[corner]._sharp) {
            Ep[corner] = e0[corner];
            Em[corner] = e1[corner];
        } else if (! cornerBoundary[corner]) {
            Ep[corner] = P[corner];
            Ep[corner].AddWithWeight(e0[corner], cosf(faceAngleNext));
            Ep[corner].AddWithWeight(e1[corner], sinf(faceAngleNext));

            Em[corner] = P[corner];
            Em[corner].AddWithWeight(e0[corner], cosf(faceAnglePrev));
            Em[corner].AddWithWeight(e1[corner], sinf(faceAnglePrev));
        } else if (cornerNumFaces[corner] > 1) {
            Ep[corner] = P[corner];
            Ep[corner].AddWithWeight(e0[corner], cosf(faceAngleNext));
            Ep[corner].AddWithWeight(e1[corner], sinf(faceAngleNext));

            Em[corner] = P[corner];
            Em[corner].AddWithWeight(e0[corner], cosf(faceAnglePrev));
            Em[corner].AddWithWeight(e1[corner], sinf(faceAnglePrev));
        } else {
            //  Edge points are on the control polygon here (with P midway between):
            Ep[corner].Clear(stencilCapacity);
            Ep[corner].AddWithWeight(facePoints[corner],       2.0f / 3.0f);
            Ep[corner].AddWithWeight(facePoints[(corner+1)%4], 1.0f / 3.0f);

            Em[corner].Clear(stencilCapacity);
            Em[corner].AddWithWeight(facePoints[corner],       2.0f / 3.0f);
            Em[corner].AddWithWeight(facePoints[(corner+3)%4], 1.0f / 3.0f);
        }
    }

    //
    //  The third pass...
    //
    //  Compute the face points Fp and Fm in terms of the vertex (P) and edge points (Ep and
    //  Em) previously computed.
    //
    for (int corner = 0; corner < 4; ++corner) {

        int cornerNext = (corner+1) % 4;
        int cornerOpp  = (corner+2) % 4;
        int cornerPrev = (corner+3) % 4;

        //  Identify edges in the ring pointing to the next and previous corner of the
        //  patch and the intermediate r[] associated with each:
        Point const * rp = &r[corner*maxvalence];

        Point const & rEdgeNext = rp[cornerPatchFace[corner]];
        Point const & rEdgePrev = rp[(cornerPatchFace[corner] + 1) % cornerValences[corner]];

        //  Coefficients to arrange the face points for tangent continuity across edges:
        float cosCorner = cosf(cornerFaceAngle[corner]);
        float cosPrev   = cosf(cornerFaceAngle[cornerPrev]);
        float cosNext   = cosf(cornerFaceAngle[cornerNext]);

        float s1 = 3.0f - 2.0f * cosCorner - cosNext;
        float s2 =        2.0f * cosCorner;
        float s3 = 3.0f - 2.0f * cosCorner - cosPrev;

        if (! cornerBoundary[corner]) {
            Fp[corner].Clear(stencilCapacity);
            Fp[corner].AddWithWeight(P[corner],      cosNext / 3.0f);
            Fp[corner].AddWithWeight(Ep[corner],     s1      / 3.0f);
            Fp[corner].AddWithWeight(Em[cornerNext], s2      / 3.0f);
            Fp[corner].AddWithWeight(rEdgeNext,      1.0f    / 3.0f);

            Fm[corner].Clear(stencilCapacity);
            Fm[corner].AddWithWeight(P[corner],      cosPrev / 3.0f);
            Fm[corner].AddWithWeight(Em[corner],     s3      / 3.0f);
            Fm[corner].AddWithWeight(Ep[cornerPrev], s2      / 3.0f);
            Fm[corner].AddWithWeight(rEdgePrev,      -1.0f   / 3.0f);
        } else if (cornerNumFaces[corner] > 1) {
            Fp[corner].Clear(stencilCapacity);
            Fp[corner].AddWithWeight(P[corner],      cosNext / 3.0f);
            Fp[corner].AddWithWeight(Ep[corner],     s1      / 3.0f);
            Fp[corner].AddWithWeight(Em[cornerNext], s2      / 3.0f);
            Fp[corner].AddWithWeight(rEdgeNext,      1.0f    / 3.0f);

            Fm[corner].Clear(stencilCapacity);
            Fm[corner].AddWithWeight(P[corner],      cosPrev / 3.0f);
            Fm[corner].AddWithWeight(Em[corner],     s3      / 3.0f);
            Fm[corner].AddWithWeight(Ep[cornerPrev], s2      / 3.0f);
            Fm[corner].AddWithWeight(rEdgePrev,      -1.0f   / 3.0f);

            if (cornerBoundary[cornerPrev]) {
                Fp[corner].Clear(stencilCapacity);
                Fp[corner].AddWithWeight(P[corner],      cosNext / 3.0f);
                Fp[corner].AddWithWeight(Ep[corner],     s1      / 3.0f);
                Fp[corner].AddWithWeight(Em[cornerNext], s2      / 3.0f);
                Fp[corner].AddWithWeight(rEdgeNext,      1.0f    / 3.0f);

                Fm[corner] = Fp[corner];
            } else if (cornerBoundary[cornerNext]) {
                Fm[corner].Clear(stencilCapacity);
                Fm[corner].AddWithWeight(P[corner],      cosPrev / 3.0f);
                Fm[corner].AddWithWeight(Em[corner],     s3      / 3.0f);
                Fm[corner].AddWithWeight(Ep[cornerPrev], s2      / 3.0f);
                Fm[corner].AddWithWeight(rEdgePrev,      -1.0f   / 3.0f);

                Fp[corner] = Fm[corner];
            }
        } else {
            Fp[corner].Clear(stencilCapacity);
            Fp[corner].AddWithWeight(facePoints[corner],     4.0f / 9.0f);
            Fp[corner].AddWithWeight(facePoints[cornerOpp],  1.0f / 9.0f);
            Fp[corner].AddWithWeight(facePoints[cornerNext], 2.0f / 9.0f);
            Fp[corner].AddWithWeight(facePoints[cornerPrev], 2.0f / 9.0f);

            Fm[corner] = Fp[corner];
        }
    }

    //
    //  Offset stencil indices...
    //
    //  These stencils are currently created relative to the level and have levelVertOffset
    //  to make them absolute indices.  But we will be localizing these to the patch itself
    //  and so any association/mapping with vertices or face-varying values in a Level will
    //  be handled externally.
    //
    for (int corner = 0; corner < 4; ++corner) {
        P[corner].OffsetIndices(levelVertOffset);
        Ep[corner].OffsetIndices(levelVertOffset);
        Em[corner].OffsetIndices(levelVertOffset);
        Fp[corner].OffsetIndices(levelVertOffset);
        Fm[corner].OffsetIndices(levelVertOffset);
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
