..
     Copyright 2015 Pixar

     Licensed under the Apache License, Version 2.0 (the "Apache License")
     with the following modification; you may not use this file except in
     compliance with the Apache License and the following modification to it:
     Section 6. Trademarks. is deleted and replaced with:

     6. Trademarks. This License does not grant permission to use the trade
        names, trademarks, service marks, or product names of the Licensor
        and its affiliates, except as required to comply with Section 4(c) of
        the License and to reproduce the content of the NOTICE file.

     You may obtain a copy of the Apache License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the Apache License with the above modification is
     distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
     KIND, either express or implied. See the Apache License for the specific
     language governing permissions and limitations under the Apache License.

Subdivision Compatibility
-------------------------

.. contents::
   :local:
   :backlinks: none

This document highlights areas of compatibility with other software that deals
with subdivision surfaces (including older versions of OpenSubdiv).


Compatibility with OpenSubdiv 2.x
=================================

**Vertex Interpolation Options**

Since the various options are now presented through a new API (Sdc rather than
Hbr), based on the history of some of these options and input from interested
parties, the following changes have been implemented:

* Legacy modes of the *"smoothtriangle"* rule have been removed (as they
  were never actually enabled in the code). Values for *"TriangleSubdivision"*
  are now:

 * TRI_SUB_CATMARK - Catmull-Clark weights (default)
 * TRI_SUB_SMOOTH - "smooth triangle" weights

* The naming of the standard creasing method has been changed from *Normal*
  to *Uniform*.  Values for *"CreasingMethod"* are now:

 * CREASE_UNIFORM - the standard integer subtraction per level
 * CREASE_CHAIKIN - use Chaikin averaging around vertices

A conscious choice was also made to change the behavior of Chaikin creasing
in the presence of infinitely sharp edges (most noticeable at boundaries).
Previously the inclusion of the infinite sharpness values a semi-sharp edge
from decaying to zero.  Infinitely sharp edges are now excluded from the
Chaikin averaging yielding a much more predictable and desirable result.
Since this feature has received little use (only recently activated in
RenderMan) now seemed a good time to make the change.

**Face-varying Interpolation Options**

Face-varying interpolation was previously defined by a "boundary interpolation"
enum with four modes and an additional boolean "propagate corners" option,
which was little understood.  The latter was only used in conjunction with one
of the four modes, so it was effectively a unique fifth choice.  Deeper analysis
of all of these modes revealed unexpected and undesirable behavior in some common
cases -- to an extent that could not simply be changed -- and so additions have
been made to avoid such behavior.

All choices are now provided through a single "linear interpolation" enum --
intentionally replacing the use of "boundary" in its naming as the choice also
affects interior interpolation.  The naming now reflects the fact that
interpolation is constrained to be linear where specified by the choice.

All five of Hbr's original modes of face-varying interpolation are supported
(with minor modifications where Hbr was found to be incorrect in the presence
of semi-sharp creasing).  An additional mode has also been added to allow for
additional control around T-junctions where multiple disjoint face-varying
regions meet at a vertex.

The new values for the *"FVarLinearInterpolation"* are:

=========================  ==============================================================
FVAR_LINEAR_NONE           smooth everywhere (formerly "edge only")
FVAR_LINEAR_CORNERS_ONLY   sharpen corners only (new)
FVAR_LINEAR_CORNERS_PLUS1  sharpen corners and some junctions (formerly "edge and corner")
FVAR_LINEAR_CORNERS_PLUS2  sharpen corners and more junctions (formerly "edge and corner + propagate corner")
FVAR_LINEAR_BOUNDARIES     piecewise linear edges and corners (formerly "always sharp")
FVAR_LINEAR_ALL            bilinear interpolation (formerly "bilinear") (default)
=========================  ==============================================================

Aside from the two "corners plus" modes that preserve Hbr behavior, all other
modes are designed so that the interpolation of a disjoint face-varying region
is not affected by changes to other regions that may share the same vertex. So
the behavior of a disjoint region should be well understood and predictable
when looking at it in isolation (e.g. with "corners only" one would expect to
see linear constraints applied where there are topological corners or infinitely
sharp creasing applied within the region, and nowhere else).  This is not true
of the "plus" modes, and they are named to reflect the fact that more is taken
into account where disjoint regions meet.

These should be illustrated in more detail elsewhere in the documentation
(where exactly?) and with the example shapes and viewers.

**Hierarchical Edits**

Currently Hierarchical Edits have been marked as "extended specification" and
support for hierarchical features has been removed from the 3.0 release. This
decision allows for great simplifications of many areas of the subdivision
algorithms. If we can identify legitimate use-cases for hierarchical tags, we
will consider re-implementing them in future releases, as time and resources
allow.


Compatibility with RenderMan
============================

(More detail to come...)

Conscious deviations (incompatibilities):

- non-manifold topology
- choice of new face varying interpolation option ("corners only")
- use of Chaikin creasing with boundaries or infinitely sharp edges
- lack of hierarchical edits support

Corrections to Hbr's behavior (differences):

- smooth face varying interpolation in presence of dart vertices
- smooth face varying interpolation with fractional sharpness
- Chaikin creasing in general

Improved accuracy (differences):

- ordering of weight application (most prevalent with high-valence vertices)
- limit positions and tangents for arbitrary valence
- potential use of double precision masks
