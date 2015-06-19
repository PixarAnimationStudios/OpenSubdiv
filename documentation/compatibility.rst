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

The refactoring of OpenSubdiv 3.0 data representations presented a unique
opportunity to revisit some corners of the subdivision specification and
remove or update some legacy features.  Below are some of the changes made that
may affect compatibility with other software and previous versions of
OpenSubdiv.

**Face-varying Interpolation Options**

Face-varying interpolation options have been consolidated into a single enum
with one additional choice new to 3.0.  No functionality from 2.x has been
removed -- just re-expressed in a simpler and more comprehensible form.

Face-varying interpolation was previously defined by a "boundary interpolation"
enum with four modes and an additional boolean "propagate corners" option,
which was little understood, i.e.:

* void HbrMesh::SetFVarInterpolateBoundarMethod(InterpolateBoundaryMethod) const;

* void HbrMesh::SetFVarPropagateCorners(bool) const;

The latter was only used in conjunction with one
of the four modes ("edge and corner"), so it was effectively a unique fifth
choice.  Closer inspection of all of these modes also revealed some unexpected
and undesirable behavior in some common cases -- to an extent that could not
simply be changed -- and so an additional mode was added to avoid such behavior.

All choices are now provided through a single "linear interpolation" enum,
desribed and illustrated in more detail in the overview of
`Face-Varying Interpolation <subdivision_surfaces.html#face-varying-interpolation-rules>`__.
The use of "boundary" in the name of the enum was intentionally removed
as the choice also affects interior interpolation.  The new use of "linear"
is now intended to reflect the fact that interpolation is constrained to be
linear where specified by the choice applied.

All five of Hbr's original modes of face-varying interpolation are supported
(with minor modifications where Hbr was found to be incorrect in the presence
of semi-sharp creasing).  An additional mode ("corners only") has also been
added to avoid some of the undesired side-effects of some existing modes.

The new values for the *"Sdc::Options::FVarLinearInterpolation"* enum and its
equivalents for HbrMesh's InterpolateBoundaryMethod and PropagateCorners flag
are as follows (ordered such that the set of linear constraints applied is
always increasing -- from completely smooth to completely linear):

============================ ================================== =========================
Sdc FVarLinearInterpolation  Hbr FVarInterpolateBoundaryMethod  Hbr FVarPropogateCorners
============================ ================================== =========================
FVAR_LINEAR_NONE             k_InterpolateBoundaryEdgeOnly      N/A (ignored)
FVAR_LINEAR_CORNERS_ONLY     N/A                                N/A
FVAR_LINEAR_CORNERS_PLUS1    k_InterpolateBoundaryEdgeAndCorner false
FVAR_LINEAR_CORNERS_PLUS2    k_InterpolateBoundaryEdgeAndCorner true
FVAR_LINEAR_BOUNDARIES       k_InterpolateBoundaryAlwaysSharp   N/A (ignored)
FVAR_LINEAR_ALL              k_InterpolateBoundaryNone          N/A (ignored)
============================ ================================== =========================

Aside from the two "corners plus" modes that preserve Hbr behavior, all other
modes are designed so that the interpolation of a disjoint face-varying region
is not affected by changes to other regions that may share the same vertex. So
the behavior of a disjoint region should be well understood and predictable
when looking at it in isolation (e.g. with "corners only" one would expect to
see linear constraints applied where there are topological corners or infinitely
sharp creasing applied within the region, and nowhere else).  This is not true
of the "plus" modes, and they are named to reflect the fact that more is taken
into account where disjoint regions meet.

Differences between the modes can be seen in the regression shapes with the
prefix "catmark_fvar" -- which were specifically created for that purpose.

**Vertex Interpolation Options**

Since the various options are now presented through a new API (Sdc rather than
Hbr), based on the history of some of these options and input from interested
parties, the following changes have been implemented:

* The naming of the standard creasing method has been changed from *Normal*
  to *Uniform*.  Values for *"Sdc::Options::CreasingMethod"* are now:

============== ====================================
CREASE_UNIFORM standard integer subtraction per level (default)
CREASE_CHAIKIN Chaikin (non-uniform) averaging around vertices
============== ====================================

* Legacy modes of the *"smoothtriangle"* rule have been removed (as they
  were never actually enabled in the code). Values for
  *"Sdc::Options::TriangleSubdivision"* are now:

=============== =================
TRI_SUB_CATMARK Catmull-Clark weights (default)
TRI_SUB_SMOOTH  "smooth triangle" weights
=============== =================

These should have little impact since one is a simple change in terminology
as part of a new API while the other was removal of an option that was never
used.

**Change to Chaikin creasing method**

In the process of re-implementing the Chaikin creasing method, observations
lead to a conscious choice to change the behavior of Chaikin creasing in the
presence of infinitely sharp edges (most noticeable at boundaries).

Previously the inclusion of the infinite sharpness values in the averaging
of sharpness values that takes place around a vertex would prevent a
semi-sharp edge from decaying to zero.  Infinitely sharp edges are now
excluded from the Chaikin (non-uniform) averaging yielding a much more
predictable and desirable result, e.g. where the sharpness assignment is
actually uniform at such a vertex, the result will now behave the same as
the Uniform method.

Since this feature has received little use (only recently activated in
RenderMan) now seemed the best time to make the change before more widespread
adoption.

**Hierarchical Edits**

While extremely powerful, Hierarchical Edits come with additional maintenance
and implementation complexity.  Support for them in popular interchange formats
and major DCC applications has either been dropped or was never implemented.
As a result, the need for Hierarchical Edits is too limited to justify the cost
and support for them and they have therefore been removed from the 3.0 release
of OpenSubdiv. Dropping support for Hierarchical Edits allows for significant
simplifications of many areas of the subdivision algorithms.

While the 3.0 release does not offer direct support for Hierarchical Edits,
the architectural changes and direction of 3.0 still facilitate the application
of the most common value edits for those wishing to use them -- though not
always in the same optimized context.  Of course, support for Hierarchical
Edits in the future will be considered based on demand and resources.

**Non-Manifold Topology**

OpenSubdiv 2.x and earlier was limited to dealing with meshes whose topology
was manifold -- a limitation imposed by the use of Hbr.  With 3.0 no longer
using Hbr, the manifold restriction has also been removed.

So 3.0 will support a superset of the meshes supported by 2.x.  Non-manifold
meshes that are acceptible to 3.0 however will likely not work with 2.x or
earlier.


Compatibility with RenderMan
============================

Since RenderMan and OpenSubdiv versions prior to 3.0 share a common library
(Hbr), some differences between RenderMan and OpenSubdiv are covered in the
section of compatibility with OpenSubdiv 2.x.

While there are features between RenderMan and OpenSubdiv that are not
compatibile, there are also other differences that may be presence due to
differences in the implementations of similar features.  Given the common
past between RenderMan and OpenSubdiv (and potential expectations arising
from it), mention of both is warranted here.

Incompatibilities
+++++++++++++++++

OpenSubdiv and RenderMan will be incompatible when certain features are used
that are not common to both.  While described in the 2.x compatibility 
section, they are listed briefly here.

**OpenSubdiv Features Not Supported by RenderMan**

* non-manifold meshes

* choice of the "corners only" face varying interpolation option


**RenderMan Features Not Supported by OpenSubdiv**

* hierarchical edits (as of 3.0, may be subsequently re-introduced)


Other Differences
+++++++++++++++++

**Smooth Face-Varying Interpolation with Creasing**

There have been two discrepancies noted in the way that face-varying data is
interpolated smoothly in the presense of creases:

* interpolation around a dart vertex

* lack of blending for fractional sharpness, i.e. only integer sharpness

**The Chaikin Creasing Method**

* use of Chaikin creasing with boundaries or infinitely sharp edges

* subtle shape differences due to Hbr's use of "predictive sharpness"

**Numerical Precision**

* improved with OSD's ordering of weight application (most prevalent with
  high-valence vertices)
