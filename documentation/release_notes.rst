..
     Copyright 2013 Pixar

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


3.0.0.Beta Release Notes
------------------------

.. contents::
   :local:
   :backlinks: none

----

Release 3.0
===========

OpenSubdiv 3.0 represents a landmark release, with very profound changes to the
core algorithms. While providing faster, more efficient, and more flexible
subdivision code remains our principal goal, OpenSubdiv 3.0 introduces many
improvements that constitute a fairly radical departure from our previous
versions.

----

Improved performance
********************

Release 3.0.0 of OpenSubdiv introduces a new set of data structures and
algorithms that greatly enhance performance over previous versions.

This release focuses mostly on the CPU side, and  should provide
"out-of-the-box" speed-ups close to an order of magnitude for topology
refinement and analysis (both uniform and adaptive). Please note: a very large
portion of the 2.x code base has been completely replaced or deprecated.

On the GPU side, the replacement of subdivision tables with stencils allows
us to remove several bottlenecks in the Compute area that can yield as much as
4x faster interpolation on CUDA platforms. At the same time, stencils also
reduce the dozens of kernel launches required per primitive to a single one (this
was a known issue on certain mobile platforms). Compute calls batching is now
trivial.

We will continue releasing features and improvements throughout the release
cycle, both to match the feature set of previous releases, and to further the
general optimization strategy described above.

----

New topology entry-points
*************************

OpenSubdiv 3.0 introduces several new entry-points for client topology. Previous
releases forced client applications to define and populate instances of an Hbr
half-edge topology representation. For many applications, this representation
was both redundant and inefficient.

OpenSubdiv 3.0 introduces a new *intermediate* topological representation, named
**Vtr** (Vectorized Topology Representation). The topological relationships
held by Vtr can be populated using either a high-level interface where simplicity
has been emphasized, or a lower-level interface for enhanced efficiency. Vtr is
much more efficient for the kinds of topological analysis required by Far and
additionally is more flexible in that it supports the specification of
non-manifold topology.

As a result, Hbr is no longer a core API of OpenSubdiv. While the code is marked
as deprecated, it will remain in the source distribution for legacy and
regression purposes.

The documentation for Vtr can be found `here <vtr_overview.html>`__

----

New treatment of face-varying data
**********************************

With Hbr no longer being the entry point for client-code, OpenSubdiv 3.0 has to
provide a new interface for face-varying data. Previous versions required
face-varying data to be assigned by value to the vertex for each face, and
whether or not the set of values around a vertex was continuous was determined
by comparing these values later. In some cases this could result in two values
that were not meant to be shared being "welded" together.

Face-varying data is now specified topologically. Just as the vertex topology
is defined from a set of vertices and integer references to these vertices for
the vertex of each face, face-varying topology is defined from a set of values
and integer references to these values for the vertex of each face. So if
values are to be considered distinct around a vertex, they are given distinct
indices and no comparison of values is ever performed.

This ensures that OpenSubdiv's face-varying topology matches what is specified
in common geometry container formats like Obj or Alembic. It also allows for
more efficient processing of face-varying values during refinement, and so the
cost of interpolating a set of face-varying data should now be little more than
the cost of interpolating a similar set of vertex data (depending on the number
of distinct face-varying values versus the number of vertices).

----

Subdivision Core (Sdc)
**********************

In consideration of the existing representations (Hbr and Vtr), all low-level
details fundamental to subdivision and the specific subdivision schemes have
been factored into a new low-level layer (the lowest) called Sdc. This layer
encapsulates the full set of applicable options, the formulae required to
support semi-sharp creasing, the formulae for the refinement masks of each
subdivision scheme, etc.

Sdc provides the low-level nuts and bolts to provide a subdivision
implementation consistent with OpenSubdiv. It is used internally by Vtr and
Far but can also provide client-code with an existing implementation of their
own with the details to make that implementation consistent with OpenSubdiv.

The documentation for Sdc can be found `here <sdc_overview.html>`__

----

Introducing Stencil Tables
**************************

OpenSubdiv 3.0 replaces the serialized subdivision tables with factorized
stencil tables. Subdivision tables as implemented in 2.x releases still contain
a fairly large amount of data inter-dependencies, which incur penalties from
fences or force additional kernel launches. Most of these dependencies have now
been factorized away in the pre-computation stage, yielding *stencil tables*
instead.

Stencils remove all data dependencies and simplify all the computations into a
single trivial kernel. This simplification results in a faster pre-computation
stage, faster execution on GPU, with fewer driver overheads. The new stencil
tables Compute back-end is supported on all the same platforms as previous
releases (except GCD).

----

New Source-Code Style
*********************

OpenSubdiv 3.0 replaces naming prefixes with C++ namespaces for all API layers,
bringing the source style more in line with contemporary specifications
(mostly inspired from the `Google C++ Style Guide
<http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml>`__).

The large-scale changes introduced in this release generally break compatibility
with existing client-code. However, this gives us the opportunity to effect
some much needed updates to our code-style guidelines and general conventions,
throughout the entire OpenSubdiv code-base. We are hoping to drastically
improve the quality, consistency and readability of the source code.

While the bulk of code refactoring is mostly in place, we are still tweaking
some of the finer details. After this Beta release we are not anticipating any
further significant changes.

----

New Tutorials
*************

Documentation has been re-organized and fleshed out (although there is still a
lot of work remaining). Because the "code examples" have been generally overly
complicated, with this release we are introducing a number of new `tutorials
<tutorials.html>`__. We are trying to keep these tutorials as simple as
possible, with no external dependencies (although some of them generate OBJ or
Maya MEL scripts as a way of visualizing the output). We are planning on releasing
more tutorials as time and resources allow.

----

Changes to the Subdivision Specification
========================================

The refactoring of OpenSubdiv 3.0 data representations presents a unique
opportunity to revisit some corners of the subdivision specification and
remove or update some legacy features.

Vertex Interpolation Rules
**************************

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

      The current implementation of the *"Chaikin"* rule shows small
      numerical differences with results obtained from Hbr in 2.x releases.
      Considering that the feature is rarely used and that the current
      implementation is likely the more correct one, we consider the
      current implementation as *the standard*. Aside from a conscious
      deviation at boundaries (where infinitely sharp creases are now excluded
      from the averaging in 3.0 to allow proper decay of a semi-sharp edge
      to 0), all other deviations found have been identified as flaws in the
      implementation of 2.x (and are not easily corrected).
      
      We will review input from the community on this matter during the Beta
      release cycle.

In all cases, features in active use are not being removed but simply
re-expressed in what is hoped to be a clearer interface.

We will welcome feedback and constructive comments as we deploy these changes.
We hope to converge toward a general consensus and lock these APIs by the end
of Beta cycle.

Face-varying Interpolation Rules
********************************

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

    * FVAR_LINEAR_NONE          - smooth everywhere ("edge only")
    * FVAR_LINEAR_CORNERS_ONLY  - sharpen corners only
    * FVAR_LINEAR_CORNERS_PLUS1 - ("edge corner")
    * FVAR_LINEAR_CORNERS_PLUS2 - ("edge and corner + propagate corner")
    * FVAR_LINEAR_BOUNDARIES    - piecewise linear edges and corners ("always sharp")
    * FVAR_LINEAR_ALL           - bilinear interpolation ("bilinear") (default)

Aside from the two "corners plus" modes that preserve Hbr behavior, all other
modes are designed so that the interpolation of a disjoint face-varying region
is not affected by changes to other regions that may share the same vertex. So
the behavior of a disjoint region should be well understood and predictable
when looking at it in isolation (e.g. with "corners only" one would expect to
see linear constraints applied where there are topological corners or infinitely
sharp creasing applied within the region, and nowhere else).  This is not true
of the "plus" modes, and they are named to reflect the fact that more is taken
into account where disjoint regions meet.

These are illustrated in more detail elsewhere in the documentation, the tutorials
and the example shapes.

Hierarchical Edits
******************

Currently Hierarchical Edits have been marked as "extended specification" and
support for hierarchical features has been removed from the 3.0 release. This
decision allows for great simplifications of many areas of the subdivision
algorithms. If we can identify legitimate use-cases for hierarchical tags, we
will consider re-implementing them in future releases, as time and resources
allow.

----

Beta Release Notes
==================

Our intentions as open-source developers is to give as much access to our code,
as early as possible, because we value and welcome the feedback from the
community.

With the 'Beta' release cycle, we hope to give stake-holders a time-window to
provide feedback on decisions made and changes in the code that may impact
them. Our Beta code is likely not feature-complete yet, but the general
structure and architectures will be sufficiently locked in place for early
adopters to start building upon these releases.

Within 'Master' releases, we expect APIs to be backward compatible so that
existing client code can seamlessly build against newer releases. Changes
may include bug fixes as well as new features.

.. container:: notebox

    **Beta Features**

    The following is a short list of features that hopefully will land before
    the master release:

        #. Non-linear Face-varying Patches:
           While the fundamental refinement and interpolation of face-varying
           data is correct, it has been and remains linearly approximated in
           the patches created in Far that are most used for evaluation and
           display.  We want to update the patch tables to support non-linear
           patches for the face-varying data.

        #. Improved Robustness with Non-Manifold Topology:
           With the replacement of Hbr with Vtr in 3.0, many non-manifold
           topologies can be represented and effectively subdivided.  One
           situation that was deferred is that of a "degenerate edge", i.e an
           edge that has the same vertex at both ends.  Plans are to update
           the refinement code within Vtr to do something reasonable in these
           cases.


----

3.x Release Cycle RoadMap
=========================

Within the 3.x release cycle we would like to continue to address many of the
issues related to scaling the application of subdivision surfaces to large amounts
of primitives within typical graphics pipelines, as well as complete other
functionality that has long been missing from evaluation and display.

Enabling workflows at larger scales will require improvements on several fronts:

* Handle more primitives, but with fewer overheads:

    * Reduce Compute kernel launches,which we will achieve using stencils instead
      of subdivision tables
    * Reduce Draw calls by addressing the combinatorial explosion of tessellation
      shaders
    * Provide back-ends for next-gen APIs (D3D12, Mantle, Metal, GL 5.x)

* Handle more semi-sharp creases: feature isolation needs to become much more
  efficient to allow for complete creative freedom in using the feature.
* Faster topology analysis

As for missing functionality, as the potential standard for evaluation and display
of subdivision surfaces, OpenSubdiv is still lacking in its support of subdivision
schemes other than Catmark -- specifically Loop.  Ultimately the same level of
performance and functionality achieved with Catmark should be available for Loop,
which is more effective in dealing with triangle-based meshes.  With the refactoring
of the core refinement code in 3.0, much more of the supporting code for the schemes
can be shared so we have already reduced the effort to bring Loop up to par with
Catmark.  We hope to take steps in this direction in an upcoming 3.x release.


Release 2.x
===========

`Previous releases <release_notes_2x.html>`_
