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


Release Notes
-------------

.. contents::
   :local:
   :backlinks: none

----

General 3.x RoadMap
===================

Within the 3.x release cycle we would like to address first and foremost many of
the issues related to scaling the application of subdivision surfaces to large
amounts of primitives within typical graphics pipelines.

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


Release 3.0
===========

OpenSubdiv 3.0 represents a landmark release, with very profound changes to the
core algorithms. While providing faster, more efficient, and more flexible
subdivision code remains our principal goal, OpenSubdiv 3.0 introduces many
improvements that constitute a fairly radical departures from our previous
code releases.

Improved performance
********************

Release 3.0.0 of OpenSubdiv introduces a set of new data structures and
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

We will continue releasing features and improvements through the release cycle,
both to match the feature set of previous releases, and to further the general
optimization strategy described above.

New topology entry-points
*************************

OpenSubdiv 3.0 introduces several new entry-points for client topology. Previous
releases force client applications to define and populate instances of an Hbr
half-edge topology representation. For many applications, this representation is
both redundant and inefficient.

OpenSubdiv 3.0 introduces a new *intermediate* topological representation, named
**Vtr** (Vectorized Topology Representation). The topological relationships held
by Vtr can populated using either a high-level interface where simplicity has
been emphasized, or a lower-level interface for enhanced efficiency (still under
construction).  Vtr is much more efficient for the kinds of topological analysis
required by Far and additionally is more flexible in that it supports the
specification of non-manifold topology.

As a result, Hbr is no longer a core API of OpenSubdiv. While the code is marked
as deprecated, it will remain in the source distribution for legacy and
regression purposes.

The documentation for Vtr can be found `here <vtr_overview.html>`__

New treatment of face-varying data
**********************************

With Hbr no longer being the entry point for client-code, OpenSubdiv 3.0 has to
provide a new interface to face-varying data. Previous versions required
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

Subdivision Core (Sdc)
**********************

In consideration of the existing representations (Hbr and Vtr), all low-level
details fundamental to subdivision and the specific subdivision schemes has been
factored into a new low-level layer (the lowest) called Sdc. This layer
encapsulates the full set of applicable options, the formulae required to
support semi-sharp creasing, the formulae for the refinement masks of each
subdivision scheme, etc.

Sdc provides the low-level nuts and bolts to provide a subdivision
implementation consistent with OpenSubdiv. It is used internally by Vtr but can
also provide client-code with an existing implementation of their own with the
details to make that implementation consistent with OpenSubdiv.

The documentation for Sdc can be found `here <sdc_overview.html>`__

Changes to the Subdivision Specification
****************************************

The refactoring of OpenSubdiv 3.0 data representations presents a unique
opportunity to revisit some corners of the subdivision specification and
remove or update some legacy features.

Interpolation Rules
+++++++++++++++++++

Since the various options are now presented through a new API (Sdc rather than
Hbr), based on the history of some of these options and input from interested
parties, the following changes are being investigated:

    * the creasing method 'Normal' has been renamed 'Uniform'
    * considering renaming the Sdc 'boundary interpolation' enum and its choices
    * same as above for the 'face varying boundary interpolation' enum in Sdc

In these cases, features are not being removed but simply re-expressed in what
is hoped to be a clearer interface.

We will welcome feedback and constructive comments as we deploy these changes.
We hope to converge toward a general consensus and lock these APIs by the end
of Beta cycle.

Hierarchical Edits
++++++++++++++++++

Currently Hierarchical Edits have been marked as "extended specification" and
support for hierarchical features has been removed from the 3.0 release. This
decision allows for great simplifications of many areas of the subdivision
algorithms. If we can identify legitimate use-cases for hierarchical tags, we
will consider re-implementing them in future releases, as time and resources
allow.

Introducing Stencil Tables
**************************

OpenSubdiv 3.0 replaces the serialized subdivision tables with factorized
stencil tables. Subdivision tables as implemented in 2.x releases still contain
a fairly large amount of data inter-dependencies, which incur penalties from
fences or force addition kernel launches. Most of these dependencies have now
been factorized away in the pre-computation stage, yielding *stencil tables*
instead.

Stencils remove all data dependencies and simplify all the computations into a
single trivial kernel. This simplification results in a faster pre-computation
stage, faster execution on GPU, and fewer driver overheads. The new stencil
tables Compute back-end is supported on all the same platforms as previous
releases (except GCD).

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

New Tutorials
*************

Documentation has been re-organized and fleshed out (although there is still a
lot of work remaining). Because the "code examples" have been generally overly
complicated, with this release we are introducing a number of new `tutorials
<tutorials.html>`__. We are trying to keep these tutorials as simple as
possible, with no external dependencies (although some of them generate OBJ or
Maya MEL scripts as a way of visualizing the output). We are planning on releasing
more tutorials as time and resources allow.

Alpha Release Notes
===================

Our intentions as open-source developers is to give as much access to our code,
as early as possible, because we value and welcome the feedback from the
community.

The 'alpha' release moniker means to us that our code is still far from being
finalized. Although we are now close from being feature complete, our
public-facing interfaces are still subject to change. Therefore, we do not
recommend this version of OpenSubdiv be used in client applications until both
features and interfaces have been finalized in an official 'Beta' Release.

.. container:: notebox

    **Alpha Issues**

    The following is a short list of features and issues that are still in
    development or are likely to change during the alpha cycle:

        #. Support for Loop and Bilinear schemes (consistent with 2.x):
           Currently only the Catmull-Clark subdivision scheme is supported.
           Parity of support with 2.x versions for the Loop and Bilinear schemes
           will be re-established before Beta release.

        #. Refactor Far::TopologyRefiner interpolation functions:
           Templated interpolation methods such as Interpolate<T>(),
           InterpolateFaceVarying<T>(), Limit<T>() are not finalized yet. Both
           the methods prototypes as well the interface required for **T** are
           likely to change before Beta release.

        #. Face-varying interpolation rules:
           Currently, all 4 legacy modes of face-varying interpolation are
           supported for *interior* vertices, but not for *boundary* vertices.
           Work is currently underway to match and extend Hbr's boundary
           interpolation rules as implemented in 2.x. The new code will need
           to be tested and validated.

        #. Limit Masks:
           Currently, Sdc generates weighted masks to interpolate *vertex* and
           *face-varying* primvar data between subdivision levels. We want to
           add functionality to evaluate closed-form evaluation of weight masks
           to interpolate primvar data at the limit.

        #. Implement arbitrary and discrete limit stencils:
           Subdivision tables have been replaced with discrete vertex stencils.
           We would like to add functionality for stencils to push these
           vertices to the limit, as well as generate stencils for arbitrary
           locations on the limit surface (a feature currently available in
           2.x). This work is contingent on the implementation of limit masks.

        #. Tagging and recognition of faces as Holes:
           A solution for tagging faces as *"holes"* needs to be implemented to
           match functionality from 2.x releases.

        #. Topology entry-point API:
           The *advanced* topology entry point interface in
           Far::TopologyRefinerFactory is not final yet. Some protected
           accessors are likely to be renamed, added or removed before Beta
           release.

        #. *"Chaikin"* Rule:
           The current implementation of the *Chaikin* rule shows small
           numerical differences with results obtained from Hbr in 2.x releases.
           Considering that the feature is rarely used and that the current
           implementation is likely the more correct one, we are considering
           declaring the current implementation as *the standard*. We will
           review input from the community on this matter during Alpha and Beta
           release cycles.

        #. Code Style & Refactoring:
           While the bulk of code refactoring is mostly in place, we are still
           tweaking some of the finer details. Further changes to code styling
           and conventions should be expected throughout the Alpha release
           cycle.


Release 2.x
===========

`Previous releases <release_notes_2x.html>`_
