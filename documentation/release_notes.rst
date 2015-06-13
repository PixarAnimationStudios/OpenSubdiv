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


3.0.0 Release Notes
------------------------

.. contents::
   :local:
   :backlinks: none

----

Release 3.0
===========

OpenSubdiv 3.0 represents a landmark release, with profound changes to the core
algorithms, simplified APIs, and streamlined GPU execution. Providing
faster, more efficient, and more flexible subdivision code remains our
principal goal. To achieve this, OpenSubdiv 3.0 introduces many
improvements that constitute a fairly radical departure from previous
versions.

----

Improved Performance
********************

Release 3.0.0 of OpenSubdiv introduces a new set of data structures and
algorithms that greatly enhance performance over previous versions.

**Faster Subdivision**

 A major focus of the 3.0 release is performance. It should provide
 "out-of-the-box" speed-ups close to an order of magnitude for topology refinement
 and analysis (both uniform and adaptive).

**Introducing Stencil Tables**

 OpenSubdiv 3.0 replaces the serialized subdivision tables with factorized
 stencil tables. Subdivision tables as implemented in 2.x releases still contain
 a fairly large amount of data inter-dependencies, which incur penalties from
 fences or force additional kernel launches. Most of these dependencies have now
 been factorized away in the pre-computation stage, yielding *stencil tables*
 instead.

 Stencils remove all data dependencies and simplify all the computations into a
 single trivial kernel. This simplification results in a faster pre-computation
 stage, faster execution on GPU, with less driver overhead. The new stencil
 tables Compute back-end is supported on all the same platforms as previous
 releases (except GCD).

**Faster, Simpler GPU Kernels**

 On the GPU side, the replacement of subdivision tables with stencils greatly 
 reduces bottlenecks in compute, yielding as much as a 4x interpolation speed-up. 
 At the same time, stencils reduce the complexity of interpolation to a single 
 kernel launch per primitive, a critical improvement for mobile platforms.

 As a result of these changes, compute batching is now trivial, which in turn
 enabled API simplifications in the Osd layer.

**Unified Adaptive Shaders**

 Adaptive tessellation shader configurations have been greatly simplified. The 
 number of shader configurations has been reduced from a combinatorial per-patch 
 explosion down to a constant two global configurations. This massive improvement 
 over the 2.x code base results in significantly faster load times and a reduced
 per-frame cost for adaptive drawing.

 Similar to compute kernel simplification, this shader simplification has resulted
 in additional simplifications in the Osd layer.

**Single Crease Patch**

 OpenSubdiv 3.0 newly implements efficient evaluation of semi-smooth
 creases(*) using single crease patches. With this optimization,
 high-order edge sharpness tags can be handled very efficiently for both
 computation time and memory consumption.

 (*) Niessner et al., Efficient Evaluation of Semi-Smooth Creases in
 Catmull-Clark Subdivision Surfaces. Eurographics (Short Papers). 2012.

**New Irregular Patch Approximations**

 While "legacy" Gregory patch support is still available, we have introduced
 several new options for representing irregular patches: Legacy Gregory, fast
 Gregory Basis stencils, and BSpline patches. Gregory basis stencils provide
 the same high quality approximation of Legacy Gregory patches, but execute
 considerably faster with a simpler GPU representation. While BSpline patches
 are not as close an approximation as Gregory patches, they enable an entire
 adaptively refined mesh to be drawn with screen space tessellation via a
 single global shader configuration (Gregory Basis patches require one
 additional global shader configuration).

 The new implementations of the GregoryBasis and BSpline approximations relax
 the previous max valence limit. Users are still encouraged to use models with
 vertices of low valence for both improved model quality and performance.

----

Topological Flexibility
***********************

Given repeated limitations experienced with Hbr as the primary topological
entry point, OpenSubdiv 3.0 introduces a new *intermediate* topological
representation, named **Vtr** (Vectorized Topology Representation).  Vtr is
much more efficient for the kinds of topological analysis required by Far
and additionally is more flexible.  As a result, Hbr is no longer a core API
of OpenSubdiv. While the code is marked as deprecated, it will remain in the
source distribution for legacy and regression purposes.

Though Vtr is insulated from public access by a topology container in Far (the
Far::TopologyRefiner) -- allowing it to be enhanced in future independent of the
public API -- documentation for Vtr can be found `here <vtr_overview.html>`__

**Now Supporting Non-manifold Topology**

 With topology conversion no longer constrained by Hbr, OpenSubdiv is no
 longer restricted to meshes of manifold topology.  With one exception
 (non-triangles with Loop subdivision), any set of faces and vertices that can
 be represented in common container formats such as Obj or Alembic can be
 represented and subdivided.  With future efforts to bring the functionality
 for the Loop scheme up to par with Catmark, that last remaining topological
 restriction will hopefully be removed.

**Simpler Conversion of Topology**

 Several entry-points are now available for client topology, rather than the
 single incremental assembly of an HbrMesh that previously existed.  The new
 topological relationships can be populated using either a high-level interface
 where simplicity has been emphasized, or a more complex lower-level interface
 for enhanced efficiency.

**Face Varying Topology**

 Previously, face-varying data had to be assigned by value to the vertex for
 each face, and whether or not the set of values around a vertex was
 continuous was determined by comparing these values later. In some cases this
 could result in two values that were not meant to be shared being "welded"
 together.

 Face-varying data is now specified topologically. Just as the vertex topology
 is defined from a set of **vertices** and integer references (indices) to
 these **vertices** for the corner of each face, face-varying topology is
 defined from a set of **values** and integer references (indices) to these 
 **values** for the corner of each face. So if values are to be considered
 distinct around a vertex, they are given distinct indices and no comparison
 of values is ever performed.  Note that the number of **vertices** and
 **values** will typically differ, but since indices are assigned to the
 corners of all faces for both, the total number of indices assigned to all
 faces will be the same.
 
 This ensures that OpenSubdiv's face-varying topology matches what is specified
 in common geometry container formats like Obj or Alembic. It also allows for
 more efficient processing of face-varying values during refinement, and so the
 cost of interpolating a set of face-varying data should now be little more than
 the cost of interpolating a similar set of vertex data (depending on the number
 of distinct face-varying values versus the number of vertices).

**No more fixed valence tables**

 All tables that restricted the valence of a vertex to some relatively small
 table size have now been removed.  Limit properties of extra-ordinary vertices
 are computed for arbitrary valence and new patch types no longer rely on small
 table sizes.
 
 The only restriction on valence that exists is within the new topology
 representation, which restricts it to the size of an unsigned 16-bit integer
 (65,535).  This limit could also be removed, by recompiling with a certain
 size changed from 16- to 32-bits, but doing so would increase the memory cost
 for all common cases.  We feel the 16-bit limit was a reasonable compromise.

----

Subdivision Core (Sdc)
**********************

In consideration of the past (Hbr), present (Vtr) and future representations,
all low-level details fundamental to subdivision and the specific subdivision
schemes have been factored into a new low-level layer (the lowest) called Sdc.
This layer encapsulates the full set of applicable options, the formulae
required to support semi-sharp creasing, the formulae for the refinement masks
of each subdivision scheme, etc.

Sdc provides the low-level nuts and bolts to provide a subdivision
implementation consistent with OpenSubdiv. It is used internally by Vtr and
Far but can also provide client-code with an existing implementation of their
own with the details to make that implementation consistent with OpenSubdiv.

The documentation for Sdc can be found `here <sdc_overview.html>`__

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

RC2 Release Notes
==================

Release Candidate 2 includes following bug fixes and updates.

 * Documentation updates

 * API fixes

     * Fixed a LimitStencilTableFactory bug, which returns an invalid table

     * PatchParam encoding changed to support refinement levels up to 10

 * Build issue fixes

     * Added Xinerama link dependency

     * Fixed MSVC 32bit build problem

     * Fixed minor cmake issues

 * Examples/Tutorials fixes and updates

     * Fixed glViewer/farViewer stability bugs

     * far_tutorial_3 updates for the multiple face-varying channels

     * maya example plugin interpolates a UV channel and a vertex color channel

RC1 Release Notes
==================

Release Candidate 1 is a short-lived release intended for stabilization before
the official 3.0 release.  The APIs are now locked restricted to bug fixes and
documentation changes.

It's been a very active beta cycle and we've received and incorporated great
feedback. A larger than expected subset of the API has changed since the beta
release, to the overall benefit of the library. These changes lay a strong
foundation for future, stable 3.0 point releases.

Notable API changes in between 3.0-beta and 3.0-RC1 include:

 * Far::TopologyRefiner was split into several classes to clarify and focus
   the API.  Specifically, all level-related methods were moved to a new
   class Far::TopologyLevel for inspection of a level in the hierarchy.
   Similarly, all methods related to client "primvar" data, i.e. the suite
   of Interpolate<T>() and Limit<T>() methods, were moved to a new class
   Far::PrimvarRefiner.
   
 * Interpolation of Vertex and Varying primvars in a single pass is no longer 
   supported. As a result, AddVaryingWithWeight() is no longer required and 
   InterpolateVarying() must be called explicitly, which calls AddWithWeight(),
   instead of AddVaryingWithWeight().
   
 * The Osd layer was largely refactored to remove old designs that were
   originally required to support large numbers of kernel and shader
   configurations (thanks to stencils and unified shading).

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

Previous 2.x Release Notes
==========================

`Previous releases <release_notes_2x.html>`_
