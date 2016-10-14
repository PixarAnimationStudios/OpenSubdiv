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


Overview of Release 3.0
-----------------------

.. contents::
   :local:
   :backlinks: none

----

Release 3.0
===========

OpenSubdiv 3.0 represents a landmark release, with profound changes to the core
algorithms, simplified APIs, and streamlined GPU execution. Providing faster,
more efficient, and more flexible subdivision code remains our principal goal.
To achieve this, OpenSubdiv 3.0 introduces many improvements that constitute
a fairly radical departure from previous versions.

This document highlights some of the major changes that have gone in to the 3.0
release.

----

Subdivision Core (Sdc)
**********************

In consideration of past, present and future topological representations,
all low-level details fundamental to subdivision and the specific subdivision
schemes have been factored into a new low-level layer called **Sdc**
(SubDivision Core).  This layer encapsulates the full set of applicable
options, the formulae required to support semi-sharp creasing, the formulae
for the refinement masks of each subdivision scheme, etc.  As initially
conceived, its goal was often expressed as "separating the math from the mesh".

Sdc provides the low-level nuts and bolts to provide a subdivision
implementation consistent with OpenSubdiv. It is used by OpenSubdiv's 
libraries and may also be useful in providing an existing client's 
implementation with the details necessary to make that implementation 
consistent with OpenSubdiv.

----

Topology and Refinement
***********************

OpenSubdiv 3.0 introduces a new *intermediate* internal topological 
representation named **Vtr** (Vectorized Topology Representation).
Compared to the Hbr library used in previous versions, Vtr is much more 
efficient for the kinds of topological analysis required by Far and is more
flexible.  While Hbr is no longer used by OpenSubdiv, it will remain in
the source distribution for legacy and regression purposes.

**Faster Subdivision**

 A major focus of the 3.0 release is performance, and the improvement to
 the initial refinement of a mesh required for topological analysis is close
 to an order magnitude; often much more for uniform, but less for adaptive.

**Supporting for Non-manifold Topology**

 With topology conversion no longer constrained by Hbr, OpenSubdiv is no
 longer restricted to meshes of manifold topology.  With one exception
 (non-triangles with Loop subdivision), any set of faces and vertices that can
 be represented in common container formats such as Obj or Alembic can be
 represented and subdivided.  With future efforts to bring the functionality
 for the Loop scheme up to par with Catmark, that last remaining topological
 restriction will be removed.

**Simpler Conversion of Topology**

 Several entry-points are now available for client topology, rather than the
 single incremental assembly of an HbrMesh that previously existed.  The new
 topological relationships can be populated using either a high-level interface
 where simplicity has been emphasized, or a more complex lower-level interface
 for enhanced efficiency.

**Face Varying Topology**

 Previously, face-varying data was assigned by value to the vertex for each
 face, and whether or not the set of values around a vertex was continuous was
 determined by comparing these values later. In some cases this could result
 in two values that were not meant to be shared being "welded" together.

 Face-varying data is now specified topologically:  just as the vertex topology
 is defined from a set of **vertices** and integer references (indices) to
 these **vertices** for the corner of each face, face-varying topology is
 defined from a set of **values** and integer references (indices) to these 
 **values** for the corner of each face. So if values are to be considered
 distinct around a vertex, they are given distinct indices and no comparison
 of any data is ever performed.  Note that the number of **vertices** and
 **values** will typically differ, but since indices are assigned to the
 corners of all faces for both, the total number of indices assigned to all
 faces will be the same.
 
 This ensures that OpenSubdiv's face-varying topology matches what is often
 specified in common geometry container formats like Obj, Alembic and USD.
 Multiple "channels" of face-varying data can be defined and each is
 topologically independent of the others.

----

Limit Properties and Patches
****************************

A fundamental goal of OpenSubdiv is to provide an accurate and reliable
representation of the limit surface.  Improvements have been made both to the
properties (positions and tangents) at discrete points in the subdivision
hierarchy, as well as to the representations of patches used for the
continuous limit surface between them.

**Removed Fixed Valence Tables**

 Limit properties of extra-ordinary vertices are computed for arbitrary
 valence and new patch types no longer rely on small table sizes.  All tables
 that restricted the valence of a vertex to some relatively small table size
 have now been removed. 
 
 The only restriction on valence that exists is within the new topology
 representation, which restricts it to the size of an unsigned 16-bit integer
 (65,535).  This limit could also be removed, by recompiling with a certain
 size changed from 16- to 32-bits, but doing so would increase the memory cost
 for all common cases.  We feel the 16-bit limit is a reasonable compromise.

**Single Crease Patch**

 OpenSubdiv 3.0 newly implements efficient evaluation of semi-smooth
 creases(*) using single crease patches. With this optimization,
 high-order edge sharpness tags can be handled very efficiently for both
 computation time and memory consumption.

 (*) Niessner et al., Efficient Evaluation of Semi-Smooth Creases in
 Catmull-Clark Subdivision Surfaces. Eurographics (Short Papers). 2012.
 `<http://research.microsoft.com/en-us/um/people/cloop/EG2012.pdf>`_

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
 the previous max valence limit. Legacy Gregory patch still has a limitation
 of max valence (typically 24, depending on the hardware capability of
 GL_MAX_VARYING_VECTORS).

 Users are still encouraged to use models with vertices of low valence for
 both improved model quality and performance.

----

Faster Evaluation and Display
*****************************

OpenSubdiv 3.0 also introduces new data structures and algorithms that greatly
enhance performance for the common case of repeated evaluation both on the
CPU and GPU.

**Introducing Stencil Tables**

 OpenSubdiv 3.0 replaces the serialized subdivision tables with factorized
 stencil tables. The SubdivisionTables class of earlier releases contained
 a large number of data inter-dependencies, which incurred penalties from
 fences or force additional kernel launches. Most of these dependencies have now
 been factorized away in the pre-computation stage, yielding *stencil tables*
 (Far::StencilTable) instead.

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

 Similar to compute kernel simplification, this shader simplification has
 resulted in additional simplifications in the Osd layer.

----

Updated Source-Code Style
*************************

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

Documentation and Tutorials
***************************

The documentation has been reorganized and fleshed out. This release
introduces a number of new `tutorials <tutorials.html>`__. The tutorials
provide an easier entry point for learning the API than do the programs
provided in examples. The examples provide more fleshed out solutions and are
a good next step after the tutorials are mastered.

----

Additional Resources
====================

Porting Guide
*************

Please see the `Porting Guide <porting.html>`__ for help on how to port 
existing code written for OpenSubdiv 2.x to the new 3.0 release.

----

Subdivision Compatibility
*************************

The 3.0 release has made some minor changes to the subdivision specification
and rules.  See `Subdivision Compatibility <compatibility.html>`__ for a
complete list.
