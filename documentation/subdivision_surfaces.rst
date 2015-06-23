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


Subdivision Surfaces
--------------------

.. contents::
   :local:
   :backlinks: none

----

Introduction
============

The most common way to model complex smooth surfaces is by using a patchwork of
bicubic patches such as BSplines or NURBS.

.. image:: images/torus.png
   :align: center
   :height: 200

However, while they do provide a reliable smooth limit surface definition,
bi-cubic patch surfaces are limited to 2-dimensional topologies, which only
describe a very small fraction of real-world shapes. This fundamental
parametric limitation requires authoring tools to implement at least the
following functionalities:

    - smooth trimming
    - seams stitching

Both trimming and stitching need to guarantee the smoothness of the model both
spatially and temporally as the model is animated. Attempting to meet these
requirements introduces a lot of expensive computations and complexity.

Subdivision surfaces on the other hand can represent arbitrary topologies, and
therefore are not constrained by these difficulties.

----

Arbitrary Topology
==================

A subdivision surface, like  a parametric surface, is described by its control
mesh of points. The surface itself can approximate or interpolate this control
mesh while being piecewise smooth. But where polygonal surfaces require large
numbers of data points to approximate being smooth, a subdivision surface is
smooth - meaning that polygonal artifacts are never present, no matter how the
surface animates or how closely it is viewed.

Ordinary cubic B-spline surfaces are rectangular grids of tensor-product
patches. Subdivision surfaces generalize these to control grids with arbitrary
connectivity.

.. raw:: html

    <center>
      <p align="center">
        <IMG src="images/tetra.0.png" style="width: 20%;">
        <IMG src="images/tetra.1.png" style="width: 20%;">
        <IMG src="images/tetra.2.png" style="width: 20%;">
        <IMG src="images/tetra.3.png" style="width: 20%;">
      </p>
    </center>

----

Uniform Subdivision
===================

Applies a uniform refinement scheme to the coarse faces of a mesh.
The mesh converges closer to the limit surface with each iteration of the algorithm.

.. image:: images/uniform.gif
   :align: center
   :width: 300
   :target: images/uniform.gif

----

Feature Adaptive Subdivision
============================

Applies a progressive refinement strategy to isolate irregular features.
The resulting vertices can be assembled into bi-cubic patches defining the limit surface.

.. image:: images/adaptive.gif
   :align: center
   :width: 300
   :target: images/adaptive.gif

----

Uniform or Adaptive ?
=====================

Feature adaptive refinement can be much more economical in terms of time and memory use,
but the best method to use depends on application needs.

The following table identifies several factors to consider:

+-------------------------------------------------------+--------------------------------------------------------+
| Uniform                                               | Feature Adaptive                                       |
+=======================================================+========================================================+
|                                                       |                                                        |
| * Exponential geometry growth                         | * Geometry growth close to linear and occuring only in |
|                                                       |   the neighborhood of isolated topological features    |
|                                                       |                                                        |
+-------------------------------------------------------+--------------------------------------------------------+
|                                                       |                                                        |
| * Current implementation only produces bi-linear      | * Current implementation only produces bi-cubic        |
|   patches for uniform refinement                      |   patches for feature adaptive refinement              |
|                                                       |                                                        |
+-------------------------------------------------------+--------------------------------------------------------+
|                                                       |                                                        |
| * All face-varying interpolation rules supported at   | * Currently, only bi-linear face-varying interpolation |
|   refined vertex locations                            |   is supported for bi-cubic patches                    |
|                                                       |                                                        |
+-------------------------------------------------------+--------------------------------------------------------+

|
|
|

.. container:: notebox

   **Release Notes (3.0.0)**

   * Full support for bi-cubic face-varying interpolation is a significant
     feature which will be supported in future releases.

   * Feature adaptive refinement for the Loop subdivision scheme is
     expected to be supported in future releases.

----

Boundary Interpolation Rules
============================

Boundary interpolation rules control how boundary edges and vertices are interpolated.

The following rule sets can be applied to vertex data interpolation:

+----------------------------------+----------------------------------------------------------+
| Mode                             | Behavior                                                 |
+==================================+==========================================================+
| **VTX_BOUNDARY_NONE**            | No boundary edge interpolation should occur; instead     |
|                                  | boundary faces are tagged as holes so that the boundary  |
|                                  | edge-chain continues to support the adjacent interior    |
|                                  | faces but is not considered to be part of the refined    |
|                                  | surface                                                  |
+----------------------------------+----------------------------------------------------------+
| **VTX_BOUNDARY_EDGE_ONLY**       | All the boundary edge-chains are sharp creases; boundary |
|                                  | vertices are not affected                                |
+----------------------------------+----------------------------------------------------------+
| **VTX_BOUNDARY_EDGE_AND_CORNER** | All the boundary edge-chains are sharp creases and       |
|                                  | boundary vertices with exactly one incident face are     |
|                                  | sharp corners                                            |
+----------------------------------+----------------------------------------------------------+

On a grid example:

.. image:: images/vertex_boundary.png
   :align: center
   :target: images/vertex_boundary.png


----

Face-Varying Interpolation Rules
================================

Face-varying data is used when discontinuities are required in the data over the 
surface -- mostly commonly the seams between disjoint UV regions.
Face-varying data can follow the same interpolation behavior as vertex data, or it
can be constrained to interpolate linearly around selective features from corners,
boundaries, or the entire interior of the mesh.

The following rules can be applied to face-varying data interpolation -- the
ordering here applying progressively more linear constraints:

+--------------------------------+-------------------------------------------------------------+
| Mode                           | Behavior                                                    |
+================================+=============================================================+
| **FVAR_LINEAR_NONE**           | smooth everywhere the mesh is smooth                        |
+--------------------------------+-------------------------------------------------------------+
| **FVAR_LINEAR_CORNERS_ONLY**   | sharpen (linearly interpolate) corners only                 |
+--------------------------------+-------------------------------------------------------------+
| **FVAR_LINEAR_CORNERS_PLUS1**  | CORNERS_ONLY + sharpening of junctions of 3 or more regions |
+--------------------------------+-------------------------------------------------------------+
| **FVAR_LINEAR_CORNERS_PLUS2**  | CORNERS_PLUS1 + sharpening of darts and concave corners     |
+--------------------------------+-------------------------------------------------------------+
| **FVAR_LINEAR_BOUNDARIES**     | linear interpolation along all boundary edges and corners   |
+--------------------------------+-------------------------------------------------------------+
| **FVAR_LINEAR_ALL**            | linear interpolation everywhere (boundaries and interior)   |
+--------------------------------+-------------------------------------------------------------+

These rules cannot make the interpolation of the face-varying data smoother than
that of the vertices.  The presence of sharp features of the mesh created by
sharpness values, boundary interpolation rules, or the subdivision scheme itself
(e.g. Bilinear) take precedence.

All face-varying interpolation modes illustrated in UV space using the
catmark_fvar_bound1 regression shape -- a simple 4x4 grid of quads segmented
into three UV regions (their control point locations implied by interpolation
in the FVAR_LINEAR_ALL case):

.. image:: images/fvar_boundaries.png
   :align: center
   :target: images/fvar_boundaries.png


----

Semi-Sharp Creases
==================

It is possible to modify the subdivision rules to create piecewise smooth
surfaces containing infinitely sharp features such as creases and corners. As a
special case, surfaces can be made to interpolate their boundaries by tagging
their boundary edges as sharp.

However, we've recognized that real world surfaces never really have infinitely
sharp edges, especially when viewed sufficiently close. To this end, we've
added the notion of semi-sharp creases, i.e. rounded creases of controllable
sharpness. These allow you to create features that are more akin to fillets and
blends. As you tag edges and edge chains as creases, you also supply a
sharpness value that ranges from 0-10, with sharpness values >=10 treated as
infinitely sharp.

It should be noted that infinitely sharp creases are really tangent
discontinuities in the surface, implying that the geometric normals are also
discontinuous there. Therefore, displacing along the normal will likely tear
apart the surface along the crease. If you really want to displace a surface at
a crease, it may be better to make the crease semi-sharp.

.. image:: images/gtruck.png
   :align: center
   :height: 300
   :target: images/gtruck.png

----

Chaikin Rule
============

Chaikin's curve subdivision algorithm improves the appearance of multi-edge
semi-sharp creases with varying weights. The Chaikin rule interpolates the
sharpness of incident edges.

+---------------------+---------------------------------------------+
| Mode                | Behavior                                    |
+=====================+=============================================+
| **CREASE_UNIFORM**  | Apply regular semi-sharp crease rules       |
+---------------------+---------------------------------------------+
| **CREASE_CHAIKIN**  | Apply "Chaikin" semi-sharp crease rules     |
+---------------------+---------------------------------------------+

Example of contiguous semi-sharp creases interpolation:

.. image:: images/chaikin.png
   :align: center
   :target: images/chaikin.png

----

"Triangle Subdivision" Rule
===========================

The triangle subdivision rule is a rule added to the Catmull-Clark scheme that
can be applied to all triangular faces; this rule was empirically determined to
make triangles subdivide more smoothly. However, this rule breaks the nice
property that two separate meshes can be joined seamlessly by overlapping their
boundaries; i.e. when there are triangles at either boundary, it is impossible
to join the meshes seamlessly

+---------------------+---------------------------------------------+
| Mode                | Behavior                                    |
+=====================+=============================================+
| **TRI_SUB_CATMARK** | Default Catmark scheme weights              |
+---------------------+---------------------------------------------+
| **TRI_SUB_SMOOTH**  | "Smooth triangle" weights                   |
+---------------------+---------------------------------------------+

Cylinder example :

.. image:: images/smoothtriangles.png
   :align: center
   :height: 300
   :target: images/smoothtriangles.png


----

Manifold vs Non-Manifold Geometry
=================================

Continuous limit surfaces generally require that the topology be a
two-dimensional manifold for the limit surface to be unambiguous.  It is
possible (and sometimes useful, if only temporarily) to model non-manifold
geometry and so create surfaces whose limit is not as well-defined.

The following examples show typical cases of non-manifold topological
configurations.

----

Non-Manifold Fan
****************

This "fan" configuration shows an edge shared by 3 distinct faces.

.. image:: images/nonmanifold_fan.png
   :align: center
   :target: images/nonmanifold_fan.png

With this configuration, it is unclear which face should contribute to the
limit surface (assuming it is singular) as three of them share the same edge.
Fan configurations are not limited to three incident faces: any configuration
where an edge is shared by more than two faces incurs the same problem.

These and other regions involving non-manifold edges are dealt with by
considering regions that are "locally manifold".  Rather than a single limit
surface through this problematic edge with its many incident faces, the edge
locally partitions a single limit surface into more than one.  So each of the
three faces here will have their own (locally manifold) limit surface -- all
of which meet at the shared edge.

----

Non-Manifold Disconnected Vertex
********************************

A vertex is disconnected from any edge and face.

.. image:: images/nonmanifold_vert.png
   :align: center
   :target: images/nonmanifold_vert.png

This case is fairly trivial: there is a very clear limit surface for the four
vertices and the face they define, but no possible way to exact a limit surface
from the disconnected vertex.

While the vertex does not contribute to any
limit surface, it may not be completely irrelevant though.  Such vertices may
be worth retaining during subdivision (if for no other reason than to preserve
certain vertex ordering) and simply ignored when it comes time to consider
the limit surface.

