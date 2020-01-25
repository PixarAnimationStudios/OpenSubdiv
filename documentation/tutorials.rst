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


Tutorials
---------

.. contents::
   :local:
   :backlinks: none

----

The tutorial source code can be found in the `github.com repository
<https://github.com/PixarAnimationStudios/OpenSubdiv/tree/master/tutorials>`__
or in your local ``<repository root>/tutorials``.

Far Tutorials
=============

1. Basic Construction and Interpolation
***************************************

Tutorial 1.1
^^^^^^^^^^^^
 This tutorial presents the requisite steps to instantiate a mesh as a
 Far::TopologyRefiner from simple topological data and to interpolate
 vertex data associated with the mesh. `[code] <far_tutorial_1_1.html>`__

.. image:: images/far_tutorial_1_1.0.png
   :align: center
   :width: 100px
   :target: images/far_tutorial_1_1.0.png

Tutorial 1.2
^^^^^^^^^^^^
 This tutorial makes use of a different vertex data definition for use when vertex
 data is of arbitrary width.  Uniform refinement is applied to data buffers of three
 types:  two of fixed but different sizes and the third a union of the two that is
 dynamically sized and constructed.  `[code] <far_tutorial_1_2.html>`__

2. Uniform Refinement and Primvar Data Types
********************************************

Tutorial 2.1
^^^^^^^^^^^^
 Building on the basic tutorial, this example shows how to instantiate a simple mesh,
 refine it uniformly and then interpolate both 'vertex' and 'varying' primvar data.
 `[code] <far_tutorial_2_1.html>`__

.. image:: images/far_tutorial_2_1.0.png
   :align: center
   :width: 100px
   :target: images/far_tutorial_2_1.0.png

Tutorial 2.2
^^^^^^^^^^^^
 Building on the previous tutorial, this example shows how to instantiate a simple mesh,
 refine it uniformly and then interpolate both 'vertex' and 'face-varying' primvar data.
 The resulting interpolated data is output in Obj format, with the 'face-varying' data
 recorded in the UV texture layout.  `[code] <far_tutorial_2_2.html>`__

.. image:: images/far_tutorial_2_2.0.png
   :align: center
   :width: 100px
   :target: images/far_tutorial_2_2.0.png

Tutorial 2.3
^^^^^^^^^^^^
 Building on previous tutorials, this example shows how to instantiate a simple mesh,
 refine it uniformly, interpolate both 'vertex' and 'face-varying' primvar data, and
 finally calculate approximated smooth normals.  The resulting interpolated data is
 output in Obj format.  `[code] <far_tutorial_2_3.html>`__

3. Creating a Custom Far::TopologyRefinerFactory
************************************************

Tutorial 3.1
^^^^^^^^^^^^
 Previous tutorials have instantiated topology from a simple face-vertex list via the
 Far::TopologyDescriptor and its TopologyRefinerFactory.  This tutorial shows how to
 more efficiently convert an existing high-level topology representation to a
 Far::TopologyDescriptor with a custom factory class.  `[code] <far_tutorial_3_1.html>`__

4. Construction and Usage of Far::StencilTables
***********************************************

Tutorial 4.1
^^^^^^^^^^^^
 This tutorial shows how to create and manipulate a StencilTable. Factorized stencils
 are used to efficiently interpolate vertex primvar data buffers.
 `[code] <far_tutorial_4_1.html>`__

Tutorial 4.2
^^^^^^^^^^^^
 This tutorial shows how to create and manipulate StencilTables for both 'vertex' and
 'varying' primvar data buffers: vertex positions and varying colors.
 `[code] <far_tutorial_4_2.html>`__

Tutorial 4.3
^^^^^^^^^^^^
 This tutorial shows how to create and manipulate tables of cascading stencils to apply
 hierarchical vertex edits. `[code] <far_tutorial_4_3.html>`__

5. Construction and Usage of Far::PatchTables
*********************************************

Tutorial 5.1
^^^^^^^^^^^^
 This tutorial shows how to compute points on the limit surface at arbitrary parametric
 locations using a Far::PatchTable constructed from adaptive refinement.
 `[code] <far_tutorial_5_1.html>`__

.. image:: images/far_tutorial_5_1.0.png
   :align: center
   :width: 100px
   :target: images/far_tutorial_5_1.0.png

Tutorial 5.2
^^^^^^^^^^^^
 Building on the previous tutorial, this example shows how to manage the limit surface
 of a potentially large mesh by creating and evaluating separate PatchTables for selected
 groups of faces of the mesh.  `[code] <far_tutorial_5_2.html>`__

Tutorial 5.3
^^^^^^^^^^^^
 Building on the previous tutorials for both PatchTables and StencilTables, this example
 shows how to construct a LimitStencilTable to repeatedly evaluate an arbitrary
 collection of points on the limit surface.  `[code] <far_tutorial_5_3.html>`__

----

Osd Tutorials
=============

Tutorial 0
**********
 This tutorial demonstrates the manipulation of Osd Evaluator and BufferDescriptor.
 `[code] <osd_tutorial_0.html>`__

----

Hbr Tutorials
=============

Use of Hbr is no longer recommended -- these tutorials are included solely for
historical reference.

Tutorial 0
**********
 This tutorial presents, in a very succinct way, the requisite steps to
 instantiate an Hbr mesh from simple topological data. `[code] <hbr_tutorial_0.html>`__

Tutorial 1
**********
 This tutorial shows how to safely create Hbr meshes from arbitrary topology.
 Because Hbr is a half-edge data structure, it cannot represent non-manifold
 topology. Ensuring that the geometry used is manifold is a requirement to use
 Hbr safely. This tutorial presents some simple tests to detect inappropriate
 topology. `[code] <hbr_tutorial_1.html>`__

Tutorial 2
**********
 This tutorial shows how to subdivide uniformly a simple Hbr mesh. We are
 building upon previous tutorials and assuming a fully instantiated mesh:
 we start with an HbrMesh pointer initialized from the same pyramid shape
 used in hbr_tutorial_0. We then apply the Refine() function sequentially
 to all the faces in the mesh to generate several levels of uniform
 subdivision. The resulting data is then dumped to the terminal in Wavefront
 OBJ format for inspection. `[code] <hbr_tutorial_2.html>`__

.. image:: images/hbr_tutorial_2.0.png
   :align: center
   :width: 100px
   :target: images/hbr_tutorial_2.0.png

