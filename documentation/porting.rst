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
     KIND, either express or implied.  See the Apache License for the specific
     language governing permissions and limitations under the Apache License.

Porting Guide: 2.x to 3.0
-------------------------

.. contents::
   :local:
   :backlinks: none


Porting Guide: 2.x to 3.0
=========================

This document is a high-level description of how to port exiting OpenSubdiv 2.x
code to use OpenSubdiv 3.0.

**NOTE:** If your questions are not answered here, please contact us on the
OpenSubdiv forum and we will be happy to help!


Source Code Organization
========================

Given the scale of functional changes that were being made to the public
interface, we took the opportunity in 3.0 to update the coding style and
organization -- most notably making use of namespaces for each library.

================= ==================== ===============================================
Subdirectory      Namespace            Relevance
================= ==================== ===============================================
hbr/              N/A                  Historical, no longer used
sdc/              Sdc                  New, low-level, public options, constants, etc.
vtr/              Vtr                  New, internal use, topology representation
far/              Far                  Revised, similar functionality with new API
osd/              Osd                  Revised, similar functionality with new API
================= ==================== ===============================================


Hbr Layer Translation
=====================

Client mesh topology is now translated into an instance of Far::TopologyRefiner
instead of HbrMesh.

================= ====================
OpenSubdiv 2.x    OpenSubdiv 3.0
================= ====================
HbrMesh<VTX_TYPE> Far::TopologyRefiner
================= ====================

The Far::TopologyRefiner is now the core representation of topology from which
all other major classes in Far and Osd are constructed.  It was designed to
support efficient refinement (uniform or sparse) of a base mesh of arbitrary
topology (no manifold restrictions).  Once constructed it can be directly
refined to meet some need, or passed to other contexts that will refine it to
meet their needs.

In contrast to directly assembling an HbrMesh, the TopologyRefiner, like other
classes in Far, requires a Factory class for its construction.  One of the 
early goals of these factories was to allow a client to convert their existing
boundary representation -- with its full topological traversal abilities --
directly into the TopologyRefiners representation.  While this is now possible,
this also represents the most complex construction process and is only
recommended for usage where this conversion process is critical.

Details on how to construct a TopologyRefiner can be found in the 
`Far overview <far_overview.html#far-topologyrefinerfactory>`__ documentation.
Additionally, documentation for Far::TopologyRefinerFactory<MESH> outlines the
requirements, and Far tutorial 3.1 (tutorials/far/tutorial_3_1) provides an example
of a factory for directly converting HbrMeshes to TopologyRefiners.

Its worth a reminder here that Far::TopologyRefiner contains only topological
information (which does include sharpness, since that is considered relating
to subdivision topology) and not the positions or other data associated with
a mesh.  While HbrMesh<T> required some definition of a vertex type <T> and
dimensions of face-varying data, TopologyRefiner is more clearly separated
from the data.  So the construction of the TopologyRefiner does not involve
data specification at all.

Subdivision Schemes and Options in Sdc
++++++++++++++++++++++++++++++++++++++

The creation of a new TopologyRefiner requires specification of a subdivision 
scheme and a set of options that are applicable to all schemes.  With HbrMesh, 
the scheme was specified by declaring a static instance of a specific subclass
of a subdivision object, and the options were specified with a number of
methods on the different classes.

Such general information about the schemes has now been encapsulated in the
Sdc layer for use throughout OpenSubdiv.  The subdivision scheme is now a
simple enumerated type (Sdc::SchemeType) and the entire set of options that
can be applied to a scheme is encapsulated in a single simple struct of
flags and enumerated types (Sdc::Options).

===============================================  ===========================================
OpenSubdiv 2.x                                   OpenSubdiv 3.0
===============================================  ===========================================
HbrMesh<T>::SetInterpolateBoundaryMethod()       Sdc::Options::SetVtxBoundaryInterpolation()
HbrMesh<T>::SetFVarInterpolateBoundaryMethod()   Sdc::Options::SetFVarLinearInterpolation()
HbrSubdivision<T>::SetCreaseSubdivisionMethod()  Sdc::Options::SetCreasingMethod()
===============================================  ===========================================


Specifying Face Varying Topology and Options
++++++++++++++++++++++++++++++++++++++++++++

Both the way in which face varying data is associated with a mesh and the
options used to control its interpolation have changed.  The documentation on
`Compatibility with OpenSubdiv 2.x <compatibility.html#compatibility-with-opensubdiv-2.x>`__
details the equivalence of interpolation options between Hbr and the new
*Sdc::Options::FVarLinearInterpolation* enum, while the section on
`Face Varying Interpolation <subdivision_surfaces.html#face-varying-interpolation-rules>`__
illustrates their effects.

Face varying data is now specified by index rather than by value, or as often
stated, it is specified topologically.  Just as vertices for faces are specified
by indices into a potential buffer of positions, face varying values are
specified by indices into a potential buffer of values.  Both vertices and
face varying values (frequently referred to as *FVarValues* in the API) are
assigned and associated with the corners of all faces.

In many cases this will simplify representation as many common geometry
container formats such as Obj or Alembic specify texture coordinates the same
way.  For other cases, where a value per face-corner is provided with no
indication of which values incident each vertex should be considered shared,
it will be necessary to determine shared indices for values at each vertex if
any non-linear interpolation is desired.


Far Layer Translation
=====================

While TopologyRefiner was introduced into Far as the new intermediate
topology representation, several other changes were made to classes in Far
to provide more modular building blocks for use by the Osd layer or directly.

===================== =====================
OpenSubdiv 2.x        OpenSubdiv 3.0
===================== =====================
FarMesh<U>            N/A, no longer needed
FarSubdivisionTables  Far::StencilTable
FarPatchTables        Far::PatchTable
===================== =====================

Ordering of Refined Vertices
++++++++++++++++++++++++++++

The FarMesh was previously responsible for refining an HbrMesh -- generating
new vertices and faces in successive levels of refinement in the
FarSubdivisionTables.  Vertices were grouped and reordered from the native
ordering of HbrMesh so that vertices requiring similar processing were
consecutive.  Such grouping alleviated most of the idiosyncrasies of
HbrMesh's native ordering but not all.

Far::ToplogyRefiner is inherently a collection of refinement levels, and
within each refined level (so excluding the base level), all components are
still grouped for the same reasons.  There are two issues here though:

* the ordering of these groups has changed (though an option exists to
  preserve it)

* the ordering of components within these groups is not guaranteed to have
  been preserved

Vertices in a refined level are grouped according to the type of component in
the parent level from which they originated, i.e. some vertices originate
from the center of a face (face-vertices), some from an edge (edge-vertices)
and some from a vertex (vertex-vertices).  (Note that there is a conflict in
terminology here -- face-vertices and edge-vertices most often refer to
vertices incident a face or edge -- but for the sake of this discussion, we
use them to refer to the component from which a child vertex originates.)

The following table shows the ordering of these groups in 2.x and the two
choices available in 3.0.  The option is the *orderVerticesFromFacesFirst*
flag that can be set in the Option structs passed to the uniform and adaptive
refinement methods of TopologyRefiner:

============================================ =============================================
Version and option                           Vertex group ordering
============================================ =============================================
2.x                                          face-vertices, edge-vertices, vertex-vertices
3.0 default                                  vertex-vertices, face-vertices, edge-vertices
3.0 orderVerticesFromFacesFirst = true       face-vertices, edge-vertices, vertex-vertices
============================================ =============================================

The decision to change the default ordering was based on common feedback;
the rationale was to allow a trivial mapping from vertices in the cage to
their descendants at all refinement levels.  While the grouping is
fundamental to the refinement process, the ordering of the groups is
internally flexible, and the full set of possible orderings can be made
publicly available in future if there is demand for such flexibility.

The ordering of vertices within these groups was never clearly defined given
the way that HbrMesh applied its refinement.  For example, for the
face-vertices in a level, it was never clear which face-vertices would be
first as it depended on the order in which HbrMesh traversed the parent faces
and generated them. Given one face, HbrMesh would often visit neighboring
faces first before moving to the next intended face.

The ordering with Far::TopologyRefiner is much clearer and predictable.  Using
the face-vertices as an example, the order of the face-vertices in level *N+1*
is identical to the order of the parent faces in level *N* from which they
originated.  So if we have face-vertices *V'i*, *V'j* and *V'k* at some level,
originating from faces *Fi*, *Fj* and *Fk* in the previous level, they will
be ordered in increasing order of *i*, *j* and *k*.  For uniform refinement
the ordering of face vertices *V'i* will therefore exactly match the ordering
of the parent faces *Fi*.  For adaptive or otherwise sparse refinement, the
subset of *Vi* will be ordered similarly, just with components missing from
those not refined.

The same is true of all vertices, i.e. edge-vertices and vertex-vertices,
and also for other components in refined levels, i.e. the child faces and
edges.  

For child faces and edges, more than one will originate from the same parent
face or edge.  In addition to the overall ordering based on the parent faces
or edges, another ordering is imposed on multiple children originating from 
the same face or edge.  They will be ordered based on the corner or
end-vertex with which they are associated.

In the case of refined faces, another way to view the ordering is to consider
the way that faces are originally defined -- by specifying the set of vertices
for the corners of each face, often aggregated into a single large array.  The
ordering of the set of refined faces for each level will correspond directly
to such an array of vertices per face in the previous level.


Osd Layer Translation
=====================

Two big changes in the 3.0 API have allowed the Osd layer to be significantly
simpler, the first is the move to stencil tables from subdivision tables and the
second is shader simplification. With this refactoring, the focus has been to
use more meaningful names and to make the data contained within an object more
apparent.

Controller Objects
++++++++++++++++++

.. _Evaluator: doxy_html/a00024.html

The API-specific ComputeController has been replaced with the Evaluator_. It
reflects the fact that stencil compute batches are significantly simpler than
subdivision table compute batches.

The name "Evaluator" was chosen with the hope that is more meaningful than the
generic "ComputeController" moniker: the Evaluator evaluates stencil and
patch tables.

In the 2.x code base, subdiv level buffers were always required to be allocated
contiguously, however in 3.0 with the shift to stencil tables, this strict
allocation scheme is no longer required. As a result, the EvalStencils() and
EvalPatches() methods now accept both a source and a destination descriptor.

======================================= ========================================
OpenSubdiv 2.x                          OpenSubdiv 3.0
======================================= ========================================
ComputeController::Refine()             Osd::...Evaluator::EvalStencils()
ComputeController::Synchronize()        Osd::...Evaluator::Synchronize()
EvalStencilsController::UpdateValues()  Osd::...Evaluator::EvalStencils()
EvalStencilsController::UpdateDerivs()  Osd::...Evaluator::EvalStencils()
EvalLimitController::EvalLimitSample()  Osd::...Evaluator::EvalPatches()
======================================= ========================================

Also note that OsdVertexDescriptor has been renamed, however it's data members
and semantic purpose remains the same:

======================================= ========================================
OpenSubdiv 2.x                          OpenSubdiv 3.0
======================================= ========================================
OsdVertexBufferDescriptor               Osd::BufferDescriptor
======================================= ========================================

ComputeContext, DrawContext
+++++++++++++++++++++++++++

ComputeContext and DrawContext have been replaced with API-specific StencilTable
and PatchTable objects, for example Osd::GLStencilTableSSBO.

======================================= ========================================
OpenSubdiv 2.x                          OpenSubdiv 3.0
======================================= ========================================
ComputeContext                          Osd::...StencilTable (e.g. GLStencilTableTBO)
EvalStencilsContext                     Osd::...StencilTable
DrawContext                             Osd::...PatchTable (e.g. GLPatchTable)
======================================= ========================================

EvalLimitContext
++++++++++++++++

The data stored in EvalLimitContext has been merged into the Evaluator class as
well.

EvalCoords have been moved into their own type, Osd::PatchCoords. The primary
change here is that the PTex face ID is no longer part of the data structure,
rather the client can use a Far::PatchMap to convert from PTex face ID to a
Far::PatchTable::PatchHandle.

======================================= ========================================
OpenSubdiv 2.x                          OpenSubdiv 3.0
======================================= ========================================
EvalLimitContext                        PatchTable 
EvalLimitContext::EvalCoords            Osd::PatchCoords (types.h)
======================================= ========================================

OsdMesh
+++++++

While not strictly required, OsdMesh is still supported in 3.0 as convenience
API for allocating buffers. OsdMesh serves as a simple way to allocate all
required data, in the location required by the API (for example, GPU buffers for
OpenGL).

OsdKernelBatch
++++++++++++++

No translation, it is no longer part of the API.

OsdVertex
+++++++++

No translation, it is no longer part of the API.

Feature Adaptive Shader Changes
===============================

In 3.0, the feature adaptive screen-space tessellation shaders have been
dramatically simplified, and the client-facing API has changed dramatically as
well. The primary shift is to reduce the total number of shader combinations, and
as a result, some of the complexity management mechanisms are no longer
necessary.

In the discussion below, some key changes are highlighted, but deep
integrations may require additional discussion; please feel free to send
follow up questions to the OpenSubdiv google group.

 * The number of feature adaptive shaders has been reduced from N to exactly 1
   or 2, depending on how end-caps are handled.

 * Osd layer no longer compiles shaders, rather it returns shader source for the
   client to compile. This source is obtained via 
   Osd::[GLSL|HLSL]PatchShaderSource.

 * The API exposed in shaders to access patch-based data has been consolidated
   and formalized, see osd/glslPatchCommon.glsl and osd/hlslPatchCommon.hlsl for
   details.

 * Patches are no longer rotated and transition patches have been eliminated,
   simplifying PatchDescriptor to a 4 bits. Additionally, FarPatchTables::Descriptor
   has been moved into its own class in the Far namespace.

The following table outlines the API translation between 2.x and 3.0:

======================================= ========================================
OpenSubdiv 2.x                          OpenSubdiv 3.0
======================================= ========================================
OsdDrawContext::PatchDescriptor         N/A, no longer needed.
OsdDrawContext::PatchArray              OSd::PatchArray (types.h)
FarPatchTables::PatchDescriptor         Far::PatchDescriptor (patchDescriptor.h)
FarPatchTables::PatchArray              made private.
======================================= ========================================

End Cap Strategies
++++++++++++++++++

By default, OpenSubdiv uses Gregory patches to approximate the patches around
extraordinary vertices at the maximum isolation level, this process is referred
to as "end-capping".

If ENDCAP_BSPLINE_BASIS is specified to PatchTableFactory::Options, BSpline
patches are used, which gives less accuracy, but it makes possible to render an
entire mesh in a single draw call. Both patches require additional control
points that are not part of the mesh, we refer to these as "local points". In
3.0, the local points of those patches are computed by applying a stencil table
to refined vertices to construct a new stencil table for the local points.

Since this new stencil table is topologically compatible with the primary
stencil table for refinement, it is convenient and efficient to splice those 
stencil tables together. This splicing can be done in the following way::

  Far::StencilTable const *refineStencils = 
                                Far::StencilTableFactory::Create(topologyRefiner);

  Far::PatchTable cosnt *patchTable = 
                                Far::PatchTableFactory::Create(topologyRefiner);

  Far::StencilTable const *localPointStencils = 
                                    patchTable->GetLocalPointStencilTable();

  Far::StencilTable const *splicedStencils = 
          Far::StencilTableFactory::AppendLocalPointStencilTables(topologyRefiner,
                                                            refineStencils, 
                                                            localPointStencils);

**NOTE:** Once the spliced stencil table is created, the refined stencils can be
released, but the local point stencils are owned by patchTable, it should not be
released.

OpenSubdiv 3.0 also supports 2.x style Gregory patches, if ENDCAP_LEGACY_GREGORY
is specified to PatchTableFactory::Options. In this case, such an extra stencil
splicing isn't needed, however clients must still bind additional buffers
(VertexValence buffer and QuadOffsets buffer). 

See Osd::GLLegacyGregoryPatchTable for additional details. 

Changes to Subdivision 
======================

The refactoring of OpenSubdiv 3.0 data representations presented a unique
opportunity to revisit some corners of the subdivision specification and
remove or update some legacy features -- none of which was taken lightly.
More details are provided in
`Subdivision Compatibility <compatibility.html>`__, while the
following offers a quick overview:

* All face-varying interpolation options have been combined into a single enum.

* Vertex interpolation options have been renamed or removed:

  * The naming of the standard creasing method has changed from *Normal* to *Uniform*.

  * Unused legacy modes of the *"smoothtriangle"* option have been removed.

* The averaging of Chaikin creasing with infinitely sharp edges has changed.

* Support for Hierarchical Edits has been removed.


Build Support for Combining 2.x and 3.0
=======================================

Running OpenSubdiv 2.0 and 3.0 in a single process is supported, however some
special care must be taken to avoid namespace collisions, both in terms of
run-time symbols (avoid "using OpenSubdiv::Osd", for example) and in terms of
build-time search paths.

To support both OpenSubdiv 2.0 and 3.0 in your build environment, you can
prefix the header install directory of OpenSubdiv 3.0. Do this using the build
flag "CMAKE_INCDIR_BASE" when configuring cmake (i.e. 
-DCMAKE_INCDIR_BASE=include/opensubdiv3) and then including files from
"opensubdiv3/..." in client code.

