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

Porting Guide: 2.x to 3.0
-------------------------

.. contents::
   :local:
   :backlinks: none

This document is a high-level description of how to port exiting OpenSubdiv 2.x
code to use OpenSubdiv 3.0.

**NOTE:** If your questions are not answered here, please contact us on the
OpenSubdiv forum and we will be happy to help!


Source Code Organization
========================

Given the scale of functional changes that were being made to the public
interface, we took the oppoortunity in 3.0 to update the coding style and
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

With HbrMesh having been the source of a number of functional and performance
issues, client mesh topology is now translated into an instance of the new
TopologyRefiner class in the Far level.

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

There are three ways to construct a TopologyRefiner -- ranging from the very
simple but less optimal to the more complex just noted.  The first involves
use of a predefined factory class provided in Far, while the others require
writing custom factories, i.e. Far::TopologyRefinerFactory<MESH>.  These are
typically stateless factories with a static Create() method that will be used
to instantiate a new TopologyRefiner.  All three are illustrated in either
tutorials or examples as noted in the subsections that follow.

Its worth a reminder here that Far::TopologyRefiner contains only topological
information (which does include sharpness, since that is considered relating
to subdivision topology) and not the positions or other data associated with
a mesh.  While HbrMesh<T> required some definition of a vertex type <T> and
dimensions of face-varying data, TopologyRefiner is more clearly separated
from the data.  So the construction of the TopologyRefiner does not involve
data specification at all.

Subdivision Schemes and Options
+++++++++++++++++++++++++++++++

Before detailing the topology conversion, since the creation of a new
TopologyRefiner requires specification of a subdivision scheme and a set of
options that are applicable to all schemes.  With HbrMesh, the scheme was
specified by declaring a static instance of a specific subclass of a
subdivision object, while the options were specified with a number of
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

Regardless of the three construction choices outlined below, the specification
of both the scheme and all options related to it is the same.

Specifying Face Varying Topology
++++++++++++++++++++++++++++++++

(Just a place holder for now -- more to come...)

Factories to Build Far::TopologyRefiners
++++++++++++++++++++++++++++++++++++++++

Here we outline the three approaches for converting mesh topology into the
required Far::TopologyRefiner.  Additional documentation is provided with
the Far::TopologyRefinerFactory<MESH> class template used by all, and each
has a concrete example provided in one of the tutorials or in the Far code
itself.  Please contact the OpenSubdiv forum if questions are not answered
here or in the other documentation and examples cited.

1)  Use the Far::TopologyDescriptor
***********************************

Far::TopologyDescriptor is a simple struct that can be initialized to refer
to raw mesh topology information -- primarily a face-vertex list -- and then
passed to a provided factory class to create a TopologyRefiner from each.
The minimum information required is typical of what many mesh construction
tools require:  the number of vertices and faces, the number of vertices per
face, and the complete set of face-vertices for all faces.

Almost all of the Far tutorials (i.e. tutorials/far/tutorial_*) illustrate
use of the TopologyDescriptor and its factory for creating TopologyRefiners,
i.e. TopologyRefinerFactory<TopologyDescriptor>.

For situations when users have raw mesh data and have not yet constructed a
boundary representation of their own, it is hoped that this will suffice.
Options have even been provided to indicate that raw topology information
has been defined in a left-hand winding order and the factory will handle
the conversion to right-hand (counter-clockwise) winding on-the-fly to avoid
unnecessary data duplication.

2)  Custom Factory for Face Vertices
************************************

If the nature of the TopologyDescriptor's data expectations is not helpful,
and so conversion to large temporary arrays would be necessary to properly
make use of it, it may be worth writing a custom factory.

There are two ways to write such a factory:  provide only the face-vertex
information for topology and let the factory infer all edges and other
relationships, or provide the complete edge list and all other topological
relationships directly.  The latter is considerably more involved and
described in a following section.

The definition of TopologyRefinerFactory<TopologyDescriptor> provides a clear
and complete example of constructing a TopologyRefiner with minimal topology
information, i.e. the face-vertex list.  The class template
TopologyRefinerFactory<MESH> documents the needs here and the
TopologyDescriptor instantiation and specialization should illustrate that.

3)  Custom Factory for Direct Conversion
****************************************

This is not recommended as an introduction to 3.0.  It is recommended that
one of the previous two methods initially be used to convert your mesh
topology into a TopologyRefiner and get other aspects of 3.0 working first.
If the conversion performance is critical, or significant enough to warrant
improvement, then its worth writing a factory for full topological conversion.

Documentation for Far::TopologyRefinerFactory<MESH> outlines the requirements
and a Far tutorial (tutorials/far/tutorial_1) provides an example of a factory
for directly converting HbrMeshes to TopologyRefiners.

This approach requires dealing directly with edges, unlike the other two.  In
order to convert edges into a TopologyRefiner's representation, the edges need
to be expressed as a collection of some size N -- each of which is referred to
directly by indices [0,N-1].  This can be awkward for representations such as
half-edge or quad-edge that do not treat the instance of an edge uniquely.

Particular care is also necessary when representing non-manifold features.  The
previous two approaches will construct non-manifold features as required from
the face-vertex list -- dealing with degenerate edges and other non-manifold
features as encountered.  When directly translating full topology it is
necessary to tag non-manifold features, and also to ensure that certain
edge relationships are satisfied in their presence.  More details are
available with the assembly methods of the factory class template.

The factory does provide run-time validation on the topology constructed that
can be used for debugging purposes.


Far Layer Translation
=====================

(More to be said here -- a place holder for now...)

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

(Need to address this topic at some point -- is this the right place?)


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

Essentially replaced with API-specific StencilTable and PatchTable objects, for
example Osd::GLStencilTableSSBO.

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
dramatically simplified and the client-facing API has changed dramatically as
well. The primary shift is to reduce the total number of shader combinations and
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
`Subdivision Compatibility Guide <compatibility.html>`__, while the
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

