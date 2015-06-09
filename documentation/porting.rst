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

Compatibility and Porting Guide
-------------------------------

.. contents::
   :local:
   :backlinks: none

This document is a high-level description of how to port exiting OpenSubdiv 2.x
code to use OpenSubdiv 3.0.

**NOTE:** If your questions are not answered here, please contact us on the
OpenSubdiv forum and we will be happy to help!

Far and Hbr Layer Translation
=============================

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

Changes to Subdiv Specification
===============================

RenderMan Compatibility Notes
=============================

Build Support for 2.x and 3.0
=============================

Running OpenSubdiv 2.0 and 3.0 in a single process is supported, however some
special care must be taken to avoid namespace collisions, both in terms of
run-time symbols (avoid "using OpenSubdiv::Osd", for example) and in terms of
build-time search paths.

To support both OpenSubdiv 2.0 and 3.0 in your build environment, you can
prefix the header install directory of OpenSubdiv 3.0. Do this using the build
flag "CMAKE_INCDIR_BASE" when configuring cmake (i.e. 
-DCMAKE_INCDIR_BASE=include/opensubdiv3) and then including files from
"opensubdiv3/..." in client code.

