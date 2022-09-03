..
     Copyright 2019 Pixar

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


Overview of Release 3.4
=======================

.. contents::
   :local:
   :backlinks: none

New Features
------------

Triangular Patches for Loop Subdivision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support for the drawing and evaluation of Loop subdivision meshes with
triangular patches was added.  This includes the full set of Far and Osd interfaces
for both evaluation and drawing.

+-----------------------------------+-----------------------------------+
| .. image:: images/loop_rel_1.png  | .. image:: images/loop_rel_2.png  |
|    :align:  center                |    :align:  center                |
|    :width:  95%                   |    :width:  95%                   |
|    :target: images/loop_rel_1.png |    :target: images/loop_rel_2.png |
+-----------------------------------+-----------------------------------+

The feature set supported for Loop subdivision now matches that of Catmark,
including creases, face-varying patches, non-manifold topology, etc.

+----------------------------------+----------------------------------------------------------------+
| .. image:: images/loop_rel_3.png | .. image:: images/loop_rel_4.png                               |
|    :align:  center               |    :align:  center                                             |
|    :width:  95%                  |    :width:  95%                                                |
|    :target: images/loop_rel_3.png|    :target: images/loop_rel_4.png                              |
+----------------------------------+----------------------------------------------------------------+

The long standing requirement that Loop meshes be purely triangular remains, as
Loop subdivision is not defined for non-triangular faces.  And as is the case with
the use of the Catmark scheme, application of Loop subdivision to dense, poorly
modeled meshes may lead to unexpectedly poor performance and/or surface quality.

The patch representation used for Loop subdivision is intended to exactly match the
underlying limit surface where regular, and so uses quartic triangular Box-splines.
This is in contrast to approaches that use simpler patches to approximate the Loop
limit surface everywhere.  As with Catmark, Gregory patches are used to approximate
irregular areas.  Though other choices are available that compromise surface
quality in favor of improved performance, they may be less effective with Loop than
they are with Catmark.

Major Improvements to Introductory Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A significant rewrite of the `Subdivision Surfaces <subdivision_surfaces.html>`__
page is included in this release.  The new documentation emphasizes the *piecewise
parametric surface* nature of subdivision surfaces and the implications of
supporting *arbitary topology*.

+---------------------------------------+---------------------------------------+
| .. image:: images/val6_regular.jpg    | .. image:: images/val6_irregular.jpg  |
|    :align:  center                    |    :align:  center                    |
|    :width:  95%                       |    :width:  95%                       |
|    :target: images/val6_regular.jpg   |    :target: images/val6_irregular.jpg |
+---------------------------------------+---------------------------------------+

As a true surface primitive, the distinction between the control points and the
limit surface and the corresponding operations of *subdivision* and *tessellation*
that are applied to them is made clear.

Sparse Patch Tables
~~~~~~~~~~~~~~~~~~~
Interfaces in Far for the construction of PatchTables and the required adaptive
refinement have been extended to apply to an arbitrary subset of faces.  This
allows patches for either large meshes or meshes that may otherwise benefit
from some kind of partioning (e.g. areas of static and dynamic topology) to be
managed in an arbitrary number of groups.  In the extreme, a PatchTable forming
the tree of patches for a single base face can be constructed.

Client data buffers for the base mesh do not need to be partitioned and base mesh
topology can be shared by multiple instances of Far::TopologyRefiner used to
create corresponding instances of Far::PatchTables.

See the new `Far tutorial 5.2 <far_tutorial_5_2.html>`__ for a simple example.

Support for Double Precision in Far
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classes and interfaces in Far have been adapted to allow support for double
precision via templates parameterized for float or double.  Class templates for
major classes such as Far::StencilTable have been introduced and the original
classes preserved for compatibility.  Other classes such as Far::PatchTable have
had methods overloaded or replaced with template functions to support both single
and double precision.  Internally, all use of floating point constants and math
library functions has been adapted to maximize accuracy appropriate to the
precision of the template instance.

Interfaces in Osd have not been extended.  The extensions in Far provide the
basis for extensions in Osd, but demand is limited.  For those benefiting from
such Osd extensions, contributions are welcomed.

See the revised `Far tutorial 5.1 <far_tutorial_5_1.html>`__ that constructs a
Far::PatchTable for a simple example.


API Additions
-------------

See associated `Doxygen <doxy_html/index.html>`__ for full details.

Far extensions for triangular patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - enum PatchDescriptor::Type::GREGORY_TRIANGLE
    - PatchParam::NormalizeTriangle()
    - PatchParam::UnnormalizeTriangle()
    - PatchParam::IsTriangleRotated()

Construction and refinement of topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - overloaded TopologyRefinerFactory::Create()
    - extensions to TopologyRefiner::RefineAdaptive()

Construction and interface of Far::PatchTable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - overloaded PatchTableFactory::Create()
    - PatchTableFactory::GetRefineAdaptiveOptions()
    - member PatchTableFactory::Options::includeBaseLevelIndices
    - member PatchTableFactory::Options::includeFVarBaseLevelIndices
    - member PatchTableFactory::Options::generateVaryingTables
    - member PatchTableFactory::Options::generateVaryingLocalPoints
    - member PatchTableFactory::Options::setPatchPrecisionDouble
    - member PatchTableFactory::Options::setFVarPatchPrecisionDouble
    - PatchTable::GetFVarPatchDescriptorRegular()
    - PatchTable::GetFVarPatchDescriptorIrregular()
    - PatchTable::GetFVarValueStride()

Construction and use of Far stencil tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - overloaded StencilTable::UpdateValues()
    - enum LimitStencilTableFactory::Mode
    - member LimitStencilTableFactory::Options::interpolationMode
    - member LimitStencilTableFactory::Options::fvarChannel

Far class templates for double precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - class StencilReal
    - class StencilTableReal
    - class StencilTableFactoryReal
    - class LimitStencilReal
    - class LimitStencilTableReal
    - class LimitStencilTableFactoryReal
    - class PrimvarRefinerReal

Far member functions converted to templates for double precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - PatchParam::Normalize()
    - PatchParam::Unnormalize()
    - PatchTable::EvaluateBasis()
    - PatchTable::EvaluateBasisVarying()
    - PatchTable::EvaluateBasisFaceVarying()
    - PatchTable::GetLocalPointStencilTable()
    - PatchTable::GetLocalPointVaryingStencilTable()
    - PatchTable::GetLocalPointFaceVaryingStencilTable()
    - PatchMap::FindPatch()

Osd::MeshBits
~~~~~~~~~~~~~
    - enumeration MeshEndCapBilinearBasis

Osd::PatchArray
~~~~~~~~~~~~~~~
    - GetDescriptorRegular()
    - GetDescriptorIrregular()
    - GetPatchTyperRegular()
    - GetPatchTyperIrregular()
    - GetStride()

Osd extensions for patch evaluation common to all shaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - struct OsdPatchArray and OsdPatchArrayInit()
    - struct OsdPatchCoord and OsdPatchCoordInit()
    - struct OsdPatchParam and OsdPatchParamInit()
    - OsdPatchParamGetFaceId()
    - OsdPatchParamGetU()
    - OsdPatchParamGetV()
    - OsdPatchParamGetTransition()
    - OsdPatchParamGetBoundary()
    - OsdPatchParamGetNonQuadRoot()
    - OsdPatchParamGetDepth()
    - OsdPatchParamGetParamFraction()
    - OsdPatchParamIsRegular()
    - OsdPatchParamIsTriangleRotated()
    - OsdPatchParamNormalize()
    - OsdPatchParamUnnormalize()
    - OsdPatchParamNormalize(Triangle)
    - OsdPatchParamUnnormalizeTriangle()
    - OsdEvaluatePatchBasisNormalized()
    - OsdEvaluatePatchBasis()

Osd extensions for patch tessellation common to all shaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - OsdInterpolatePatchCoordTriangle()
    - OsdComputePerPatchVertexBoxSplineTriangle
    - OsdEvalPatchBezierTriangle()
    - OsdEvalPatchGregoryTriangle()
    - OsdGetTessLevelsUniformTriangle()
    - OsdEvalPatchBezierTessLevels()
    - OsdEvalPatchBezierTriangleTessLevels()
    - OsdGetTessParameterizationTriangle()

Other Changes
-------------

Improvements
~~~~~~~~~~~~
    - Added new build script (GitHub #1068)
    - Added support for newer DirectX SDKs (GitHub #1066)
    - Patch arrays extended to support combined regular and irregular types (GitHub #995)
    - Far::PatchTables and adaptive refinement supported for Bilinear scheme (GitHub #1035)
    - New Far::PatchTableFactory method to determine adaptive refinement options ((GitHub #1047)
    - New Far::PatchTableFactory options to align primvar buffers of uniform tables (GitHub #986)
    - Far::StencilTable::UpdateValues() overloaded to support separate base buffer (GitHub #1011)
    - Far::LimitStencilTableFactory updated to create face-varying tables (GitHub #1012)
    - Regular patches on boundaries no longer require additional isolation (GitHub #1025)
    - Inclusion of OpenSubdiv header files in source code now consistent (GitHub #767)
    - Re-organization of and additions to Far tutorials (GitHub #1083)
    - Examples now use common command-line conventions and parsing (GitHub #1056)

Bug Fixes
~~~~~~~~~
    - Fixed Far::PrimvarRefiner internal limitFVar() prototype (GitHub #979)
    - Fixed Far::StencilTable append when base StencilTable empty (GitHub #982)
    - Patches around non-manifold vertices now free of cracks (GitHub #1013)

