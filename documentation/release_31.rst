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


Overview of Release 3.1
=======================

.. contents::
   :local:
   :backlinks: none

New Features
------------

Bicubic Face-Varying Patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The motivation for this feature is to improve drawing and evaluation of face-varying
primvar values for all
`face-varying interpolation options <subdivision_surfaces.html#face-varying-interpolation-rules>`__.

A common use of non-linear face-varying data is to capture a UV projection on
the surface.  The following example shows a simple shape with the face-varying
interpolation option assigned to a non-linear choice to achieve the desired
projection (in this case FVAR_LINEAR_NONE):

+----------------------------------------------+----------------------------------------------+
| .. image:: images/fvar_patch_linearall.png   | .. image:: images/fvar_patch_linearnone.png  |
|    :align:  center                           |    :align:  center                           |
|    :width:  75%                              |    :width:  75%                              |
|    :target: images/fvar_patch_linearall.png  |    :target: images/fvar_patch_linearnone.png |
|                                              |                                              |
| Linear Face-Varying Patches                  | Bicubic Face-Varying Patches                 |
+----------------------------------------------+----------------------------------------------+

The result on the left shows the old linearly interpolated patches, which
ignores any non-linear settings.  The result on the right shows the new use of
bicubic face-varying patches to accurately interpolate the desired projection.

Generation of a full face-varying patch representation can be enabled using a new option
in Far::PatchTableFactory::Options.  Additionally, topological refinement can be improved
to consider fvar channel topology using a new option in Far::TopologyRefiner::AdaptiveOptions.  See the API additions below and their associated Doxygen text
for more details.

Evaluation of patch basis weights for all patch types as been added to the GPU shader
source provided by Osd::GLSLPatchShaderSource, and Osd::HLSLPatchShaderSource.

Use of non-linear face-varying patches increases the storage size of the patch table and may also require additional data access and computation while drawing.

Varying and Face-Varying Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature extends existing evaluation methods to support evaluation of varying and face-varying
primvar data at arbitrary limit surface locations.

+-----------------------------------------+-----------------------------------------+
| .. image:: images/eval_varying.png      | .. image:: images/eval_facevarying.png  |
|    :align:  center                      |    :align:  center                      |
|    :width:  75%                         |    :width:  75%                         |
|    :target: images/eval_varying.png     |    :target: images/eval_facevarying.png |
|                                         |                                         |
| Varying Primvar Evaluation              | Face-Varying Primvar Evaluation         |
+-----------------------------------------+-----------------------------------------+

The image on the left shows evaluation of varying primvar values and the image on the right
shows evaluation of face-varying primvar values.

The EvaluateBasis API of Far::PatchTable has been extended as well as the OSD Evaluator API.

Second Order Derivative Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature extends the Far::LimitStencilTable and Far::PatchTable interfaces to support
evaluation of 2nd order partial derivatives.  The generation of derivative values
for Far::LimitStencilTable is controlled by new options that can be specified when creating
the stencil table.

Additionally, the implementation exposes a more accurate method to compute derivatives
for Gregory basis patches.  This can be enabled using the CMake configuration and
compile time definition OPENSUBDIV_GREGORY_EVAL_TRUE_DERIVATIVES.

Separate Levels of Feature Isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary motivation for this feature is to reduce the number of patches generated by
adaptive refinement when they can be adequately represented at a lower level.  A single
level of isolation must be as high as the greatest semi-sharp feature to properly resolve
the shape of that feature.  That high isolation level generates many unnecessary patches
for smooth extra-ordinary vertices.

In the following example, a single semi-sharp vertex is refined to level 5:

+--------------------------------------+--------------------------------------+
| .. image:: images/sec_level_off.png  | .. image:: images/sec_level_on.png   |
|    :align:  center                   |    :align:  center                   |
|    :width:  75%                      |    :width:  75%                      |
|    :target: images/sec_level_off.png |    :target: images/sec_level_on.png  |
|                                      |                                      |
| Single Isolation Level 5             | Primary Level 5, Secondary Level 2   |
+--------------------------------------+--------------------------------------+

Single isolation to level 5 on the left results in 312 patches.  The right shows the
semi-sharp feature isolated to 5, but with the new "secondary level" set to 2, the
number of patches is reduced to 123.

The second specified level of adaptive refinement is used
to halt isolation for features that typically do not require the specified maximum.
These include interior and boundary extra-ordinary vertices and those infinitely sharp
patches that correspond to boundary extra-ordinary patches.

The secondary level is available as a new option in Far::TopologyRefiner::AdaptiveOptions.


Sharp Patches for Infinitely Sharp Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The motivation for sharp patches is to accurately represent the limit surface of
infinitely sharp features, which otherwise can only be approximated by very high levels
of adaptive refinement, resulting in many unnecessary patches.

The true limit surface for regular faces along an infinitely sharp crease is a
regular B-Spline patch -- the same as regular faces along a boundary.
Similarly, the limit surface for faces around an extra-ordinary vertex on an infinitely
sharp crease is the same as that of faces around an extra-ordinary vertex on a boundary.
So these patches are identified and isolated to the same degree -- the regular patches
as soon as possible, and the irregular patches to the depth specified.

Consider the following (regression/shape/catmark_cube_creases2):

+------------------------------------+------------------------------------+------------------------------------+
| .. image:: images/inf_sharp_a.png  | .. image:: images/inf_sharp_b.png  | .. image:: images/inf_sharp_c.png  |
|    :align:  center                 |    :align:  center                 |    :align:  center                 |
|    :width:  100%                   |    :width:  100%                   |    :width:  100%                   |
|    :target: images/inf_sharp_a.png |    :target: images/inf_sharp_b.png |    :target: images/inf_sharp_c.png |
|                                    |                                    |                                    |
| Level 5 without Sharp Patches      | Level 5 with Sharp Patches         | Level 2 with Sharp Patches         |
+------------------------------------+------------------------------------+------------------------------------+

Without use of sharp patches on the left, isolating to level 5 generates 1764 patches and does still
not capture the sharp edges.  With sharp patches in the center, isolating to the same degree (level
5) reduces the number of patches to 96 and captures the sharp edges.  The sharp features can be
captured at a lower degree with comparable accuracy as illustrated on the right where isolation to
level 2 further reduces the number of patches to 42.

The use of infinitely sharp patches can be enabled both at a high level as an new option to Osd::Mesh,
or more directly when adaptively refining or construction the patch tables in
Far::TopologyRefiner::AdaptiveOptions and Far::PatchTableFactory::Options.

Given the improved accuracy and reduced patches by the use of simple regular patches, we would prefer
that this be the default behavior, but it was made an explicit option in order to avoid disrupting
existing usage.  In a future major release this feature will hopefully be the norm.


API Additions
-------------

See associated `Doxygen <doxy_html/index.html>`__ for full details.

Osd::CpuEvaluator, GLComputeEvaluator, etc:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - EvalPatchesVarying()
    - EvalPatchesFaceVarying()

Osd::CpuPatchTable, GLPatchTable, etc:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - GetFVarPatchArrayBuffer()
    - GetFVarPatchIndexBuffer()
    - GetFVarPatchIndexSize()
    - GetFVarPatchParamBuffer()
    - GetFVarPatchParamSize()
    - GetNumFVarChannels()
    - GetVaryingPatchArrayBuffer()
    - GetVaryingPatchIndexBuffer()
    - GetVaryingPatchIndexSize()

Osd::MeshBits:
~~~~~~~~~~~~~~
    - member MeshFVarAdaptive
    - member MeshUseInfSharpPatch

Osd::PatchParam
~~~~~~~~~~~~~~~
    - IsRegular()
    - Unnormalize()
    - extensions to Set()

Osd::GLSLPatchShaderSource, HLSLPatchShaderSource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - GetPatchBasisShaderSource()

Far::LimitStencil
~~~~~~~~~~~~~~~~~
    - GetDuuWeights()
    - GetDuvWeights()
    - GetDvvWeights()
    - extensions to LimitStencil()

Far::LimitStencilTable
~~~~~~~~~~~~~~~~~~~~~~
    - GetDuuWeights()
    - GetDuvWeights()
    - GetDvvWeights()
    - Update2ndDerivs()
    - extensions to LimitStencilTable()

Far::LimitStencilTableFactory::Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - member generate1stDerivatives
    - member generate1stDerivatives
    - extensions to LimitTableStencilFactory::Create()

Far::PatchParam
~~~~~~~~~~~~~~~
    - IsRegular()
    - Unnormalize()
    - extensions to Set()

Far::PatchTable
~~~~~~~~~~~~~~~
    - ComputeLocalPointValuesFaceVarying()
    - ComputeLocalPointValuesVarying()
    - GetFVarPatchDescriptor()
    - GetFVarPatchParam()
    - GetNumLocalPointsFaceVarying()
    - GetNumLocalPointsVarying()
    - GetPatchArrayVaryingVertices()
    - GetPatchArrayFVarPatchParam()
    - GetPatchArrayFVarValues()
    - GetPatchFVarPatchParam()
    - GetPatchVaryingVertices()
    - GetVaryingPatchDescriptor()
    - GetVaryingVertices()
    - EvaluateBasisFaceVarying()
    - EvaluateBasisVarying()
    - extensions to EvaluateBasis()

Far::PatchTableFactory::Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - member useInfSharpPatch
    - member genenerateFVarLegacyLinearPatches

Far::TopologyLevel
~~~~~~~~~~~~~~~~~~
    - DoesEdgeFVarTopologyMatch()
    - DoesFaceFVarTopologyMatch()
    - DoesVertexFVarTopologyMatch()
    - IsEdgeBoundary()
    - IsEdgeNonManifold()
    - IsVertexBoundary()
    - IsVertexNonManifold()

Far::TopologyRefiner::AdaptiveOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - member secondaryLevel
    - member useInfSharpPatch
    - member considerFVarChannels


Other Changes
-------------

Improvements
~~~~~~~~~~~~
    - Enabled the use of CMake's folder feature
    - Removed the use of iso646 alternative keywords ('and', 'or', 'not', etc.) to improve portability
    - Added numerical valued preprocessor directives (OPENSUBDIV_VERSION_MAJOR, etc.) to <opensubdiv/version.h>
    - Improved documentation for Far::PatchParam and added Unnormalize() to complement Normalize()
    - Added additional topology queries to Far::TopologyLevel
    - Updated glFVarViewer and glEvalLimit viewer to make use of bicubic face-varying patches
    - Updated glViewer and dxViewer to add a toggle for InfSharpPatch
    - Updated dxPtexViewer for improved feature parity with glPtexViewer
    - Improved far_regression to exercise shapes independent of Hbr compatibility
    - Added support for Appveyor continuous integration testing
    - Removed cmake/FindIlmBase
    - Removed mayaPolySmooth example

Bug Fixes
~~~~~~~~~
    - Fixed Ptex version parsing and compatibility issues
    - Fixed compatibility issues with VS2015
    - Fixed bug interpolating face-varying data with Bilinear scheme
    - Fixed bug with refinement using Chaikin creasing
    - Fixed bugs with HUD sliders in the example viewers
