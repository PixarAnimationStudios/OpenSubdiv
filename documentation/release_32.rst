..
     Copyright 2017 Pixar

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


Overview of Release 3.2
=======================

.. contents::
   :local:
   :backlinks: none

New Features
------------

Face-Varying Stencil Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Face-Varying primvar values may now be refined using stencil tables.

The stencil table for a face-varying channel is created by specifying the desired fvarChannel and setting
the Far::StencilTableFactory::Option interpolationMode to INTERPOLATE_FACE_VARYING when creating the stencil table.

1st and 2nd Derivative Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Osd Evaluator API has been extended to support 1st derivative and 2nd partial derivative evaluation for stencils and patches.

+----------------------------------------------------+------------------------------------------------------+
| .. image:: images/osd_eval_1st_deriv_normal.png    | .. image:: images/osd_eval_2nd_deriv_curvature.png   |
|    :align:  center                                 |    :align:  center                                   |
|    :width:  75%                                    |    :width:  75%                                      |
|    :target: images/osd_eval_1st_deriv_normal.png   |    :target: images/osd_eval_2nd_deriv_curvature.png  |
|                                                    |                                                      |
| 1st Derivative Surface Normal                      | 2nd Derivative Surface Curvature                     |
+----------------------------------------------------+------------------------------------------------------+

On the left is an example of computing a surface normal at each point using the evaluated 1st derivatives,
while on the right is an example of computing surface curvature at each point using the evaluated 2nd partial derivatives.

Smooth Corner Patch
~~~~~~~~~~~~~~~~~~~

An option has been added to disable the legacy behavior of generating a sharp-corner patch at a smooth corner.
Corners which are actually sharp will continue to generate sharp-corner patches.

The differences between the two methods is most apparent at low-levels of feature isolation.

This feature is controlled by the generateLegacySharpCornerPatches option added to Far::PatchTableFactory::Options.

+------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/far_legacy_sharp_corner_patch_true.png   | .. image:: images/far_legacy_sharp_corner_patch_false.png   |
|    :align:  center                                         |    :align:  center                                          |
|    :width:  75%                                            |    :width:  75%                                             |
|    :target: images/far_legacy_sharp_corner_patch_true.png  |    :target: images/far_legacy_sharp_corner_patch_false.png  |
|                                                            |                                                             |
| Sharp Corner Patch (legacy behavior)                       | Smooth Corner Patch                                         |
+------------------------------------------------------------+-------------------------------------------------------------+

On the left is the legacy behavior of generating sharp corner patches at smooth corners.
The image on the right shows the correct smooth corner patches generated when this legacy behavior is disabled.

API Additions
-------------

See associated `Doxygen <doxy_html/index.html>`__ for full details.

Osd::CpuEvaluator, GLComputeEvaluator, etc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Create()
    - EvalStencils()
    - EvalPatches()
    - EvalPatchesVarying()
    - EvalPatchesFaceVarying()

Osd::Mesh
~~~~~~~~~
    - Create()

Osd::MeshBits
~~~~~~~~~~~~~
    - member MeshUseSmoothCornerPatch

Far::PatchTableFactory::Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - member generateLegacySharpCornerPatches

Far::StencilTableFactory
~~~~~~~~~~~~~~~~~~~~~~~~
    - enumeration Mode::INTERPOLATE_FACE_VARYING
    - AppendLocalPointStencilTableFaceVarying()

Far::StencilTableFactory::Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - member fvarChannel

Other Changes
-------------

Improvements
~~~~~~~~~~~~
    - Corrected numerous spelling errors in doxygen comments
    - Updated glFVarViewer with improved error detection and command line parsing
    - Added option to build using MSVC with static CRT

Bug Fixes
~~~~~~~~~
    - Fixed a double delete of GL program in Osd::GLComputeEvaluator
