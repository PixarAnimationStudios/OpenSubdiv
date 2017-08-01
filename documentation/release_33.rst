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


Overview of Release 3.3
=======================

.. contents::
   :local:
   :backlinks: none

New Features
------------

Metal Compute and Tessellation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added support for drawing and evaluation using Apple's Metal API for
high performance and low-overhead GPU access.

This includes the full set of Osd interfaces needed to draw using
Metal graphics rendering (including tessellation shaders) and evaluate
stencils and patch tables using Metal compute processing.

Also includes an example mtlViewer that can be built to run on either macOS or iOS.

+----------------------------------------------------+
| .. image:: images/osd_metal_tess.png               |
|    :align:  center                                 |
|    :width:  75%                                    |
|    :target: images/osd_metal_tess.png              |
|                                                    |
| Metal Graphics Rendering with Tessellation         |
+----------------------------------------------------+

+----------------------------------------------------+
| .. image:: images/osd_metal_eval.png               |
|    :align:  center                                 |
|    :width:  75%                                    |
|    :target: images/osd_metal_eval.png              |
|                                                    |
| Metal Patch Evaluation using Compute Processing    |
+----------------------------------------------------+

API Additions
-------------

See associated `Doxygen <doxy_html/index.html>`__ for full details.

Osd::MTLContext
~~~~~~~~~~~~~~~
    - device and commandQueue member data

Osd::MTLComputeEvaluator
~~~~~~~~~~~~~~~~~~~~~~~~
    - Create()
    - EvalStencils()
    - EvalPatches()
    - EvalPatchesVarying()
    - EvalPatchesFaceVarying()
    - Compile()
    - Synchronize()

Osd::MTLStencilTable
~~~~~~~~~~~~~~~~~~~~
    - Create()
    - GetSizesBuffer()
    - GetIndicesBuffer()
    - GetDuWeightsBuffer()
    - GetDvWeightsBuffer()
    - GetDuuWeightsBuffer()
    - GetDuvWeightsBuffer()
    - GetDvvWeightsBuffer()
    - GetNumStencils()

Osd::MTLMeshInterface
~~~~~~~~~~~~~~~~~~~~~
    - GetNumVertices()
    - GetMaxValence()
    - UpdateVertexBuffer()
    - UpdateVaryingBuffer()
    - Refine()
    - Synchronize()
    - GetPatchTable()
    - GetFarPatchTable()
    - BindVertexBuffer()
    - BindVaryingBuffer()

Osd::MTLPatchTable
~~~~~~~~~~~~~~~~~~
    - Create()
    - GetPatchArrays()
    - GetPatchIndexBuffer()
    - GetPatchParamBuffer()
    - GetVaryingPatchArrays()
    - GetVaryingPatchIndexBuffer()
    - GetNumFVarChannels()
    - GetFVarPatchArrays()
    - GetFVarPatchIndexBuffer()
    - GetFVarPatchParamBuffer()

Osd::MTLLegacyGregoryPatchTable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - UpdateVertexBuffer()
    - GetVertexBuffer()
    - GetVertexValenceBuffer()
    - GetQuadOffsetsBuffer()

Osd::PatchShaderSource
~~~~~~~~~~~~~~~~~~~~~~
    - GetCommonShaderSource()
    - GetPatchBasisShaderSource()
    - GetVertexShaderSource()
    - GetHullShaderSource()
    - GetDomainShaderSource()

Osd::CPUMTLVertexBuffer
~~~~~~~~~~~~~~~~~~~~~~~
    - Create()
    - UpdateData()
    - GetNumElements()
    - GetNumVertices()
    - BindCpuBuffer()
    - BindMTLBuffer()
    - BindVBO()

Other Changes
-------------

Improvements
~~~~~~~~~~~~
    - Fixed several instances of local variable shadowing that could cause build warnings
    - Updated continuous-integration build scripts and added testing on macOS

