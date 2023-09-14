..
     Copyright 2023 Pixar

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


Overview of Release 3.6
=======================

.. contents::
   :local:
   :backlinks: none

New Features
------------

The purpose of this release is to address concerns which
improve support for current typical use cases and
provide support for significant new use cases.

Modern Graphics APIs and Parallel Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenSubdiv is organized as a set of components for working with
subdivision surface representations, i.e. Bfr, Far, Vtr, Sdc along with
a collection of back-end components in Osd supporting the use of
specific low-level subsystems to work with subdivision surface data.

These back-end components in Osd have taken many forms including in
some cases specific complete GPU shaders along with controller classes
to manage compilation and execution of these shaders.

In practice it has been more effective for Osd to simply supply the
functions needed to operate on subdivision surface data, allowing
the client application or client library to take care of using these
functions from client provided shaders and computation kernels using
client provided execution controllers.

This has been the direction for the Osd library for some time and the
changes implemented in this release make this even more straightforward.

The existing methods:

    - Osd::GLSLPatchShaderSource::GetPatchBasisShaderSource()
    - Osd::HLSLPatchShaderSource::GetPatchBasisShaderSource()
    - Osd::MTLPatchShaderSource::GetPatchBasisShaderSource()

continue to return shader source strings at runtime which contain
definitions and functions allowing client shader code to evaluate
values and first and second derivatives on the piecewise parametric
patches resulting from subdivison refinement.

The identical code is now available at compile time as:

   - opensubdiv/osd/patchBasis.h

and is essentially a "shader" interface that can be used from client
kernels including those implemented using TBB, CUDA, C++, etc.

Similarly, the new methods:

    - Osd::GLSLPatchShaderSource::GetPatchDrawingShaderSource()
    - Osd::HLSLPatchShaderSource::GetPatchDrawingShaderSource()
    - Osd::MTLPatchShaderSource::GetPatchDrawingShaderSource()

return shader source strings at runtime which contain definitions
and functions allowing clients to draw the piecewise parametric
patches resulting from subdivision, e.g. using GPU tessellation
shaders or GPU mesh shaders.

The returned shader source has been stripped of resource binding
and other potentially problematic defintions since these are usually
best handled by client shader code.

These methods have been tested successfully with new client code
using Vulkan and DirectX 12 in addition to existing client code
using OpenGL, Metal, DirectX 11, etc.

Updated Third-party APIs and Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the methods described above will be the most effective way
to use OpenSubdiv, some of the existing back-end components have been
updated to accommodate evolving third-party APIs and tools.

Specifically, the TBB implementation has been updated to allow
use with the oneTBB API while continuing to maintain
compatibility with earlier releases of TBB.

Also, there have been minor fixes to the CMake build to accommodate
using the Ninja build system and also systems with OpenCL 3.0.

API Additions
-------------

See associated `Doxygen <doxy_html/index.html>`__ for full details.

Additions to Osd::PatchShaderSource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - GLSLPatchShaderSource::GetPatchDrawingShaderSource()
    - HLSLPatchShaderSource::GetPatchDrawingShaderSource()
    - MTLPatchShaderSource::GetPatchDrawingShaderSource()

Osd extensions for patch evaluation from client shaders and compute kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

Other Changes
-------------

Deprecation Announcements
~~~~~~~~~~~~~~~~~~~~~~~~~
    - The methods Osd::TbbEvaluator::SetNumThreads() and Osd::OmpEvaluator::SetNumThreads() have been marked deprecated.

Improvements
~~~~~~~~~~~~
    - Updated Osd patch drawing shader source to exclude legacy shader constructs to improve compatibility with Vulkan, DX12, etc. (GitHub #1320)
    - Installed Osd patch evaluation headers to allow use from client shaders and compute kernels (GitHub #1321)
    - Updated CMake build to locate TBB using TBB's CMake config in order to support oneTBB (GitHub #1319)
    - Updated CMake FindOpenCL module to support parsing version information from recent OpenCL headers (GitHub #1322)
    - Removed obsolete .travis.yml (GitHub #1324)

Bug Fixes
~~~~~~~~~
    - Fixed inconsistent warning levels for MSVC builds when using Ninja (GitHub #1318)
    - Fixed documentation build errors when using Ninja (GitHub #1323)
    - Fixed build errors resulting from oneTBB API changes (GitHub #1317)
