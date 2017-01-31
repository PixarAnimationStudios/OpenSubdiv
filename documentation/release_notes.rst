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


3.0 - 3.2 Release Notes
-----------------------

.. contents::
   :local:
   :backlinks: none

----

Release 3.2.0
=============

Release 3.2.0 is a minor release containing API additions and bug fixes

**New Features**
    - Extended Far::StencilTableFactory to support face-varying
    - Extended Osd Evaluator classes to support evaluation of 1st and 2nd derivatives
    - Added an option to disable generation of legacy sharp corner patches

**Changes**
    - Corrected numerous spelling errors in doxygen comments
    - Updated glFVarViewer with improved error detection and command line parsing
    - Added option to build using MSVC with static CRT

**Bug Fixes**
    - Fixed a double delete of GL program in Osd::GLComputeEvaluator

Release 3.1.1
=============

Release 3.1.1 is a minor bug-fix release.

**Bug Fixes**
    - Fixed a bug with non-manifold face-varying topology causing a crash during patch table creation
    - Fixed GLEW compilation and linking with dynamic GLEW libraries on Windows
    - Fixed GLFW linking with GLFW 3.2 on X11 platforms

Release 3.1.0
=============

Release 3.1.0 is a significant release with several new features, bug fixes, and general
code and configuration improvements.  For more information on the following, please see
`Release 3.1 <release_31.html>`__

**New Features**
    - Bicubic Face-Varying Patches
    - Varying and Face-Varying Evaluation
    - Second Order Derivative Evaluation
    - Separate Levels of Feature Isolation
    - Sharp Patches for Infinitely Sharp Features

**Changes**
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

**Bug Fixes**
    - Fixed Ptex version parsing and compatibility issues
    - Fixed compatibility issues with VS2015
    - Fixed bug interpolating face-varying data with Bilinear scheme
    - Fixed bug with refinement using Chaikin creasing
    - Fixed bugs with HUD sliders in the example viewers

Release 3.0.5
=============

Release 3.0.5 is a minor stability release with performance and correctness bug fixes.

**Bug Fixes**
    - The previous release reduced transient memory use during PatchTable construction, but increased the amount of memory consumed by the resulting PatchTable itself, this regression has been fixed.
    - The example Ptex texture sampling code has been fixed to prevent sampling beyond the texels for a face when multisample rasterization is enabled.

Release 3.0.4
=============

Release 3.0.4 is a minor stability release which includes important performance
and bug fixes.

**New Features**
    - Added accessor methods to Far::LimitStencilTable to retrieve limit stencil data including derivative weights
    - Added support for OpenCL event control to Osd::CLVertexBuffer and Osd::CLEvaluator

**Changes**
    - Major reduction in memory use during Far::PatchTable construction for topologies with large numbers of extraordinary features
    - Improved performance for GL and D3D11 tessellation control / hull shader execution when drawing BSpline patches with the single crease patch optimization enabled

**Bug Fixes**
    - Restored support for drawing with fractional tessellation
    - Fixed far_tutorial_6 to refine primvar data only up to the number of levels produced by topological refinement
    - Fixed build warnings and errors reported by Visual Studio 2015

Release 3.0.3
=============

Release 3.0.3 is a minor stability release which includes important performance
and bug fixes.

**New Features**
    - Smooth normal generation tutorial, far_tutorial_8

**Changes**
    - Major performance improvement in PatchTable construction
    - Improved patch approximations for non-manifold features

**Bug Fixes**
    - Fixed double delete in GLSL Compute controller
    - Fixed buffer layout for GLSL Compute kernel
    - Fixed GL buffer leak in Osd::GLPatchTable
    - Fixed out-of-bounds data access for TBB and OMP stencil evaluation
    - Fixed WIN32_LEAN_AND_MEAN typo
    - Fixed Loop-related shader issues glFVarViewer

Release 3.0.2
=============

Release 3.0.2 is a minor release for a specific fix.

**Bug Fixes**
    - Fixed drawing of single crease patches

Release 3.0.1
=============

Release 3.0.1 is a minor release focused on stability and correctness.

**Changes**
    - Added a references section to the documentation, please see `References <references.html>`__
    - Removed references to AddVaryingWithWeight from examples and tutorials
    - Added more regression test shapes
    - Addressed general compiler warnings (e.g. signed vs unsigned comparisons)
    - Addressed compiler warnings in the core libraries reported by GCC's -Wshadow
    - Eased GCC version restriction, earlier requirement for version 4.8 or newer is no longer needed
    - Replaced topology initialization assertions with errors
    - Improved compatibility with ICC
    - Improved descriptive content and formatting of Far error messages
    - Improved build when configured to include no GPU specific code

**Bug Fixes**
    - Fixed handling of unconnected vertices to avoid out of bounds data access
    - Fixed non-zero starting offsets for TbbEvalStencils and OmpEvalStencils
    - Fixed Far::StencilTableFactory::Options::factorizeIntermediateLevels
    - Fixed Far::PatchTablesFactory::Options::generateAllLevels
    - Fixed the behavior of VTX_BOUNDARY_NONE for meshes with bilinear scheme
    - Fixed some template method specializations which produced duplicate definitions
    - Disabled depth buffering when drawing the UI in the example viewers
    - Disabled the fractional tessellation spacing option in example viewers
      since this mode is currently not supported

Release 3.0.0
=============

Release 3.0.0 is a major release with many significant improvements and
changes.  For more information on the following, please see
`Release 3.0 <release_30.html>`__

**New Features**
    - Faster subdivision using less memory
    - Support for non-manifold topology
    - Face-Varying data specified topologically
    - Elimination of fixed valence tables
    - Single-crease patch for semi-sharp edges
    - Additional irregular patch approximations
    - Introduction of Stencil Tables
    - Faster, simpler GPU kernels
    - Unified adaptive shaders
    - Updated coding style with namespaces
    - More documentation and tutorials

**Bug Fixes**
    - Smooth Face-Varying interpolation around creases


Release 3.0.0 RC2
=================

**New Features**
    - Documentation updates
    - far_tutorial_3 updates for the multiple face-varying channels
    - maya example plugin interpolates a UV channel and a vertex color channel

**Bug Fixes**
    - Fixed a LimitStencilTableFactory bug, which returns an invalid table
    - PatchParam encoding changed to support refinement levels up to 10
    - Added Xinerama link dependency
    - Fixed MSVC 32bit build problem
    - Fixed minor cmake issues
    - Fixed glViewer/farViewer stability bugs


Release 3.0.0 RC1
=================

**Changes**
    - Far::TopologyRefiner was split into several classes to clarify and focus
      the API.
    - Interpolation of Vertex and Varying primvars in a single pass is no longer
      supported.
    - The Osd layer was largely refactored.

Previous 2.x Release Notes
==========================

`Previous releases <release_notes_2x.html>`_
