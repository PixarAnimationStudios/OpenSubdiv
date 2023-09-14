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


3.0 - 3.6 Release Notes
-----------------------

.. contents::
   :local:
   :backlinks: none

----

Release 3.6
~~~~~~~~~~~

Release 3.6.0 - Sep 2023
==============================

Release 3.6.0 is a significant release with new features, several
configuration improvements, and bug fixes.
For more information on the following, see `Release 3.6 <release_36.html>`__

**Changes**
    - Updated Osd patch drawing shader source to exclude legacy shader constructs to improve compatibility with Vulkan, DX12, etc. (GitHub #1320)
    - Installed Osd patch evaluation headers to allow use from client shaders and compute kernels (GitHub #1321)
    - Updated CMake build to locate TBB using TBB's CMake config in order to support oneTBB (GitHub #1319)
    - Updated CMake FindOpenCL module to support parsing version information from recent OpenCL headers (GitHub #1322)
    - Removed obsolete .travis.yml (GitHub #1324)

**Bug Fixes**
    - Fixed inconsistent warning levels for MSVC builds when using Ninja (GitHub #1318)
    - Fixed documentation build errors when using Ninja (GitHub #1323)
    - Fixed build errors resulting from oneTBB API changes (GitHub #1317)

Release 3.5
~~~~~~~~~~~

Release 3.5.1 - July 2023
=========================

Release 3.5.1 is a minor release including bug fixes and configuration improvements.

**Changes**
    - Updated CMake to set fallback CMAKE_CXX_STANDARD to C++14 (GitHub #1276)
    - Updated CMake with OpenGL import targets to avoid link errors (GitHub #1277)
    - Updated CMake to set gpu architecture fallback only for older CUDA versions (GitHub #965 #1299)
    - Updated CMake to use append for CMAKE_MODULE_PATH (GitHub #1296)
    - Fixed interface includes for CMake config (GitHub #1278)
    - Fixed warnings with newer and stricter use of Clang (GitHub #1275 #1289 #1290)
    - Fixed potential float constant cast errors for OpenCL (GitHub #1285)
    - Fixed generation of Apple Frameworks with no OSD_GPU targets enabled (GitHub #1224 #1236)
**Bug Fixes**
    - Fixed Bfr::Surface construction bug for rare topological case (GitHub #1301)
    - Fixed CUDA example dependencies with GLX on Linux (GitHub #1294)

Release 3.5.0 - Sep 2022
========================

Release 3.5.0 is a significant release with new features, several
configuration improvements, and a few other improvements and bug fixes.
For more information on the following, see `Release 3.5 <release_35.html>`__

**Deprecation Announcements**
    - Hbr is deprecated and will be removed from subsequent releases

**New Features**
    - Simplified Surface Evaluation (Bfr)
    - Tessellation Patterns (Bfr)

**Changes**
    - Suppression of GCC compiler warnings (GitHub #1253, #1254, #1270)
    - Additional methods for Far::TopologyLevel (GitHub #1227, #1255)
    - Improved mixed partial derivative at Gregory patch corners (GitHub #1252)
    - Minor improvements to Far tutorials (GitHub #1226, #1241)
    - Added CMake config (GitHub #1242)
    - Updated CMake minimum version to 3.12 (GitHub #1237, #1261)
    - Updated documentation build scripts for Python 3 (#1265, #1266)
    - Updated 'stringify' build tool for improved cross compilation support
      (GitHub #1267)
    - Added 'NO_MACOS_FRAMEWORKS' build option (GitHub #1238)
    - Updated Azure pipelines agents for Unbuntu and macOS (GitHub #1247, #1256)
    - Removed obsolete AppVeyor and Travis CI scripts (GitHub #1259)

**Bug Fixes**
    - Cache active program for Osd::GLComputeEvaluator (GitHub #1244)
    - Fixed member initialization warnings in Osd::D3D11ComputeEvaluator
      (GitHub #1239)
    - Fixed GLSL shader source to remove storage qualifiers from struct members
      (GitHub #1271)
    - Fixed use of CMake variables for Apple builds (GitHub #1235)
    - Fixed build errors when using OpenGL without GLFW (GitHub #1257)
    - Fixed links to embedded videos (GitHub #1231)

Release 3.4
~~~~~~~~~~~

Release 3.4.4 - Feb 2021
========================

Release 3.4.4 is a minor release including bug fixes and configuration improvements

**Changes**
    - The "master" branch on GitHub has been renamed "release" (GitHub #1218 #1219)
    - The CMake configuration has been updated to allow use as a sub-project (GitHub #1206)
    - Removed obsolete references to hbr from examples/farViewer (GitHub #1217)

**Bug Fixes**
    - Fixed bug with sparse PatchTables and irregular face-varying seams (GitHub #1203)
    - Fixed loss of precision when using double precision stencil tables (GitHub #1207)
    - Fixed reset of Far::TopologyRefiner::GetMaxLevel() after call to Unrefine() (GitHub #1208)
    - Fixed linking with -ldl on unix systems (GitHub #1196)
    - Fixed naming and installation of macOS frameworks (GitHub #1194 #1201)
    - Fixed GL version and extension processing and dynamic loading on macOS (GitHub #1216)
    - Fixed FindDocutils.cmake to be more robust (GitHub #1213 #1220)
    - Fixed errors using build_scripts/build_osd.py with Python3 (GitHub #1206)

Release 3.4.3 - Apr 2020
========================

Release 3.4.3 is a minor release including bug fixes and configuration improvements

**Changes**
    - GLEW is no longer required by default (GitHub #1183 #1184)
    - Removed false Ptex link dependency from libosdCPU (GitHub #1174)
    - Removed false GLFW link dependency from DX11 and Metal examples (GitHub #1178)
    - Removed link dependency on unused TBB libraries (GitHub #1064)
    - Added option to disable building of dynamic shared libraries (GitHub #1169)
    - Added new tutorial for Far::LimitStencilTable (GitHub #1176)
    - Updated use of EXT_direct_state_access to ARB_direct_state_access (GitHub #1184)
    - Fixed C++ strict aliasing warnings (GitHub #1182)
    - Fixed MSVC warnings in example code (GitHub #1158 #1172)
    - Fixed compatibility with Visual Studio 2019 (GitHub #1173 #1189)
    - Fixed CMake CMP0054 warnings (GitHub #1180)
    - Added prefix to OpenSubdiv CMake macros (GitHub #1157)
    - Moved utilities in examples/common to regression/common (GitHub #1167)
    - Minor fixes to Far tutorials (GitHub #1175 #1177)
    - Switched to Azure Pipelines for continuous integration testing instead of Travis-CI and AppVeyor (GitHub #1168 #1190)

**Bug Fixes**
    - Fixed selective boundary interpolation for case Sdc::Options::VTX_BOUNDARY_NONE (GitHub #1170 #1171)
    - Fixed static library linking to address missing symbols (GitHub #1192)
    - Additional fixes for dynamic and static linking (GitHub #1193)

Release 3.4.0 - Jun 2019
========================

Release 3.4.0 is a significant release with several new features, bug fixes, and general
code and configuration improvements.  For more information on the following, please see
`Release 3.4 <release_34.html>`__

**New Features**
    - Triangular Patches for Loop subdivision
    - Improvements to Introductory Documentation
    - Sparse Patch Tables and Adaptive Refinement
    - Full Support for Double Precision in Far

**Changes**
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
    - examples now use common command-line conventions and parsing (GitHub #1056)

**Bug Fixes**
    - Fixed Far::PrimvarRefiner internal limitFVar() prototype (GitHub #979)
    - Fixed Far::StencilTable append when base StencilTable empty (GitHub #982)
    - Patches around non-manifold vertices now free of cracks (GitHub #1013)

Release 3.3
~~~~~~~~~~~

Release 3.3.3 - Jul 2018
========================

Release 3.3.3 is bug-fix release addressing regressions from release 3.3.2

**Bug Fixes**
    - Fixed a regression in PatchTable construction with varying patches (GitHub #976)
    - Fixed a regression in PatchTable construction for face-varying patches (GitHub #972)
    - Fixed a bug in the initialization of Far::SourcePatch (GitHub #971)

Release 3.3.2 - Jun 2018
========================

Release 3.3.2 is a minor release with potentially significant performance
improvements to the patch pre-processing stages

**Changes**
    - Improved performance of PatchTable construction (GitHub #966)
    - The resulting improved accuracy will produce slight numerical differences in computations involving patches, e.g. StencilTable and PatchTable evaluation

**Bug Fixes**
    - Far::PatchTableFactory now supports PatchTable construction with ENDCAP_BILINEAR_BASIS specified

Release 3.3.1 - Feb 1018
========================

Release 3.3.1 is a minor bug-fix release

**Bug Fixes**
    - Fixed GLSL/HLSL/Metal patch shader code to resolve degenerate normals (GitHub #947)
    - Fixed problems with face-varying patches in uniform PatchTables (GitHub #946)
    - Fixed integer overflow bugs for large meshes in PatchTable factories (GitHub #957)
    - Fixed computation of PatchParam for triangle refinement (GitHub #962)

**Changes**
    - Added build options: NO_GLFW and NO_GLFW_X11
    - Added additional shapes with infinitely sharp creases to the Metal and DX11 example viewers
    - Disabled GL tests during CI runs on Linux
    - Improved stability of examples/glImaging in CI runs by testing GL version

Release 3.3.0 - Aug 2017
========================

Release 3.3.0 is significant release adding an Osd implementation for Apple's Metal API

**New Features**
    - Added an Osd implementation for Apple's Metal API
    - Added the mtlViewer example

**Changes**
    - Fixed several instances of local variable shadowing that could cause build warnings
    - Updated continuous-integration build scripts and added testing on macOS

Release 3.2
~~~~~~~~~~~

Release 3.2.0 - Feb 2017
========================

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

Release 3.1
~~~~~~~~~~~

Release 3.1.1 - Jan 2017
========================

Release 3.1.1 is a minor bug-fix release.

**Bug Fixes**
    - Fixed a bug with non-manifold face-varying topology causing a crash during patch table creation
    - Fixed GLEW compilation and linking with dynamic GLEW libraries on Windows
    - Fixed GLFW linking with GLFW 3.2 on X11 platforms

Release 3.1.0 - Oct 2016
========================

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

Release 3.0
~~~~~~~~~~~

Release 3.0.5 - Mar 2016
========================

Release 3.0.5 is a minor stability release with performance and correctness bug fixes.

**Bug Fixes**
    - The previous release reduced transient memory use during PatchTable construction, but increased the amount of memory consumed by the resulting PatchTable itself, this regression has been fixed.
    - The example Ptex texture sampling code has been fixed to prevent sampling beyond the texels for a face when multisample rasterization is enabled.

Release 3.0.4 - Feb 2016
========================

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

Release 3.0.3 - Oct 2015
========================

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

Release 3.0.2 - Aug 2015
========================

Release 3.0.2 is a minor release for a specific fix.

**Bug Fixes**
    - Fixed drawing of single crease patches

Release 3.0.1 - Aug 2015
========================

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

Release 3.0.0 - Jun 2015
========================

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Previous releases <release_notes_2x.html>`_
