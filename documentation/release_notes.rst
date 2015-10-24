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


3.0 Release Notes
-----------------

.. contents::
   :local:
   :backlinks: none

----

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
`Introduction to 3.0 <intro_30.html>`__

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
