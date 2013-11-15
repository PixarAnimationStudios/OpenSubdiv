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


Release Notes
-------------

.. contents::
   :local:
   :backlinks: none

----

Release 2.3.2
=============

**New Features**
    - Adding control cage drawing to ptexViewer
    - Adding Maya osdPolySmooth plugin into OpenSubdiv examples. 

**Changes**
    - Removing some glGetError checks that are causing problems for Autodesk
    - D3D11DrawRegistry returns the common shader config for all non-tess patcharrays.
    - Updates to simple cpu osdutil classes

**Bug Fixes**
    - Fix Hbr Chaikin crease rule
    - Fix Chaikin tag parsing
    - Fix return value of allocate function for OsdCPUGLVertxBuffer
    - Fixed GLSL shader portability.
    - Fix FindGLFW.cmake for GLFW 3.03 on OSX
    - Fixed compiler warnings.
    - Fixed VS2010 build errors
    - Fixed WIN32 build error when no DXSDK installed.
    - Fix OSX build: stdlib.h needs to be included in glPtexMipmapTexture.h
    - Fix for crash in new mesh/refiner code in OsdUtil


Release 2.3.1
=============

**New Features**
    - Add DX11 version of ptex mipmap loader
    - Add DX11 ptex viewer (work in progress)
    - Add DX11 fractional partitioning, normal derivatives computation
    - Add memory usage controls to Ptex loader
    - Add face-varying boundary interpolation parsing to shape_utils
    - Add simple HbrMesh and FarMesh wrapper classes to osdutil

**Changes**
    - Amend language of attribution file 'NOTICE.txt'
    - Optimize a bit of ptex mipmap lookup.
    - Show ptex memory usage in GL and DX11 ptexViewers
    - Improve ptex guttering
    - Addding some video links to our collection of external resources

**Bug Fixes**
    - Fix edge-only face-varying interpolation
    - Fix Far to handle disconnected vertices in an Hbr mesh
    - Fixed ptex cache resource release sequence
    - Fix build symbol conflict in Far
    - Fix patch parambuffer generation in OsdD3D11DrawContext
    - Fix a minor osdutil build warning (seen with gcc 4.8.1)
    - Fix VS2010 build errors

Release 2.3.0
=============

**New Features**
    - Added Analytical displacement mapping ('Analytic Displacement Mapping using
      Hardware Tessellation; Niessner and Loop [TOG 2013])
    - Added a new ptex mipmap loader
    - Added face varying macros for loop subdivision
    - Added the uvViewer example to see how face varying interpolation rule works
    - Added a slider component and cleanup hud code.

**Changes**
    - Adding license & attribution files, improved language of the code headers
    - Install documentation into the Filesystem Hierarchy Standard location
    - Set GLFW_OPENGL_FORWARD_COMPAT on Mac OS to make samples work on that platform
    - Added surface normal mode & mipmap to ptxViewer

**Bug Fixes**
    - Fix a bug of bad fvar splicing for loop surface.
    - Fix incorrect bilinear limit tangents in FarStencilTablesFactory
    - Fix boundary interpolation rules doc
    - Added an error check on updating cuda buffer
    - Fix face varying rendering on loop surface
    - Fixed glBatchViewer build for GLFW 2.x
    - Expand search paths for FindGLFW.cmake for Debian and other Linux architectures
    - Fix CMake executable builds for ICC
    - Fix bhr baseline regression, so reference files are real OBJ's
    - Fixed clKernelBundle.cpp to build on Android.
    - Fix misc build warings

Release 2.2.0
=============

**New Features**
    - Added subdivision stencil functionality (Far & OsdEval)

**Bug Fixes**
    - Fix D3D11DrawContext to check for NULL pointers
    - Fix cpuEvalLimitController crash bug
    - Fixed search path suffixes for ICC libs
    - Fixed invalid initialization of glslTransformFeedback kernel.

Release 2.1.0
=============

**New Features**
    - Added TBB Compute back-end on Linux (contribution from Sheng Fu)
    - Added support for ICC compiler (still Beta)

**Changes**
    - Added constructor to OsdMesh with a FarMesh * as input
    - Modify CMake to name and sym-link DSO's based on Linux ABI versioning spec
    - Added command line input to DX11 viewer
    - FarMultiMesh can splice uniform and adaptive meshes together.

**Bug Fixes**
    - Fix FarMultiMesh splicing
    - Removed unnecessary cudaThreadSynchronize calls.
    - Fix glViewer overlapping HUD menus
    - Fix facevarying rendering in glBatchViewer
    - Fix build of GLSL transform feedback kernels
    - Fix 'Getting Started' documentation


Release 2.0.1
=============

**New Features**
    - New CLA files to reflect Apache 2.0 licensing

**Changes**
    - Move all public headers to include/opensubdiv/...
    - Adding Osd documentation based on Siggraph slides

**Bug Fixes**
    - Fix incorrect transition pattern 3 in GLSL / HLSL shaders
    - Fix CMake build to not link GPU-based libraries into libosdCPU
    - Fix support for GLEW on OSX
    - Fix GLFW Xrandr & xf86vmode dependency paths for X11 based systems
    - Fix HUD display overlaps in code examples
    - Fix FindGLEW.cmake to be aware of multiarch on linux systems
    - Fix some hard-coded include paths in CMake build


Release 2.0.0
=============

**New Features**
    - New CMake build flags: NO_LIB, NO_CUDA, NO_PYTHON)

**Changes**
    - OpenSubdiv is now under Apache 2.0 license
    - HbrHalfedge and HbrFVarData copy constructors are now private
    - Documentation style matched to graphics.pixar.com + new content
    - Add an animation freeze button to ptexViewer
    - Variable name changes for better readability across all example
      shader code

**Bug Fixes**

    - Fix incorrect patch generation for patches with 2 non-consecutive boundary edges
    - Fix "undefined gl_PrimitiveID" shader build errors
    - Fix for shader macro "OSD_DISPLACEMENT_CALLBACK"
    - Fix out-of-bounds std::vector access in FarPatchTablesFactory

----

Release 1.2.4
=============

**New Features**

    - Adding support for fractional tessellation of patches
    - Adding a much needed API documention system based on Docutils RST markup
    - Adding support for face-varying interpolation in GLSL APIs
    - Adding varying data buffers to OsdMesh
    - Adding accessors to the vertex buffers in OsdGlMesh
    - Adding face-varying data to regression shapes

**Changes**

    - Cleanup of common bicubic patch shader code (GLSL / HLSL) for portability
      (ATI / OSX drivers)

**Bug Fixes**

    - Fix FarVertexEditTablesFactory to insert properly vertex edit batches
      (fixes incorrect hierarchical hole in regression shape)
    - Fix FarPatchMap quadtree to not drop top-level non-quad faces
    - Fix Gregory patches bug with incorrect max-valence
    - Fix FarPatchTables::GetNumFaces() and FarPatchTables::GetFaceVertices()
      functions to return the correct values
    - Fix face indexing GLSL code (ptex works on non-quads again)
    - Fix face-varying data splicing in FarMultiMeshFactory
    - Fix ptex face indexing in FarMultiMeshFactory
    - Fix glew #include to not break builds
    - Fix Clang / ICC build failures with FarPatchTables
    - Fix build and example code to work with GFLW 3.0+
    - Fix cmake to have ptex dynamically linked in OSX

----

Release 1.2.3
=============

**New Features**

    - Adding Varying and Face-Varying data interpolation to EvalLimit

**Changes**

    - EvalLimit API refactor : the EvalContext now has dedicated structs to track all
      the vertex, varying and face-varying data streams. Also renamed some "buffers"
      into "tables" to maintain code consistency
    - EvalLimit optimization : switch serial indexing to a quad-tree based search

**Bug Fixes**

    - Face-varying data bug fixes : making sure the data is carried around appropriately
      Fixes for OpenCL use with the new batching APIs
    - GLSL general shader code cleanup & fixes for better portability
    - GLSL Tranform Feedback initialization fix
    - Critical fix for FarMultiMesh batching (indexing was incorrect)
    - Fix osdutil CL implementation (protect #includes on systems with no OpenCL SDK
      installed)
    - Fix face-varying interpolation on adaptive patches
    - FarPatchTables : fix IsFeatureAdaptive() to return the correct answer
    - Fix Far factories to handle the absence of face-varying data correctly.
    - Many GLSL shader code style fixes which should help with ATI / OSX shader compiling

----

Release 1.2.2
=============

**New Features**

    - Introducing the EvalLimit API : the Eval module aims at providing support for
      computational tasks that are not related to drawing the surfaces. The EvalLimit
      sub-module provides an API that enables client code to evaluate primitive variables
      on the limit surface.

    .. image:: images/evalLimit_hedit0.jpg
       :height: 300px
       :align: center
       :target: images/evalLimit_hedit0.jpg

    - Osd<xxx>ComputeController : minor optimization. Added early exit to Refine method
      to avoid unnecessary interop.

**Changes**

    - OsdGLDawContext : minor API change. Protecting some member variables and adding
      const accessors
    - OsdError : minor API refactor, added Warning functions.

**Bug Fixes**

    - Fix Ptex bug : prevent corner texel guttering code to from going into infinite
      loops
    - Adding the ability for a FarMeshFactory to construct patchTables starting from
      'firstLevel' in uniform subdivision mode
    - Consolidating the color coding of bicubic patch types through all our our code
      examples (this is used mostly as a debugging tool)
    - Fixing some MSVC++ build warnings
    - Update to the outdated README.md

----

Release 1.2.1
=============

**New Features**

    - Added CUDA runtime error checking

----

Release 1.2.0
=============

**Changes**

    - Major Far refactor around patchTables to introduce the draw batching API
    - Renaming osd_util to osdutil

**Bug Fixes**

    - Fix GLSL transform feedback initialization bug in ptexViewer
    - Minor bug & typo fixes

----

Release 1.1.0
=============

**New Features**

    - release initiated because of the switch to Git Flow

----

Release 1.0.0
=============

Oringal release:

