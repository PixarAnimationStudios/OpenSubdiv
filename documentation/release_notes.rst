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

Release 2.6.0
=============

**New Features**
    - Add subdivision kernels for ARM NEON
    - Add OsdUtilVertexSplit which creates a vertex-varying data table by duplicating
      vertices in a `FarMesh`
    - Add basic functions to work with FV data via evaluator API

**Changes**
    - Added Catmark restricted vertex compute kernels that optimize for vertices
      with no semi-sharp creases
    - Fix accessor omissions in osd/mesh.h
    - Add support for different subdivision schemes for OsdUtilMesh

**Bug Fixes**
    - Fix crashes when using rather low-end cards like Intel ones
    - Fix a bug in the creation of an edge-vertex kernel batch
    - Fix mismatch in declaration and usage of OsdCudaComputeRestrictedVertexA 
    - Fix a bug in the vertex order for restricted Catmark vertex-vertex kernel batches
    - Fix a bug in FarCatmarkSubdivisionTablesFactory that prevented the
      CATMARK_QUAD_FACE_VERTEX kernel from being selected for subdivision
      level 2 or greater.
    - Fix a bug in OsdUtilVertexSplit that occurs when getting the address of
      the end of a std::vector
    - Fix error in createCLBuffer that occurs when the buffer size is zero
    - Fix a bug in the CUDA computeRestrictedEdge kernel
    - Fix duplicate variables with identical name
    - Fix osdutil build errors
    - Fix cmake diagnostic messsage

Release 2.5.1
=============

**New Features**
    - Add CATMARK_QUAD_FACE_VERTEX and CATMARK_TRI_QUAD_FACE_VERTEX compute kernels
      optimization that takes advantage of all-quads or all-triange-and-quads meshes

**Bug Fixes**
    - Fix a compiler error in the GLSL Transform Feedback kernels on OS X
    - Fix boundary interpolation in osdutil
    - Fix bilinear stencil tangent computions

Release 2.5.0
=============

**New Features**
    - Add ability to generate triangle patches for a uniformly subdivided mesh
    - Add new example 'topologySharing'
    - Add interleaved buffer mode in glViewer
    - Add GLSL compute kernel to glBatchViewer
    - Add TBB compute kernel to glBatchViewer
    - Add a PullDown widget to our HUD in examples/common
    - GUI updates & cosmetic changes to GL example code
    - Adding a programmable image shader to gl_hud
    - Code cleanup for GLFrameBuffer in examples/common
    - Implement C-API accessor to evaluator topology (osdutil)
    - Add command line option to CMake's options
    - Add a CMake option to disable OpenCL
    - Add a FindCLEW.cmake module in anticipation of using CLEW as a dependency
    - Integrate CLEW into osd library and examples

**Changes**
    - Change interleaved buffer support in OsdCompute: 
    - Removed OsdVertexDescriptor and replaced with OsdVertexBufferDescriptor
    - Reorganize ComputeContext and ComputeController.
    - Reorganize EvalStencilContext and EvalStencilController 
      Moved transient states (current vertex buffer etc) to controller
    - Reorganize EvalLimitContext and EvalLimitController
      Moved transient states (current vertex buffer etc) to controller
    - Fix adaptive isolation of sharp corner vertices
    - Fix incorrect FarMeshFactory logic for isolating multiple corner vertices in corner patches
    - Change EvalLimit Gregory patch kernels to the large weights table to accomodate higher valences
    - Fix calculation of screen space LOD tess factors for transition corner patches.
    - Add a public constructor to OsdMesh
    - Decrease compiler warning thresholds and fix outstanding warnings
    - Make PTex support optional
    - Add a NO_MAYA flag to CMakeLists to disable all Autodesk Maya dependencies in the build
    - Document NO_MAYA command line option

**Bug Fixes**
    - Fix mistakenly deleted memory barrier in glsl OsdCompute kernel.
    - Fix shape_utils genRIB function to use streams correctly.
    - Temporary workaround for the synchronization bug of glsl compute kernel
    - Fix Hud display for higher DPI (MBP retina)
    - Fix Hud (d3d11)
    - Fix examples to use GL timer query to measure the GPU draw timing more precisely
    - Fix glViewer: stop updating during freeze.
    - Fix file permissions on farPatchTablesFactory.h
    - Fix some meory leaks in adaptive evaluator (osdutil)
    - Fix OsdUtilAdaptiveEvaluator concurrency issue
    - Fix OsdUtilRefiner incorrect "Invalid size of patch array" error reporting.
    - Fix OsdUtilPatchPartitioner failure for triangle patches
    - Fixes a bug that causes OsdUtilPatchPartitioner to fail to rebuild the face-varying
      data table correctly for triangle patches.
    - Add missing third parameter to templated OsdDrawContext usage (osdutil/batch.h)
    - Return success status from openSubdiv_finishEvaluatorDescr() (osdutil)
    - Remove debugging std::cout calls (osdutil)
    - Build errors & warnings:
    - Fix OSX Core Profile build (GLFrameBuffer)
    - Fix ptexViewer build error on OSX
    - Fix framebuffer shader compiling for OSX
    - Reordering includes to address a compile error on OSX/glew environment
    - Fix compilation errors with CLEW enabled
    - Fix icc build problems
    - Fix compiler warnings in OsdClVertexBuffer
    - Fix compilation error on windows+msvc2013 
    - Fix build warnings/errors with VS2010 Pro
    - Fix Windows build warning in FarPatchTablesFactory
    - Fix doxygen generation errors


Release 2.4.1
=============

**Changes**
    - Add correct OpenSubdiv namespace begin/end blocks.

**Bug Fixes**
    - Compile osdutil with -fPIC for correct linking.
    - Fix a bug of OsdUtilMeshBatch, the varying buffer isn't computed with CL kernels
    - Fix FindGLFW.cmake to use the %GLFW_LOCATION% environment variable in Windows
    - Fix Draw contexts do not fully initialize patch arrays

Release 2.4.0
=============

**New Features**
    - Adding functionality to store uniform face-varying data across multiple levels of subdivision
    - Add OsdUtilPatchPartitioner.
      It splits patcharray into subsets so that clients can draw partial surfaces
      for both adaptive and uniform.

**Changes**
    - Remove FarMesh dependency from Osd*Context.
    - Use DSA APIs for GL buffer update (if available).
    - Refactor Far API
    - replace void- of all kernel applications with CONTEXT template parameter.
      It eliminates many static_casts from void- for both far and osd classes.
    - move the big switch-cases of far default kernel launches out of Refine so
      that osd controllers can arbitrary mix default kernels and custom kernels.
    - change FarKernelBatch::kernelType from enum to int, clients can add
      custom kernel types.
    - remove a back-pointer to farmesh from subdivision table.
    - untemplate all subdivision table classes and template their compute methods
      instead. Those methods take a typed vertex storage.
    - remove an unused argument FarMesh from the constructor of subdivision
      table factories.
    - Refactor FarSubdivisionTables.
      Delete scheme specialized subdivision tables. The base class FarSubdivisionTables
      already has all tables, so we just need scheme enum to identify which scheme
      the subdivision tables belong to. This brings a lot of code cleanups around far
      factory classes.
    - Move FarMultiMeshFactory to OsdUtil.
    - Move table splicing functions of FarMultiMeshFactory into factories
    - Change PxOsdUtil prefix to final OsdUtil prefix.
    - Improve error reporting in osdutil refinement classes, and fix a build issue

**Bug Fixes**
    - Fix another multi mesh splicing bug of face varying data.
    - Make CMake path variables more robust
    - Fixing a crash on Marvericks w/glew
    - Update dxViewer example documentation
    - Fix wrong logic in openSubdiv_setEvaluatorCoarsePositions
    - Remove debug print from adaptive evaluator's initialization

Release 2.3.5
=============

**New Features**
    - Add the ability to read obj files to the dxViewer example
    - Add screen-capture function to ptexViewer
    - Update documention for Xcode builds
    - Add documentation (boundary interpolation rules and face-varying boundary interpolation rules)

**Changes**
    - Refactoring FarPatchTables and FarPatchTablesFactory
    - Move GL vertex buffer VBO buffer allocation out of allocate() and into BindVBO()
    - Enable uvViewer on OS X now that Mavericks is released.
    - Replacing un-necessary dynamic_cast with reinterpret_cast within FarDispatcher
    - Minor code cleanup of FarMeshFactory
    - Remove address space qualifiers from OpenCL kernel functions
    - Fix OpenCL initialization to be slightly more robust
    - Add OpenCL header include paths where necessary
    - Add 'static' specifiers for non-kernel CL funcs at program scope
    - Add stddef.h to python/osd/osdshim.i
    - Modify ptexViewer and uvViewer shaders to address some portability issues

**Bug Fixes**
    - Fix Gregory Boundary patch buffer overrun
    - Fix black texels when the resolution of a ptex face is less than 4
    - Fix a splicing bug in FarMultiMeshFactory
    - Fix a build error when using older versions of GLFW
    - Fix build warnings (optimized)
    - Fix FindTBB.cmake
    - Fix FindMaya.cmake
    - Fix glViewer support for GLSL compute
    - Fix ptexViewer: enable specular pass in both IBL and point lighting
    - Fix Zlib include in ptexViewer
    - Fix ptexViewer shader errors.
    - Fix osdPolySmooth Maya plugin
    - Fix UV merging in osdPolySmooth code example
    - Add cleanup function to osdPolySmooth Maya plugin
    - Fix Maya OsdPolySmooth node component output
    - Fix GLSL array instantiation syntax for glStencilViewer
    - Fix examples to run correctly on high DPI displays with GLFW 3

Release 2.3.4
=============

**New Features**
    - Adding CPU/OMP/TBB Context / Controller pairs for CPU evaluation of smooth normals
    - Added adaptiveEvaluator class inspired by Sergey's work in blender (OsdUtil)

**Changes**
    - Changed the HUD to ignore mouse clicks when not visible.
    - Updates for blender development (OsdUtil)
    - Add C compatible API to access the adaptiveEvaluator class from non-C++ (OsdUtil)
    - Update license headers to apache (OsdUtil)
    - CMake build improvement : make osd a cmake object library & remove compiling redundancies
    - Improve stringification of shaders & kernels in CMake build

**Bug Fixes**
    - Fixed iOS build
    - Fixed VS2010 warnings/errors.
    - Fix OsdCpuEvalLimitKernel
    - Fix maxvalence calculation in FarMeshFactory
    - Fix FarStencilFactory control stencil caching
    - Removing assert for high-valence vertices running off limit tangent pre-computed table.
    - Fix degenerate stencil limit tangent code path.
    - Fix unused variable build warnings (gcc 4.8.2 - Fedora 19)
    - Fix build warning from osdutil/adaptiveEvaluator.cpp

Release 2.3.3
=============

**Changes**
    - Modify Far remapping of singular vertices to point to their source vertex.
    - Refactoring Ptex Mipmap and Analytic Displacement code
    - Adding some documentation for Chaikin crease rule
    - Misc. improvements to PxOsdUtilsMesh
    - Adding recommended isolation output to OsdPolySmooth node

**Bug Fixes**
    - Adding an error check on version parsing of main CMakeLists
    - Fix regex in FindMaya.cmake that breaks with recent versions of Maya
    - Fix crashes induced by typeid
    - Fixed VS2010 build warning
    - Fix build break in hbr_regression
    - Fix incorrect capitalization in GL ptexViewer shader.glsl
    - Fix OSX build: add stdlib.h include

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

