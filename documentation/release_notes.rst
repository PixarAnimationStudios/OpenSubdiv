..  
       Copyright 2013 Pixar

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License
       and the following modification to it: Section 6 Trademarks.
       deleted and replaced with:

       6. Trademarks. This License does not grant permission to use the
       trade names, trademarks, service marks, or product names of the
       Licensor and its affiliates, except as required for reproducing
       the content of the NOTICE file.

       You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing,
       software distributed under the License is distributed on an
       "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
       either express or implied.  See the License for the specific
       language governing permissions and limitations under the
       License.
  

Release Notes
-------------

.. contents::
   :local:
   :backlinks: none

----

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

