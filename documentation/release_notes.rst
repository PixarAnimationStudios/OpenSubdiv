..  
       Copyright (C) Pixar. All rights reserved.
  
       This license governs use of the accompanying software. If you
       use the software, you accept this license. If you do not accept
       the license, do not use the software.
  
       1. Definitions
       The terms "reproduce," "reproduction," "derivative works," and
       "distribution" have the same meaning here as under U.S.
       copyright law.  A "contribution" is the original software, or
       any additions or changes to the software.
       A "contributor" is any person or entity that distributes its
       contribution under this license.
       "Licensed patents" are a contributor's patent claims that read
       directly on its contribution.
  
       2. Grant of Rights
       (A) Copyright Grant- Subject to the terms of this license,
       including the license conditions and limitations in section 3,
       each contributor grants you a non-exclusive, worldwide,
       royalty-free copyright license to reproduce its contribution,
       prepare derivative works of its contribution, and distribute
       its contribution or any derivative works that you create.
       (B) Patent Grant- Subject to the terms of this license,
       including the license conditions and limitations in section 3,
       each contributor grants you a non-exclusive, worldwide,
       royalty-free license under its licensed patents to make, have
       made, use, sell, offer for sale, import, and/or otherwise
       dispose of its contribution in the software or derivative works
       of the contribution in the software.
  
       3. Conditions and Limitations
       (A) No Trademark License- This license does not grant you
       rights to use any contributor's name, logo, or trademarks.
       (B) If you bring a patent claim against any contributor over
       patents that you claim are infringed by the software, your
       patent license from such contributor to the software ends
       automatically.
       (C) If you distribute any portion of the software, you must
       retain all copyright, patent, trademark, and attribution
       notices that are present in the software.
       (D) If you distribute any portion of the software in source
       code form, you may do so only under this license by including a
       complete copy of this license with your distribution. If you
       distribute any portion of the software in compiled or object
       code form, you may only do so under a license that complies
       with this license.
       (E) The software is licensed "as-is." You bear the risk of
       using it. The contributors give no express warranties,
       guarantees or conditions. You may have additional consumer
       rights under your local laws which this license cannot change.
       To the extent permitted under your local laws, the contributors
       exclude the implied warranties of merchantability, fitness for
       a particular purpose and non-infringement.
  

Release Notes
-------------

.. contents::
   :local:
   :backlinks: none


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

