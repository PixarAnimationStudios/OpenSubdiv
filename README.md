# OpenSubdiv #

OpenSubdiv is a set of open source libraries that implement high performance subdivision surface (subdiv) evaluation on massively parallel CPU and GPU architectures. This codepath is optimized for drawing deforming subdivs with static topology at interactive framerates. OpenSubdiv can amplify a 30,000 polygon base mesh into a smooth limit surface of 500,000 polygons in under 3 milliseconds on Kepler Nvidia hardware.  The architecture features a precomputation step that uses Renderman's hbr library to compute fast run time data structures that are evaluated with backends in any of C++, CUDA, OpenCL, or GLSL.  The resulting limit surface matches Pixar's Renderman to numerical precision. OpenSubdiv also includes support for semi-sharp creases and hierarchical edits on subdivs which both are powerful tools for shaping surfaces.  

OpenSubdiv is covered by a modified version of the Microsoft Public License (included below), and is free to use for commercial or non-commercial use. All Pixar patents covering algorithms used inside for semi-sharp crease evaluation and texture coordinate interpolation have also been released for public use. Our intent is to encourage high performance accurate subdiv drawing by giving away the "good stuff" that we use internally.   We welcome any involvement in the development or extension of this code, we'd love it in fact.  Please contact us if you're interested.

This is the fifth generation subdiv library in use by Pixar's animation system in a lineage that started with code written by Tony DeRose and Tien Truong for Geri's Game in 1996.  Each generation has been a from-scratch rewrite that has built upon our experience using subdivision surfaces to make animated films.  OpenSubdiv is exactly the same code used in the Presto animation system for films after Brave. This code is live, developments to OpenSubdiv made by Pixar for current and future films will be released as open source at the same time they are rolled out to Pixar animation production.

The technology here is based on the work by  Niessner, Loop, Meyer, and DeRose in:

  * http://research.microsoft.com/en-us/um/people/cloop/tog2012.pdf


## Quickstart ##

Basic instructions to get started with the code.

### Dependencies ###

Required :
* IlmBase 1.0.1 : http://www.openexr.com/downloads.html

Optional :
* Maya SDK (sample code for Maya viewport 2.0 primitive)

### Build instructions for linux : ###

<pre><code>
* clone the repository :
    git clone git@github.com:PixarAnimationStudios/OpenSubdiv.git

* generate Makefiles :
    cd OpenSubdiv
    mkdir build
    cmake ..

* build the project :
    make
</code></pre>

### Useful cmake options ###

<pre><code>
-DCMAKE_BUILD_TYPE=[Debug|Release]

-DILMBASE_LOCATION=[path to IlmBase]

-DMAYA_LOCATION=[path to Maya]

</code></pre>

## Why fast subdivision? ##

Subdivision surfaces are commonly used for final rendering of character shapes for a smooth and controllable limit surfaces. However, subdivision surfaces in interactive apps are typically drawn as their polygonal control hulls because of performance.  The polygonal control hull is an approximation that is offset from the true limit surface,  Looking at an approximation in the interactive app makes it difficult to see exact contact, like fingers touching a potion bottle or hands touching a cheek.  It also makes it difficult to see poke throughs in cloth simulation if the skin and cloth are both approximations.  This problem is particularly bad when one character is much larger than another and unequal subdiv face sizes cause approximations errors to be magnified.

Maya and Pixar's Presto animation system can take 100ms to subdivide a character of 30,000 polygons to the second level of subdivision (500,000 polygons).  By doing the same thing in 3ms OpenSubdiv allows the user to see the smooth, accurate limit surface at all times. 

## Components ##

#### hbr (hierarchical boundary rep) ####
This base library implements a half edge data structure to store edges, faces, and vertices of a subdivision surface. This code was authored by Julian Fong on the Renderman team. It is the lowest level subdivision libary in renderman. Separate objects are allocated for each vertex and edge (*2) with pointers to neighboring vertices and edges. Hbr is a generic templated API used by clients to create concrete instances by providing the implementation of the vertex class. 

#### far (feature-adaptive rep) ####
Far uses hbr to create and cache fast run time data structures for table driven subdivision of vertices and cubic patches for limit surface evaluation.  Feature-adaptive refinement logic is used to adaptively refine coarse topology near features like extrordinary vertices and creases in order to make the topology amenable to cubic patch evaluation. Far is also a generic templated algorthmic base API that clients in higher levels instantiate and use by providing an implementation of a vertex class. Subdivision schemes supported:
 * Catmull-Clark
 * Loop
 * Bilinear

#### osd (Open Subdiv) ####
Osd contains client level code that uses far to create concrete instances of meshes and compute patch CVs with different backends for table driven subdivision. We support the following backends in osd:

 * C++ with single or multiple threads
 * glsl kernels with transform feedback into vbos
 * OpenCL kernels
 * CUDA kernels.

The amount of hardware specific computation code is small, ~300 lines of code, so it isn't a large effort to support multiple different ones for different clients.

With support for cubic patch fitting in release 2.0 the results of table driven subdivision will pass to tesselation shaders in glsl to dice the patch dynamically.  In release 1.0 the tables are used to generate polygon mesh CVs at the Nth level of subdivison.

## Release 1 ##

The first release of OpenSubdiv is targeting beta for SIGGRAPH 2012.  This release supports uniform table-driven subdivision on the CPU and GPU, but not yet creating cubic patches to dice with tesselation shaders.  In addition to hbr, far, and osd, this release will contain:
 * A reference Maya viewport 2.0 draw override plugin implemented on osd.
 * A simple standalone viewer for performance testing.
 * A set of regression tests that validate correctness.

This release will support:
 * Uniform subdivision.
 * Variable and uniform creases.
 * Computing the complete Nth subdivision level for a subdiv.  Note that these aren't limit points yet, but the result of subdividing N times.
 * Ptex texture drawing
 * Some support for hierarchical edits.

## Release 2 ##

The second release of OpenSubdiv raises the performance bar to what we believe is the maximum level by computing cubic ppatches that are supported directly in hardware with tesselation shaders.   This fitting process is done adaptively, so areas of the mesh that are all quads and regular aren't subdivided at all, and the detail is clustered close to features that require it like creases, extrordinary points, hierarchical edits, boundaries, corners, etc.  The largest improvement here will be in memory over storing uniform subdivisions.

This release will also complete hierarchical edit support and support for face varying coordinate interpolation.

We are targeting release 2 for end of year 2012, hopefully earlier than that.  We have the patch code working in a very rough implementation but need to rewrite that in a development branch for release-ready code. Let us know if you're interested in contributing to that effort! 

## Wish List ##

There are many things we'd love to do to improve support for subdivs but don't have the resources to.  We hope folks feel welcome to contribute if they have the interest and time.  Some things that could be improved:
  * The precomputation step with hbr can be slow.  Does anyone have thoughts on higher performance with topology rich data structures needed for feature adaptive subdivision?  Maybe a class that packs adjacency into blocks of indices efficiently, or supports multithreading, or even feature-adaptive subdivision on the GPU?
  * The reference maya plugin doesn't integrate with Maya shading.  That would be cool.
  * John Lasseter loves looking at film assets in progress on an iPad.  If anyone were to get this working on iOS he'd be looking at your code, and the apple geeks in all of us would smile.
  * Alembic support would be wonderful, but we don't use Alembic enough internally to do the work.




## Open Source License ##

The following license text describes the open source policy adopted by Pixar and is included in every source file.

<pre><code>

//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//


</code></pre>


