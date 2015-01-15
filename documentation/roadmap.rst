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

Roadmap
-------

.. contents::
   :local:
   :backlinks: none

----

3.x Release Cycle
=================

For more details, please see the `Release Notes <release_notes.html>`_.

3.0 Alpha Releases (Q4 2014):

    * nearing feature complete : code is ready for early evaluation

3.0 Beta Releases (Q1 2015):

    * subdivision 'specification' evaluation from users & community

3.0 Master Release (Q2 2015)

    * freeze subdivision 'specification' (enforce backward compatibility)
    * add support for bi-cubic face-varying interpolation (discrete & limit)
    * add support for Loop limit evaluation & draw


To Infinity & Beyond
====================

The following is a list of pending projects and future directions for
OpenSubdiv.

Optimize Draw
+++++++++++++
  OSD specializes topological patch configurations by configuring GPU shader
  source code. This causes back-end APIs to have to bind many shaders and
  burdens the drivers with many "draw" calls for each primitive, which does
  not scale well. Our goal is to ultimately try to reduce this burden back to
  a single shader bind operation per primitive.

    - Reduce GPU shader variants:

        + Merge regular / boundary / corner cases with vertex mirroring
        + Merge transition cases with degenerate patches
        + Merge rotations cases with run-time conditionals

  Note: this project has been started at Pixar.

Dynamic feature adaptive isolation (DFAS)
+++++++++++++++++++++++++++++++++++++++++

  Adaptive feature isolation can produce a large number of patches, especially
  when the model contains a lot of semi-sharp creases. We need a LOD solution
  that can dynamically isolate features based on distance to view-point.

  Note: paper from Matthias Niessner & Henry Schafer

Implement a "high-level" API layer
++++++++++++++++++++++++++++++++++

  One of the original goals of the OpenSubdiv project is to provide a robust
  and simple API to display subdivision surfaces interactively. Now that the
  algorithms and code-base have matured, we would like to foster a consistent
  implementation of subdivision surfaces suitable for adoption in the lower
  layers of the graphics stack, such as GPU drivers. We have been working on
  the draft of a "specification document" detailing the workings of a high-
  level interface for subdivision surface data and functionality. We need an
  implementation of this high-level API.

  Note: this document drafting has been started at Pixar with partners.

Support for infinitely sharp creases
++++++++++++++++++++++++++++++++++++

  The current implementation of adaptive feature isolation requires infinitely
  sharp creases to be pushed to the highest level of isolation. The resulting
  surface is both incorrect and inefficient. We want to correctly support
  infinitely sharp creases with discontinuous patches that do not require to
  be isolated to the highest level of subdivision.

A means to control edge curvature
+++++++++++++++++++++++++++++++++

  Edge sharpness provides insufficient control over the overall contour of the
  surfaces. Artists often duplicate edge-loops around semi-sharp edges in
  order to control the local surface curvature. Ideally, they would like to be
  able to specify a radius of curvature that produces circular rounded edges.
  This will likely require the introduction of non-uniform rational splines
  (NURCCS ?) in OpenSubdiv.

"Next-gen" back-ends
++++++++++++++++++++

  Implement Osd::Draw Context & Controllers for next-gen GPU APIs such as
  Mantle, Metal, DX12, GL Next.

Regression testing
++++++++++++++++++

  OpenSubdiv currently ships with some ad-hoc regression code that unfortunately
  does not cover much of the code base: we need to implement a more rigorous QA
  process. We will probably want to leverage the CMake built-in functionalities
  of CTest in order to publish a build & test dashboard.

    * Implement a robust regression harness for numerical correctness
    * Implement a cross-platform regression harness for GPU drawing correctness
    * Implement a cross-platform regression harness for performance (speed & memory)

Documentation
+++++++++++++

  In order to facilitate adoption of OpenSubdiv, we need to provide clear,
  concise and comprehensive documentation of all APIs. In particular:

    * Update and flesh out high-level ReST documentation
    * Clean up the Doxygen documentation
    * Expand code tutorials

