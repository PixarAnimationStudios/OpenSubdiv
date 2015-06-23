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

3.0 Master Release (Q2 2015)

    * freeze subdivision 'specification' (enforce backward compatibility)

3.1 Supplemental Release (Q3/Q4 2015)

    * include any noteworthy omissions arising from 3.0
    * add support for bi-cubic face-varying limit patches
    * add support for higher order differentiation of limit patches

3.2 Future Release (2016)

    * TBD


Near Term
=========

The following is a short list of topics already expressed as priorities and
receiving ongoing attention -- some being actively prototyped to varying
degrees.

Feature parity for Loop subdivision
+++++++++++++++++++++++++++++++++++

  The more popular Catmark scheme has long received more attention and effort
  than the Loop scheme.  Given OpenSubdiv claims to support both, additional
  effort is required to bring Loop to the same level of functionality as
  Catmark.  With the feature-adaptive analysis now scheme-independent, the
  addition of triangular patches to the PatchTables will go a long way towards
  that goal.  Prototype patch gathering and evaluation methods have already
  been tested within the existing code base and discussions on extending the
  internal patch infra-structure are underway.

Improved support for infinitely sharp features
++++++++++++++++++++++++++++++++++++++++++++++

  The current implementation of adaptive feature isolation requires infinitely
  sharp creases to be pushed to the highest level of isolation -- eventually
  representing the result with a regular patch. The surface is therefore both
  inefficient and incorrect. Patches with a single infinitely sharp edge can be
  represented exactly with regular boundary patches and could be isolated at a
  much higher level.  Continuity with dart patches is necessary in such cases,
  and approximating more sharp irregular regions with alternate patch types
  (e.g. Gregory or Bezier) will help this goal and others.

Dynamic feature adaptive isolation (DFAS)
+++++++++++++++++++++++++++++++++++++++++

  Adaptive feature isolation can produce a large number of patches, especially
  when the model contains a lot of semi-sharp creases. We need a LOD solution
  that can dynamically isolate features based on distance to view-point.  (Note:
  paper from Matthias Niessner & Henry Schafer)

Longer Term
===========

The following is a list of pending projects and future directions for
OpenSubdiv.

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

"Next-gen" back-ends
++++++++++++++++++++

  Implement Osd Evaluator and Patch Drawing for next-gen GPU APIs such as
  Mantle, Metal, DX12, Vulkan.

A means to control edge curvature
+++++++++++++++++++++++++++++++++

  Edge sharpness provides insufficient control over the overall contour of the
  surfaces. Artists often duplicate edge-loops around semi-sharp edges in
  order to control the local surface curvature. Ideally, they would like to be
  able to specify a radius of curvature that produces circular rounded edges.
  This will likely require the introduction of non-uniform rational splines
  (NURCCS ?) in OpenSubdiv.


Always in Need of Improvement
=============================

And finally, a few topics that always benefit from continual improvement.
Any and all contributions in this area are greatly appreciated.

Regression testing
++++++++++++++++++

  OpenSubdiv currently ships with some regression testing code which can be
  run using CTest.  It's always great to have more regression testing that
  covers more of the code base.

    * Implement a robust regression harness for numerical correctness
    * Implement a cross-platform regression harness for GPU drawing correctness
    * Implement a cross-platform regression harness for performance (speed & memory)
    * Implement code coverage analysis

Documentation
+++++++++++++

  In order to facilitate adoption of OpenSubdiv, we need to provide clear,
  concise and comprehensive documentation of all APIs. In particular:

    * Update and flesh out high-level ReST documentation
    * Clean up the Doxygen documentation
    * Expand code tutorials

