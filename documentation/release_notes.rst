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

Release 3.0.0
=============

**New Features**
    - Faster subdivision using less memory
    - Introduction of Stencil Tables
    - Faster, simple GPU kernels
    - Unified adaptive shaders
    - Single Crease Patch
    - New Irregular Patch approximations
    - Support for non-manifold topology
    - Simpler conversion of topology
    - Face-Varying topology
    - Elmination of fixed valence tables
    - New coding style
    - More documentation and tutorials
    - For more, information please see `Introduction to 3.0 <intro_30.html>`__


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
