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


Code Examples
-------------

Standalone Viewers
==================

OpenSubdiv builds a number of standalone viewers that demonstrate various aspects
of the software.

----

.. list-table:: **OpenGL examples**
   :class: quickref
   :widths: 50 50

   * - | `glViewer <glviewer.html>`_
       | `glFVarViewer <glfvarviewer.html>`_
       | `glEvalLimit <glevallimit.html>`_
       | `glStencilViewer <glstencilviewer.html>`_
     - | `glPtexViewer <glptexviewer.html>`_
       | `glPaintTest <glpainttest.html>`_
       | `glShareTopology <glsharetopology.html>`_

.. list-table:: **DirectX examples**
   :class: quickref

   * - | `dxViewer <dxviewer.html>`_
     - | `dxPtexViewer <dxptexviewer.html>`_

.. list-table:: **Metal examples**
   :class: quickref
   :widths: 50 50

   * - | `mtlViewer <mtlviewer.html>`_
     - | `mtlPtexViewer <mtlptexviewer.html>`_

----

Common Command Line Options
===========================

While the following command line options are common, not all examples support
all -- in some cases, certain options are not applicable.

::

      -f         : launches the application in full-screen mode (if supported)
      -yup       : set initial view with Y-axis up (Z-up is the default)
      -u         : apply uniform subdivision to all meshes
      -a         : apply adaptive subdivision to create patches for the limit surface
      -l depth   : level of uniform or adaptive refinement to apply
      -c count   : number of repetitions of the animation loop when supported (default
                   of 0 is infinite)
      -anim      : interpret a given set of Obj files as an animated sequence rather
                   than a set of distinct meshes
      -catmark   : apply the Catmark scheme to all meshes in given Obj files (default)
      -loop      : apply the Loop scheme to all meshes in given Obj files
      -bilinear  : apply the Bilinear scheme to all meshes in given Obj files
      objfile(s) : a set of one or more meshes in Obj format (requiring a .obj extension)
                   that may be distinct meshes or animated versions of a single mesh

----

Common Keyboard Controls
========================

::

      Left mouse button drag   : orbit camera
      Middle mouse button drag : pan camera
      Right mouse button       : dolly camera
      n, p                     : next/prev model
      1, 2, 3, ..., 9, 0       : specify adaptive isolation or uniform refinement level
      +, -                     : increase / decrease tessellation
      Tab                      : toggle full-screen
      Esc                      : turn on / off the HUD
      w                        : switch display mode
      q                        : quit

