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
       | `glPtexViewer <glptexviewer.html>`_
       | `glEvalLimit <glevallimit.html>`_
     - | `glStencilViewer <glstencilviewer.html>`_
       | `glPaintTest <glpainttest.html>`_
       | `glShareTopology <glsharetopology.html>`_
       | `glFVarViewer <glfvarviewer.html>`_

.. list-table:: **DirectX examples**
   :class: quickref
   :widths: 50 50

   * - | `dxViewer <dxviewer.html>`_
     - | `dxPtexViewer <dxptexviewer.html>`_

.. list-table:: **Plugin examples**
   :class: quickref
   :widths: 50

   * - | `Maya osdPolySmooth <maya_osdpolysmooth.html>`_


----

Common Keyboard Controls
========================

   .. code:: c++

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

