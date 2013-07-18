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
       | `glBatchViewer <glbatchviewer.html>`_
       | `ptexViewer <ptexviewer.html>`_
     - | `paintTest <painttest.html>`_
       | `limitEval <limiteval.html>`_

.. list-table:: **DirectX examples**
   :class: quickref
   :widths: 50 50
   
   * - | `dxViewer <dxviewer.html>`_
     - |

.. list-table:: **Plugin examples**
   :class: quickref
   :widths: 50 50

   * - | `mayaViewer <mayaviewer.html>`_
     - | `mayaPtexViewer <mayaptexviewer.html>`_

|

.. container:: notebox

   **Note:**
   the Maya plugins are currently unsupported and they may fail to compile
   or work with current versions of OpenSubdiv. These were originally written for 
   the sole purpose of live demonstrations and the code is provided only as an 
   implementation example.

|

----

Common Keyboard Controls
========================

   .. code:: c++
   
      Left mouse button drag   : orbit camera
      Middle mouse button drag : pan camera
      Right mouse button       : dolly camera
      n, p                     : next/prev model
      1, 2, 3, ..., 9, 0       : specify adaptive isolation or uniform refinment level
      +, -                     : increase / decrease tessellation 
      Tab                      : toggle full-screen
      Esc                      : turn on / off the HUD
      w                        : switch display mode
      q                        : quit

