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

