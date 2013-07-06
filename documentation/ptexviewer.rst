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
  

ptexViewer
----------

.. contents::
   :local:
   :backlinks: none

SYNOPSIS
========

.. parsed-literal:: 
   :class: codefhead

   **ptexViewer** 
   [**-e** *environment map*]
   [**-d** *HDR diffuse map*]
   [**-s** *HDR specular map*]
   [**-y**]
   [**--disp** *displacement scale*]
   [**--bump** *bump scale*]
   [**-l** *isolation level*] 
   [**-c** *animation loops*] 
   [**-f**] 
   *ptex color file*
   *ptex displacement file*
   *ptex occlusion file*
   *ptex specular file*
   *objfile(s)*

DESCRIPTION
===========

``ptexViewer`` is a stand-alone application that showcases advanced HDR shading
with color, displacement, occlusion and specular ptex maps. Multiple controls 
are available to experiment with the algorithms.

.. image:: images/ptexviewer.png 
   :width: 400px
   :align: center
   :target: images/ptexviewer.png 

OPTIONS
=======

**-e** *environment map*
  A low dynamic range spherical environment map used as a background. Ideally,
  a color-normalized version of the HDR light probe.

**-d** *HDR diffuse map*
  An HDR file containing a diffuse environment map (typically they are low
  resolution blurry hemispherical convolutions of the environment light probe).

**-s** *environment map*
  An HDR file containing a specular environment map.

**--disp** *displacement scale*
  A scalar multiplier for the shader displacement values.

**--bump** *displacement scale*
  A scalar multiplier for the shader bump values.

**-y**
  Swap Z-up geometry to Y-UP.

**-l** *isolation level*
  Select the desired isolation level of adaptive feature isolation. This can be 
  useful when trying to load large pieces of geometry.

**-c** *animation frequency*
  Number of repetitions of the animtion loop (default=0 is infinite)

**-f**
  Launches the application in full-screen mode (if is supported by GLFW on the
  OS)

*ptex color file*
  A ptex file containing RGB channels read as material albedo color.
  
*ptex displacement file*
  A single-channel ptex file (preferrably float precision) containing the 
  displacement values.

*ptex occlusion file*
  A single-channel ptex file (preferrably 8 bits precision) containing a 
  pre-computed ambient occlusion signal.

*ptex specular file*
  A single-channel ptex file (preferrably 8 bits precision) applied to modulate
  the specular reflectance of the material
  
*objfile(s)*
  A sequence of obj files used as an animation loop (the topology has to match
  the data contained in all the ptex files !)


Keyboard Controls
=================

   .. code:: c++
   
      e      : draw normals
      g      : toggle ptex texel guttering
      r      : reload and re-compile the shader files


SEE ALSO
========

`Code Examples <code_examples.html>`__, \
`glViewer <glviewer.html>`__, \
`glBatchViewer <glbatchviewer.html>`__, \
`ptexViewer <ptexviewer.html>`__, \
`paintTest <painttest.html>`__, \
`limitEval <limiteval.html>`__, \
`dxViewer <dxviewer.html>`__, \

