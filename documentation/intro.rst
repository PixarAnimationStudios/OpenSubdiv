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
  

Introduction
------------

.. contents::
   :local:
   :backlinks: none

.. image:: images/geri.jpg 
   :width: 600px
   :align: center

----

Introduction
============

OpenSubdiv is a set of open source libraries that implement high performance 
subdivision surface (subdiv) evaluation on massively parallel CPU and GPU 
architectures. This codepath is optimized for drawing deforming surfaces with 
static topology at interactive framerates. The resulting limit surface are a match
for Pixar's Renderman specification within numerical precision limits.

OpenSubdiv is a code API which we hope to integrate into 3rd. party digital
content creation tools. It is **not** an application nor a tool that can be used
directly to create digital assets.

----

Why Fast Subdivision ?
======================

Subdivision surfaces are commonly used for final rendering of character shapes 
for a smooth and controllable limit surfaces. However, subdivision surfaces in 
interactive apps are typically drawn as their polygonal control hulls because of 
performance. The polygonal control hull is an approximation that is offset from 
the true limit surface. Looking at an approximation in the interactive app makes 
it difficult to see exact contact, like fingers touching a potion bottle or hands 
touching a cheek. It also makes it difficult to see poke-throughs in cloth simulation 
if the skin and cloth are both approximations. This problem is particularly bad when 
one character is much larger than another and unequal subdiv face sizes cause 
approximation errors to be magnified.

Maya and Pixar's proprietary Presto animation system can take 100ms to subdivide 
a character of 30,000 polygons to the second level of subdivision (500,000 polygons). 
By doing the same thing in 3ms, OpenSubdiv allows the user to see the smooth, 
accurate limit surface at all times.

.. image:: images/efficient_subdivision.png 
   :height: 400px
   :align: center
   :target: images/efficient_subdivision.png 



----

Research
========

The new GPU technology behind OpenSubdiv is the result of a joint research effort
between Pixar and Microsoft.

    | *Feature Adaptive GPU Rendering of Catmull-Clark Subdivision Surfaces*
    | Matthias Niessner, Charles Loop, Mark Meyer, and Tony DeRose
    | ACM Transactions on Graphics, Vol. 31 No. 1 Article 6 January 2012 
    | `<http://research.microsoft.com/en-us/um/people/cloop/tog2012.pdf>`_
    |
    | *Efficient Evaluation of Semi-Smooth Creases in Catmull-Clark Subdivision Surfaces*
    | Matthias Niessner, Charles Loop, and Guenter Greiner.
    | Eurographics Proceedings, Cagliari, 2012
    | `<http://research.microsoft.com/en-us/um/people/cloop/EG2012.pdf>`_
    |
    | *Analytic Displacement Mapping using Hardware Tessellation*
    | Matthias Niessner, Charles Loop
    | ACM Transactions on Graphics, To appear 2013
    | `<http://research.microsoft.com/en-us/um/people/cloop/TOG2013.pdf>`_
    
----

Heritage
========

This is the fifth-generation subdiv library in use by Pixar's proprietary animation 
system in a lineage that started with code written by Tony DeRose and Tien Truong 
for Geri\u2019s Game in 1996. Each generation has been a from-scratch rewrite that 
has built upon our experience using subdivision surfaces to make animated films. 
This code is live, so Pixar's changes to OpenSubdiv for current and future films 
will be released as open source at the same time they are rolled out to Pixar 
animation production.

    | *Subdivision for Modeling and Animation*
    | Denis Zorin, Peter Schroder
    | Course Notes of SIGGRAPH 1999
    | `<http://www.multires.caltech.edu/pubs/sig99notes.pdf>`_
    |
    | *Subdivision Surfaces in Character Animation*
    | Tony DeRose, Michael Kass, Tien Truong
    | Proceedings of SIGGRAPH 1998
    | `<http://graphics.pixar.com/library/Geri/paper.pdf>`_
    |
    | *Recursively generated B-spline surfaces on arbitrary topological meshes*
    | Catmull, E.; Clark, J. Computer-Aided Design 10 (6) (1978)

----

Licensing
=========

OpenSubdiv is covered by the `Microsoft Public License 
<http:  www.microsoft.com/en-us/openness/licenses.aspx#MPL>`__, and is free to use 
for commercial or non-commercial use. This is the same code that Pixar uses 
internally for animated film production. Our intent is to encourage high 
performance accurate subdiv drawing by giving away the "good stuff".

Feel free to use it and let us know what you think.

----

Contributing
============

In order for us to accept code submissions (merge git pull-requests), contributors 
need to sign the "Contributor License Agreement" (found in the code repository or 
`here <https://github.com/PixarAnimationStudios/OpenSubdiv/blob/master/OpenSubdivCLA.pdf>`__)
and you can either email or fax it to Pixar.

For more details about OpenSubdiv, see `Pixar Graphics Technologies <http:  graphics.pixar.com>`__.

----

External Resources
==================

Microsoft Research:
    `Charles Loop <http://research.microsoft.com/en-us/um/people/cloop/>`__
    `Matthias Niessner <http://lgdv.cs.fau.de/people/card/matthias/niessner/>`__

Pixar Research:
    `Pixar R&D Portal <http://graphics.pixar.com/research/>`__




