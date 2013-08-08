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
static topology at interactive framerates. The resulting limit surfaces are a match
for Pixar's Renderman specification within numerical precision limits.

OpenSubdiv is a code API which we hope to integrate into 3rd. party digital
content creation tools. It is **not** an application, nor a tool that can be used
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
Being able to perform the same operation in less than 3ms allows the user to interact
with the smooth, accurate limit surface at all times.

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

OpenSubdiv is covered by the Apache License, and is free to use for commercial or
non-commercial use. This is the same code that Pixar uses internally for animated
film production. Our intent is to encourage a geometry standard for subdivision 
surfaces, by providing consistent (i.e. yielding the same limit surface), high 
performance implementations on a variety of platforms.

Why Apache? We were looking for a commercial-friendly license that would convey 
our patents to the end users. This quickly narrowed the field to Microsoft Public 
License or Apache. Initially we chose MSPL because it handled trademarks better. 
But at the request of several companies we gave Apache another look, and decided 
to go with Apache with a very slight modification that simply says you cannot use 
any contributors' trademarks. In other words, you can use OpenSubdiv to make a 
product, but you cannot use a Luxo Lamp (or other character, etc.) when marketing 
your product.


----

Contributing
============

In order for us to accept code submissions (merge git pull-requests), contributors 
need to sign the Contributor License Agreement (CLA). There are two CLAs, one for 
individuals and one for corporations. As for the end-user license, both are based 
on Apache. They are found in the code repository (`individual form 
<https://github.com/PixarAnimationStudios/OpenSubdiv/blob/master/OpenSubdivCLA_individual.pdf>`__,
`corporate form <https://github.com/PixarAnimationStudios/OpenSubdiv/blob/master/OpenSubdivCLA_corporate.pdf>`__). 
Please email the signed CLA to opensubdiv-cla@pixar.com.


For more details about OpenSubdiv, see `Pixar Graphics Technologies <http:  graphics.pixar.com>`__.

----

External Resources
==================

Microsoft Research:
    `Charles Loop <http://research.microsoft.com/en-us/um/people/cloop/>`__
    `Matthias Niessner <http://lgdv.cs.fau.de/people/card/matthias/niessner/>`__

Pixar Research:
    `Pixar R&D Portal <http://graphics.pixar.com/research/>`__




