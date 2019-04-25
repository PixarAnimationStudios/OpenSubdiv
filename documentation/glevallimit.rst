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


glEvalLimit
-----------

.. contents::
   :local:
   :backlinks: none

SYNOPSIS
========

.. parsed-literal::
   :class: codefhead

   **glEvalLimit** [**-f**] [**-yup**] [**-u**] [**-a**] [**-l** *refinement level*]
       *objfile(s)* [**-catmark**] [**-loop**] [**-bilinear**]

DESCRIPTION
===========

``glEvalLimit`` is a stand-alone application that showcases the limit surface
Eval module. On the given shape, random samples are generated in local s,t space.
Vertex, varying and face-varying data is then computed on the surface limit and
displayed as colors.

In order to emphasize the dynamic nature of the EvalLimit API, where the
locations can be arbitrarily updated before each evaluation, the glEvalLimit
example treats each sample as a 'ST particle'.

ST Particles are a simplified parametric-space particle dynamics simulation: each
particle is assigned a location on the subdivision surface limit that is
composed of a unique ptex face index, with a local (s,t) parametric pair.

The system also generates an array of parametric velocities (ds, dt) for each
particle. An Update() function then applies the velocities to the locations and
moves the points along the parametric space.

Face boundaries are managed using a ptex adjacency table obtained from the
Far::TopologyRefiner. Every time a particle moves outside of the [0.0f, 1.0f]
parametric range, a 'warp' function moves it to the neighboring face, or
bounces it, if the edge happens to be a boundary.

Note: currently the adjacency code does not handle 'diagonal' crossings, nor
crossings between quad and non-quad faces.

Multiple controls are available to experiment with the algorithms.

.. image:: images/glevallimit.jpg
   :width: 400px
   :align: center
   :target: images/glevallimit.jpg

OPTIONS
=======

See the description of the
`common comand line options <code_examples.html#common-command-line-options>`__
for the subset of common options supported here.

.. include:: examples_see_also.rst
