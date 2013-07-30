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
  

HBR Overview
------------

.. contents::
   :local:
   :backlinks: none


Hierarchical Boundary Representation (Hbr)
==========================================

Hbr is an interconnected topological data representation. The high level of vertex
connectivity information makes this representation well suited for creation and
editing purposes. It is however inefficient for interactive refinement operations:
Separate objects are allocated for each vertex and edge with pointers to neighboring 
vertices and edges.

Hbr is also the lowest-level subdivision library in Pixar's `Photorealistic RenderMan`.

----

Half-edge Data Structure
========================

The current implementation is based on a half-edge data structure.

.. image:: images/half_edge.png
   :align: center

----

Half-edge cycles and Manifold Topology
======================================

Because half-edges only carry a reference to their opposite half-edge, a given 
edge can only access a single neighboring edge cycle. 

.. image:: images/half_edge_cycle.png
   :align: center
   
This is a fundamental limitation of the half-edge data structure, in that it
cannot represent non-manifold geometry, in particular fan-type topologies. A
different approach to topology will probably be necessary in order to accomodate
non-manifold geometry.

----

Templated Vertex Class
======================

The vertex class has been abstracted into a set of templated function accesses. 
Providing Hbr with a template vertex class that does not implement these functions 
allows client-code to use Hbr as a pure topological analysis tool without having 
to pay any costs for data interpolation. It also allows client-code to remain in 
complete control of the layout of the vertex data : interleaved or non-interleaved.

----

Boundary Interpolation Rules
============================

**Hbr** recognizes 4 rule-sets of boundary interpolation:

+------------------------------------+
| Interpolation Rule-Sets            |
+====================================+
| k_InterpolateBoundaryNone          |
+------------------------------------+
| k_InterpolateBoundaryEdgeOnly      |
+------------------------------------+
| k_InterpolateBoundaryEdgeAndCorner |
+------------------------------------+
| k_InterpolateBoundaryAlwaysSharp   |
+------------------------------------+
