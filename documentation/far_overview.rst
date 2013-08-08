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
  

FAR Overview
------------

.. contents::
   :local:
   :backlinks: none

Feature Adaptive Representation (Far)
=====================================

Far is a serialized topoloigcal data representation.Far uses hbr to create and 
cache fast run time data structures for table driven subdivision of vertices and 
cubic patches for limit surface evaluation. `Feature-adaptive <subdivision_surfaces.html#feature-adaptive-subdivision>`__ 
refinement logic is used to adaptively refine coarse topology near features like 
extraordinary vertices and creases in order to make the topology amenable to 
cubic patch evaluation. Far is also a generic, templated algorithmic base API 
that clients in higher levels instantiate and use by providing an implementation 
of a vertex class. It supports these subdivision schemes:

Factories & Tables
==================

Subdivision Tables
==================

Patch Tables
============
