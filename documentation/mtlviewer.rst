..  
     Copyright 2019 Pixar
  
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
  

mtlViewer
---------

.. contents::
   :local:
   :backlinks: none

SYNOPSIS
========

.. parsed-literal::
   :class: codefhead

   **open** **mtlViewer.app** **--args** [**-yup**] [**-u**] [**-a**] [**-l** *refinement level*]
      *objfile(s)* [**-catmark**] [**-loop**] [**-bilinear**]

DESCRIPTION
===========

``mtlViewer`` is a stand-alone application that showcases the application of
uniform and feature adaptive subdivision schemes to a collection of geometric
shapes. Multiple controls are available to experiment with the algorithms.

+----------------------------------------------------+
| .. image:: images/mtlviewer.png                    |
|    :width: 400px                                   |
|    :align: center                                  |
|    :target: images/mtlviewer.png                   |
|                                                    |
| mtlViewer running on macOS                         |
+----------------------------------------------------+

+----------------------------------------------------+
| .. image:: images/mtlviewer_ios.png                |
|    :width: 400px                                   |
|    :align: center                                  |
|    :target: images/mtlviewer_ios.png               |
|                                                    |
| mtlViewer running on iOS                           |
+----------------------------------------------------+

KEYBOARD CONTROLS
=================

   .. code:: c++

      q      : quit
      f      : fit frame
      +/-    : increase / decrease tessellation rate

.. include:: examples_see_also.rst
