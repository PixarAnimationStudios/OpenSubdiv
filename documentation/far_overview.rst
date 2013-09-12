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


Stencil Tables
==============


Stencils are the most direct method of evaluation of specific locations on the
limit of a subdivision surface starting from the coarse vertices of the control
cage.

.. image:: images/far_stencil0.png
   :align: center

Sample Location
***************

Each stencil is associated with a singular parametric location on the coarse
mesh. The paramatric location is defined as face location and local [0.0 - 1.0]
(u,v) triplet:

In the case of a non-coarse quad face, the parametric sub-face quadrant needs to
be identified. This can be done either explicitly or implicitly by using the
unique ptex face indices for instance.

.. image:: images/far_stencil6.png
   :align: center


Principles
**********

Iterative subdivision algorithms such as the one used in `FarSubdivisionTables <#subdivision-tables>`__
converge towards the limit surface by sucessively refining the vertices of the
coarse control cage.

.. image:: images/far_stencil4.png
   :align: center

Each step is dependent upon the previous subidivion step being completed, and a
substantial number of steps may be required in order approximate the limit. Since
each subdivision step incurs an O(4 :superscript:`n`) growing amount of
computations, the accrued number of interpolations can be quite large.

However, every intermediate subdivided vertex can be expressed as a linear
interpolation of vertice from the previous step. So, eventually, every point at
on the limit surface can be expressed as a weighted average of the set of coarse 
control vertices from the one-ring surrounding the face that the point is in:

.. image:: images/far_stencil3.png
   :align: center
   
Where:

.. image:: images/far_stencil2.png
   :align: center

Stencils are created by combining the list of control vertices of the 1-ring
to a set of interpolation weights obtained by successive accumulation of
subdivision interpolation weights.

The weight accumulation process is made efficient by adaptively subdividing the
control cage only around extraordinary locations, and otherwise reverting to fast
bi-cubic bspline patch evaluation. The use of bi-cubic patches also allows the
accumulation of analytical derivatives.

API Architecture
****************

The base container for stencil data is the FarStencilTables class. As with most
other Far entities, it has an associated FarStencilTablesFactory that requires
an HbrMesh:

.. image:: images/far_stencil5.png
   :align: center

Assuming a properly qualified HbrMesh:

.. code:: c++

    HMesh<OpenSubdiv::FarStencilFactoryVertex> * mesh;

    FarStencilTables controlStencils;

    OpenSubdiv::FarStencilTablesFactory<> factory(mesh);
    
    for (int i=0; i<nfaces; ++i) {

        HFace * f = mesh->GetFace(i);

        int nv = f->GetNumVertices();

        if (nv!=4) {

            // if the face is not a quad, we have to iterate over sub-quad(rants)
            for (int j=0; j<f->GetNumVertices(); ++j) {

                factory.SetCurrentFace(i,j);

                factory.AppendStencils( &controlStencils, nsamples/nv, u, v, reflevel );
            }
        } else {

            factory.SetCurrentFace(i);

            factory.AppendStencils( &controlStencils, g_nsamples, u, v, reflevel );
        }
    }

When the control vertices (controlPoints) move in space, the limit locations can 
be very efficiently recomputed simply by applying the blending weights to the 
series of coarse control vertices:

.. code:: c++

    class StencilType {
    public:

        void Clear() {
            memset( &x, 0, sizeof(StencilType));
        }

        void AddWithWeight( StencilType const & cv, float weight  ) {
            x += cv.x * weight;
            y += cv.y * weight;
            z += cv.z * weight;
        }

        float x,y,z;
    };

    std::vector<StencilType> controlPoints,
                             points,
                             utan,
                             vtan;
    
    // Uppdate points by applying stencils
    controlStencils.UpdateValues<StencilType>( reinterpret_cast<StencilType const *>(
        &controlPoints[0]), &points[0] );

    // Uppdate tangents by applying derivative stencils
    controlStencils.UpdateDerivs<StencilType>( reinterpret_cast<StencilType const *>(
        &controlPoints[0]), &utan[0], &vtan[0] );

