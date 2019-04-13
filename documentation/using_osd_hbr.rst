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


Using Hbr
---------

.. contents::
   :local:
   :backlinks: none


----

.. container:: notebox

   **Note**

       As of OpenSubdiv 3.0, all **Hbr** dependencies have been removed from the
       core APIs (**Sdc**, **Vtr**, **Far**, **Osd**). The legacy source code of
       **Hbr** is provided purely for regression and legacy purposes. If your code
       is currently depending on Hbr functionality, we recommend migrating to the
       newer APIs as we cannot guarantee that this code will be maintained in
       future releases.
       For more information see the `3.0 release notes <release_notes.html>`_


Vertex Template API
===================

The **Hbr** API abstracts the vertex class through templating. Client-code is
expected to provide a vertex class that implements the requisite interpolation
functionality.

Here is an example of a simple vertex class that accounts for 3D position, but
does not support arbitrary variables or varying interpolation.

.. code:: c++

    struct Vertex {

        Vertex() { }

        Vertex( int /*i*/ ) { }

        Vertex( const Vertex & src ) { _pos[0]=src._pos[0]; _pos[1]=src._pos[1]; _pos[2]=src._pos[2]; }

       ~Vertex( ) { }

        void AddWithWeight(const Vertex& src, float weight, void * =0 ) {
            _pos[0]+=weight*src._pos[0];
            _pos[1]+=weight*src._pos[1];
            _pos[2]+=weight*src._pos[2];
        }

        void AddVaryingWithWeight(const Vertex& , float, void * =0 ) { }

        void Clear( void * =0 ) { _pos[0]=_pos[1]=_pos[2]=0.0f; }

        void SetPosition(float x, float y, float z) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

        void ApplyVertexEdit(const OpenSubdiv::HbrVertexEdit<Vertex> & edit) {
            const float *src = edit.GetEdit();
            switch(edit.GetOperation()) {
              case OpenSubdiv::HbrHierarchicalEdit<Vertex>::Set:
                _pos[0] = src[0];
                _pos[1] = src[1];
                _pos[2] = src[2];
                break;
              case OpenSubdiv::HbrHierarchicalEdit<Vertex>::Add:
                _pos[0] += src[0];
                _pos[1] += src[1];
                _pos[2] += src[2];
                break;
              case OpenSubdiv::HbrHierarchicalEdit<Vertex>::Subtract:
                _pos[0] -= src[0];
                _pos[1] -= src[1];
                _pos[2] -= src[2];
                break;
            }
        }

        void ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<Vertex> &) { }

        // custom functions & data not required by Hbr -------------------------

        Vertex( float x, float y, float z ) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

        const float * GetPos() const { return _pos; }

        float _pos[3];
    };

In some cases, if only topological analysis is required, the class can be left un-implemented.
Far and Osd for instance store vertex data in serialized interleaved vectors. Here
is the Osd::Vertex class for reference:

.. code:: c++

    class Vertex {
    public:
        Vertex() {}

        Vertex(int /* index */) {}

        Vertex(Vertex const & /* src */) {}

        void AddWithWeight(Vertex const & /* i */, float /* weight */, void * = 0) {}

        void AddVaryingWithWeight(const Vertex & /* i */, float /* weight */, void * = 0) {}

        void Clear(void * = 0) {}

        void ApplyVertexEdit(FarVertexEdit const &) { }
    };



Creating a Mesh
===============

The following tutorial walks through the steps of instantiating a simple **Hbr**
mesh.

The code found in regression/common/shape_utils.h can also be used as an example.
While this implementation covers many of **Hbr**'s features, it does not provide
coverage for the complete Renderman specification though.

----

Instantiating an HbrMesh
************************

First we need to instantiate a mesh object.

**Hbr** supports 3 subdivision schemes:
   * Catmull-Clark (catmark)
   * Loop
   * Bilinear

The scheme is selected by passing an specialized instance of *HbrSubdivision<T>*,
*HbrCatmarkSubdivision<T>* in this case. The scheme can be shared across multiple
mesh objects, so we only need a single instance.

.. code:: c++

    static OpenSubdiv::HbrCatmarkSubdivision<Vertex> _scheme;

    OpenSubdiv::HbrMesh<Vertex> * mesh = new OpenSubdiv::HbrMesh<Vertex>( _scheme );

----

Creating Vertices
*****************

Adding vertices to the mesh is accomplished using the *HbrMesh::NewVertex()* method.

Because **Hbr** uses a dedicated vertex allocator to help alleviate the performance
impact of intensive fragmented memory allocations. This optimization results in
the following design pattern:

.. code:: c++

    Vertex vtx;
    for(int i=0;i<numVerts; i++ ) {
        Vertex * v = mesh->NewVertex( i, vtx);

        // v->SetPosition();
    }

We instantiate a single "default" vertex object named 'vtx' on the stack. We then
recover the pointer to the actual vertex created in the mesh from the NewVertex()
method. Once we have recovered that pointer, we can set the data for our vertex
by using any of the custom accessors.

----

Creating Faces
**************

Once all the vertices have been registered in the mesh, we can start adding the
faces with *HbrMesh::NewFace()*. Assuming we had an *obj* style reader, we need
to know the number of vertices in the face and the indices of these vertices.

.. code:: c++

    for (int f=0; f<numFaces; ++f) {

        int nverts = obj->GetNumVertices(f);

        const int * faceverts = obj->GetFaceVerts(f);

        mesh->NewFace(nv, fv, 0);
    }

However, **Hbr** is not able to handle `non-manifold <subdivision_surfaces.html#non-manifold-topology>`__
geometry. In order to avoid tripping asserts or causing memory violations, let's
rewrite the previous loop with some some prototype code to check the validity of
the topology.

.. code:: c++

    for (int f=0; f<numFaces; ++f) {

        int nv = obj->GetNumVertices(f);

        const int * fv = obj->GetFaceVerts(f);

        // triangles only for Loop subdivision !
        if ((scheme==kLoop) and (nv!=3)) {
            printf("Trying to create a Loop subd with non-triangle face\n");
            continue;
        }

        // now check the half-edges connectivity
        for(int j=0;j<nv;j++) {
            OpenSubdiv::HbrVertex<T> * origin      = mesh->GetVertex( fv[j] );
            OpenSubdiv::HbrVertex<T> * destination = mesh->GetVertex( fv[(j+1)%nv] );
            OpenSubdiv::HbrHalfedge<T> * opposite  = destination->GetEdge(origin);

            if(origin==NULL || destination==NULL) {
                printf(" An edge was specified that connected a nonexistent vertex\n");
                continue;
            }

            if(origin == destination) {
                printf(" An edge was specified that connected a vertex to itself\n");
                continue;
            }

            if(opposite && opposite->GetOpposite() ) {
                printf(" A non-manifold edge incident to more than 2 faces was found\n");
                continue;
            }

            if(origin->GetEdge(destination)) {
                printf(" An edge connecting two vertices was specified more than once."
                       " It's likely that an incident face was flipped\n");
                continue;
            }
        }

        mesh->NewFace(nv, fv, 0);
    }

----

Wrapping Things Up
******************

Once we have vertices and faces set in our mesh, we still need to wrap things up
by calling *HbrMesh::Finish()*:

.. code:: c++

    mesh->Finish()

*Finish* iterates over the mesh to apply the boundary interpolation rules and
checks for singular vertices. At this point, there is one final topology check
remaining to validate the mesh:

.. code:: c++

    mesh->Finish()

    if (mesh->GetNumDisconnectedVertices()) {
        printf("The specified subdivmesh contains disconnected surface components.\n");

        // abort or iterate over the mesh to remove the offending vertices
    }



----

Boundary Interpolation Rules
============================

The rule-set can be selected using the following accessors:

*Vertex* and *varying* data:

.. code:: c++

    mesh->SetInterpolateBoundaryMethod( OpenSubdiv::HbrMesh<Vertex>::k_InterpolateBoundaryEdgeOnly );

*Face-varying* data:

.. code:: c++

    mesh->SetFVarInterpolateBoundaryMethod( OpenSubdiv::HbrMesh<Vertex>::k_InterpolateBoundaryEdgeOnly );


Additional information on boundary interpolation rules can be found
`here <hbr_overview.html#boundary-interpolation-rules>`__

.. container:: impnotip

   **Warning**

   The boundary interpolation rules **must** be set before the call to
   *HbrMesh::Finish()*, which sets the sharpness values to boundary edges
   and vertices based on these rules.

Adding Creases
==============

*Hbr* supports a sharpness attribute on both edges and vertices.


Sharpness is set using the *SetSharpness(float)* accessors.

----

Vertex Creases
**************

Given an index, vertices are very easy to access in the mesh.

.. code:: c++

    int idx;     // vertex index
    float sharp; // the edge sharpness

    OpenSubdiv::HbrVertex<Vertex> * v = mesh->GetVertex( idx );

    if(v) {
        v->SetSharpness( std::max(0.0f, sharp) );
    } else
       printf("cannot find vertex for corner tag (%d)\n", idx );

----

Edge Creases
************

Usually, edge creases are described with a vertex indices pair. Here is some
sample code to locate the matching half-edge and set a crease sharpness.

.. code:: c++

    int v0, v1;  // vertex indices
    float sharp; // the edge sharpness

    OpenSubdiv::HbrVertex<Vertex> * v = mesh->GetVertex( v0 ),
                                  * w = mesh->GetVertex( v1 );

    OpenSubdiv::HbrHalfedge<Vertex> * e = 0;

    if( v && w ) {

        if((e = v->GetEdge(w)) == 0)
            e = w->GetEdge(v);

        if(e) {
            e->SetSharpness( std::max(0.0f, sharp) );
        } else
           printf("cannot find edge for crease tag (%d,%d)\n", v0, v1 );
    }


----

Holes
=====

**Hbr** faces support a "hole" tag.

.. code:: c++

    int idx; // the face index

    OpenSubdiv::HbrFace<Vertex> * f = mesh->GetFace( idx );
    if(f) {
        f->SetHole();
    } else
       printf("cannot find face for hole tag (%d)\n", idx );



.. container:: note

   **Note**

   The hole tag is hierarchical : sub-faces can also be marked as holes.

   See: `Hierarchical Edits`_

----

Hierarchical Edits
==================

**Hbr** supports the following types of hierarchical edits:

+-------------------+----------------------------------------+
| Type              | Function                               |
+===================+========================================+
| Corner edits      | Modify vertex sharpness                |
+-------------------+----------------------------------------+
| Crease edits      | Modify edge sharpness                  |
+-------------------+----------------------------------------+
| FaceEdit          | Modify custom "face data"              |
+-------------------+----------------------------------------+
| FVarEdit          | Modify face-varying data               |
+-------------------+----------------------------------------+
| VertexEdit        | Modify vertex and varying data         |
+-------------------+----------------------------------------+
| HoleEdit          | Set "hole" tag                         |
+-------------------+----------------------------------------+

Modifications are one of the following 3 operations:

+-----------+
| Operation |
+===========+
| Set       |
+-----------+
| Add       |
+-----------+
| Subtract  |
+-----------+

Here is a simple example that creates a hierarchical vertex edit.

.. code:: c++

    // path = 655, 2, 3, 0
    int faceid = 655,
        nsubfaces = 2,
        subfaces[2] = { 2, 3 },
        vertexid = 0;

    int offset = 0,       // offset to the vertex or varying data
        numElems = 3;     // number of elements to apply the modifier to (x,y,z = 3)

    bool isP = false;     // shortcut to identify modifications to the vertex position "P"

    OpenSubdiv::HbrHierarchicalEdit<Vertex>::Operation op =
         OpenSubdiv::HbrHierarchicalEdit<T>::Set;

    float values[3] = { 1.0f, 0.5f, 0.0f }; // edit values

    OpenSubdiv::HbrVertexEdit<T> * edit =
         new OpenSubdiv::HbrVertexEdit<T>(faceid,
                                          nsubfaces,
                                          subfaces,
                                          vertexid,
                                          offset,
                                          floatwidth,
                                          isP,
                                          op,
                                          values);

----

Face-varying Data
=================

Here is a walk-through of how to store face-varying data for a (u,v) pair.
Unlike vertex and varying data which is accessed through the templated vertex
API, face-varying data is directly aggregated as vectors of float data.


Instantiating the *HbrMesh*
***************************

The *HbrMesh* needs to retain some knowledge about the face-varying data that it
carries in order to refine it correctly.

.. code:: c++

    int fvarwidth = 2; // total width of the fvar data

    static int indices[1] = { 0 }, // 1 offset set to 0
               widths[1] = { 2 };  // 2 floats in a (u,v) pair

    int const   fvarcount   = fvarwidth > 0 ? 1 : 0,
              * fvarindices = fvarwidth > 0 ? indices : NULL,
              * fvarwidths  = fvarwidth > 0 ? widths : NULL;

    mesh = new OpenSubdiv::HbrMesh<T>( &_scheme,
                                       fvarcount,
                                       fvarindices,
                                       fvarwidths,
                                       fvarwidth );

Setting the Face-Varying Data
*****************************

After the topology has been created, **Hbr** is ready to accept face-varying data.
Here is some sample code:

.. code:: c++

    for (int i=0, idx=0; i<numFaces; ++i ) {

        OpenSubdiv::HbrFace<Vertex> * f = mesh->GetFace(i);

        int nv = f->GetNumVertices(); // note: this is not the fastest way

        OpenSubdiv::HbrHalfedge<Vertex> * e = f->GetFirstEdge();

        for (int j=0; j<nv; ++j, e=e->GetNext()) {

            OpenSubdiv::HbrFVarData<Vertex> & fvt = e->GetOrgVertex()->GetFVarData(f);

            float const * fvdata = GetFaceVaryingData( i, j );

            if (not fvt.IsInitialized()) {

                // if no fvar daa exists yet on the vertex
                fvt.SetAllData(2, fvdata);

            } else if (not fvt.CompareAll(2, fvdata)) {

                // if there already is fvar data and there is a boundary add the new data
                OpenSubdiv::HbrFVarData<T> & nfvt = e->GetOrgVertex()->NewFVarData(f);
                nfvt.SetAllData(2, fvdata);

            }
        }
    }


Retrieving the Face-Varying Data
********************************

The HbrFVarData structures are expanded during the refinement process, with every
sub-face being assigned a set of interpolated face-varying data. This data can be
accessed in 2 ways :

From a face, passing a vertex index:

.. code:: c++

    // OpenSubdiv::HbrFace<Vertex> * f

    OpenSubdiv::HbrFVarData const &fv = f.GetFVarData(vindex);

    const float * data = fv.GetData()


From a vertex, passing a pointer to an incident face:

.. code:: c++

    // OpenSubdiv::HbrFace<Vertex> * f

    OpenSubdiv::HbrFVarData const &fv = myVertex.GetFVarData(f);

    const float * data = fv.GetData()


----

Valence Operators
=================

When manipulating meshes, it is often necessary to iterate over neighboring faces
or vertices. Rather than gather lists of pointers and return them, Hbr exposes
an operator pattern that guarantees consistent mesh traversals.

The following example shows how to use an operator to count the number of neighboring
vertices (use HbrVertex::GetValence() for proper valence counts)

.. code:: c++

    //OpenSubdiv::HbrVertex<Vertex> * v;

    class MyOperator : public OpenSubdiv::HbrVertexOperator<Vertex> {

    public:
        int count;

        MyOperator() : count(0) { }

        virtual void operator() (OpenSubdiv::HbrVertex<Vertex> &v) {
            ++count;
        }
    };

    MyOperator op;

    v->ApplyOperatorSurroundingVertices( op );

----

Managing Singular Vertices
==========================

Certain topological configurations would force vertices to share multiple
half-edge cycles. Because Hbr is a half-edge representation, these "singular"
vertices have to be duplicated as part of the HbrMesh::Finish() phase of the
instantiation.

These duplicated vertices can cause problems for client-code that tries to
populate buffers of vertex or varying data. The following sample code shows
how to match the vertex data to singular vertex splits:

.. code:: c++

    // Populating an OsdCpuVertexBuffer with vertex data (positions,...)
    float const * vtxData = inMeshFn.getRawPoints(&returnStatus);

    OpenSubdiv::OsdCpuVertexBuffer *vertexBuffer =
        OpenSubdiv::OsdCpuVertexBuffer::Create(numVertexElements, numFarVerts);

    vertexBuffer->UpdateData(vtxData, 0, numVertices );

    // Duplicate the vertex data into the split singular vertices
    std::vector<std::pair<int, int> > const splits = hbrMesh->GetSplitVertices();
    for (int i=0; i<(int)splits.size(); ++i) {
        vertexBuffer->UpdateData(vtxData+splits[i].second*numVertexElements, splits[i].first, 1);
    }
