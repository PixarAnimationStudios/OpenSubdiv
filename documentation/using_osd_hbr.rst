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
  

Using Hbr
---------

.. contents::
   :local:
   :backlinks: none


----

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
is the OsdVertex class for reference:

.. code:: c++

    class OsdVertex {
    public:
        OsdVertex() {}

        OsdVertex(int index) {}

        OsdVertex(OsdVertex const & src) {}

        void AddWithWeight(OsdVertex const & i, float weight, void * = 0) {}

        void AddVaryingWithWeight(const OsdVertex & i, float weight, void * = 0) {}

        void Clear(void * = 0) {}

        void ApplyVertexEdit(HbrVertexEdit<OsdVertex> const &) { }

        void ApplyVertexEdit(FarVertexEdit const &) { }

        void ApplyMovingVertexEdit(HbrMovingVertexEdit<OsdVertex> const &) { }
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

However, currently **Hbr** is not able to handle `non-manifold <subdivision_surfaces.html#manifold-geometry>`__ 
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


The rule-set can be selected using the following accessors:

*Vertex* and *varying* data:

.. code:: c++

    mesh->SetInterpolateBoundaryMethod( OpenSubdiv::HbrMesh<Vertex>::k_InterpolateBoundaryEdgeOnly );

*Face-varying* data:

.. code:: c++

    mesh->SetFVarInterpolateBoundaryMethod( OpenSubdiv::HbrMesh<Vertex>::k_InterpolateBoundaryEdgeOnly );

.. container:: impnotip

   **Note**

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



.. container:: impnotip

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

Here is a simple example that creates a hierarchical vertex edit that corresponds
to `this example <subdivision_surfaces.html#hierarchical-edits-paths>`__.

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

Here is a simple example of how to store face-varying data for a (u,v) pair.
Unlike vertex and varying data which is accessed through the templated vertex
API, face-varying data is aggregated as vectors of float data.


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
            
                fvt.SetAllData(2, fvdata);
            
            } else if (not fvt.CompareAll(2, fvdata)) {
                
                // if there is a boundary in the fvar-data, add the new data
                OpenSubdiv::HbrFVarData<T> & nfvt = e->GetOrgVertex()->NewFVarData(f);
                nfvt.SetAllData(2, fvdata);
                
            }
        }
    }

