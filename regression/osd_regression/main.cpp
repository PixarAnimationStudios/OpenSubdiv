//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
#include <stdio.h>
#include <cassert>

#include <hbr/mesh.h>
#include <hbr/face.h>
#include <hbr/vertex.h>
#include <hbr/halfedge.h>
#include <hbr/catmark.h>

#include <osd/vertex.h>
#include <osd/mesh.h>
#include <osd/cpuDispatcher.h>
#include <osd/glslDispatcher.h>

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaDispatcher.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clDispatcher.h>
#endif

#include "../common/shape_utils.h"

#include <ImathVec.h>


//
// Regression testing matching Osd to Hbr
//
// Notes:
// - precision is currently held at 1e-6
//
// - results cannot be bitwise identical as some vertex interpolations
//   are not happening in the same order. 
//
// - only vertex interpolation is being tested at the moment.
//
#define PRECISION 1e-6

//------------------------------------------------------------------------------
// Vertex class implementation
struct xyzVV {
    xyzVV() { }
    xyzVV( int /*i*/ ) { }
    xyzVV( float x, float y, float z ) : _pos(x,y,z) { }
    xyzVV( const Imath::Vec3<float> & v ) : _pos(v) { }
    xyzVV( const xyzVV & src ) : _pos(src._pos) { }
   ~xyzVV( ) { }

    void     AddWithWeight(const xyzVV& src, float weight, void * =0 ) { _pos+=weight*src._pos; }
    void     AddVaryingWithWeight(const xyzVV& , float, void * =0 ) { }
    void     Clear( void * =0 ) { _pos.setValue(0.f, 0.f, 0.f); }
    void     SetPosition(float x, float y, float z) { _pos=Imath::Vec3<float>(x,y,z); }
    const Imath::Vec3<float>& GetPos() const { return _pos; }

private:  
    Imath::Vec3<float> _pos;
};

//------------------------------------------------------------------------------
class xyzFV;
typedef OpenSubdiv::HbrMesh<xyzVV>           xyzmesh;
typedef OpenSubdiv::HbrFace<xyzVV>           xyzface;        
typedef OpenSubdiv::HbrVertex<xyzVV>         xyzvertex;      
typedef OpenSubdiv::HbrHalfedge<xyzVV>       xyzhalfedge;
typedef OpenSubdiv::HbrFaceOperator<xyzVV>   xyzFaceOperator;
typedef OpenSubdiv::HbrVertexOperator<xyzVV> xyzVertexOperator;

//------------------------------------------------------------------------------
// Returns true if a vertex or any of its parents is on a boundary
bool VertexOnBoundary( xyzvertex const * v ) {
    
    if (not v) 
        return false;
    
    if (v->OnBoundary())
        return true;
    
    xyzvertex const * pv = v->GetParentVertex();
    if (pv)
        return VertexOnBoundary(pv);
    else {
        xyzhalfedge const * pe = v->GetParentEdge();
        if (pe) {
              return VertexOnBoundary(pe->GetOrgVertex()) or 
                     VertexOnBoundary(pe->GetDestVertex());
        } else {
            xyzface const * pf = v->GetParentFace(), * rootf = pf;
            while (pf) {
                pf = pf->GetParent();
                if (pf)
                    rootf=pf;
            }
            if (rootf)
                for (int i=0; i<rootf->GetNumVertices(); ++i)
                    if (rootf->GetVertex(i)->OnBoundary())
                        return true;
        }
    }
    return false;
}

//------------------------------------------------------------------------------

int checkVertexBuffer( xyzmesh * hmesh, 
                       OpenSubdiv::OsdCpuVertexBuffer * vb, 
                       std::vector<int> const & remap) {
    int count=0; 
    Imath::Vec3<float> deltaAvg(0.0, 0.0, 0.0); 
    Imath::Vec3<float> deltaCnt(0,0,0);
                       
    int nverts = hmesh->GetNumVertices();
    for (int i=0; i<nverts; ++i) {

        xyzvertex * hv = hmesh->GetVertex(i);
        
        float * ov = & vb->GetCpuBuffer()[ remap[ hv->GetID() ] * vb->GetNumElements() ];

        // boundary interpolation rules set to "none" produce "undefined" vertices on
        // boundary vertices : far does not match hbr for those, so skip comparison.
        if ( hmesh->GetInterpolateBoundaryMethod()==xyzmesh::k_InterpolateBoundaryNone and
             VertexOnBoundary(hv) )
             continue;


        if ( hv->GetData().GetPos()[0] != ov[0] ) 
            deltaCnt[0]++;
        if ( hv->GetData().GetPos()[1] != ov[1] ) 
            deltaCnt[1]++;
        if ( hv->GetData().GetPos()[2] != ov[2] ) 
            deltaCnt[2]++;

        Imath::Vec3<float> delta = hv->GetData().GetPos() - Imath::Vec3<float>(ov[0],ov[1],ov[2]);

        deltaAvg+=delta;

        float dist = delta.length();
        if ( dist > PRECISION ) {
            printf("// HbrVertex<T> %d fails : dist=%.10f (%.10f %.10f %.10f)"
                   " (%.10f %.10f %.10f)\n", i, dist, hv->GetData().GetPos()[0],
                                                      hv->GetData().GetPos()[1],
                                                      hv->GetData().GetPos()[2],
                                                      ov[0],
                                                      ov[1],
                                                      ov[2] );
           count++;         
        }
    }

    if (deltaCnt[0])
        deltaAvg[0]/=deltaCnt[0];
    if (deltaCnt[1])
        deltaAvg[1]/=deltaCnt[1];
    if (deltaCnt[2])
        deltaAvg[2]/=deltaCnt[2];

    printf("    delta ratio : (%d/%d %d/%d %d/%d)\n", (int)deltaCnt.x, nverts, 
                                                      (int)deltaCnt.y, nverts, 
                                                      (int)deltaCnt.x, nverts );
    printf("    average delta : (%.10f %.10f %.10f)\n", deltaAvg.x, 
                                                        deltaAvg.y,
                                                        deltaAvg.z );
    if (count==0)
        printf("  success !\n");
    
    return count;
}

//------------------------------------------------------------------------------
static void refine( xyzmesh * mesh, int maxlevel ) {

    for (int l=0; l<maxlevel; ++l ) {
        int nfaces = mesh->GetNumFaces();
        for (int i=0; i<nfaces; ++i) {
            xyzface * f = mesh->GetFace(i);
            if (f->GetDepth()==l)
                f->Refine();
        }
    }

}

//------------------------------------------------------------------------------
int checkMesh( char const * msg, char const * shape, int levels, Scheme scheme=kCatmark ) {

    int result =0;

    printf("- %s (scheme=%d)\n", msg, scheme);
    
    xyzmesh * refmesh = simpleHbr<xyzVV>(shape, scheme, 0);
    
    refine( refmesh, levels );


    std::vector<float> coarseverts;
    
    OpenSubdiv::OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape, scheme, coarseverts);

    OpenSubdiv::OsdMesh * omesh = new OpenSubdiv::OsdMesh();


    std::vector<int> remap;

    { 
        omesh->Create(hmesh, levels, (int)OpenSubdiv::OsdKernelDispatcher::kOPENMP, &remap);
    
        OpenSubdiv::OsdCpuVertexBuffer * vb = 
            dynamic_cast<OpenSubdiv::OsdCpuVertexBuffer *>(omesh->InitializeVertexBuffer(3));
        
        vb->UpdateData( & coarseverts[0], (int)coarseverts.size() );
        
        omesh->Subdivide( vb, NULL );
    
        omesh->Synchronize();
        
        checkVertexBuffer(refmesh, vb, remap);        
    }
    
    delete hmesh;
    
    return result;
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    int levels=5, total=0;

    // Register Osd compute kernels
    OpenSubdiv::OsdCpuKernelDispatcher::Register();

#define test_catmark_edgeonly
#define test_catmark_edgecorner
#define test_catmark_pyramid
#define test_catmark_pyramid_creases0
#define test_catmark_pyramid_creases1
#define test_catmark_cube
#define test_catmark_cube_creases0
#define test_catmark_cube_creases1
#define test_catmark_cube_corner0
#define test_catmark_cube_corner1
#define test_catmark_cube_corner2
#define test_catmark_cube_corner3
#define test_catmark_cube_corner4
#define test_catmark_dart_edgeonly
#define test_catmark_dart_edgecorner
#define test_catmark_tent
#define test_catmark_tent_creases0
#define test_catmark_tent_creases1

#define test_loop_triangle_edgeonly
#define test_loop_triangle_edgecorner
#define test_loop_icosahedron
#define test_loop_cube
#define test_loop_cube_creases0
#define test_loop_cube_creases1

#define test_bilinear_cube

    printf("precision : %f\n",PRECISION);

#ifdef test_catmark_edgeonly
#include "../shapes/catmark_edgeonly.h"
    total += checkMesh( "test_catmark_edgeonly", catmark_edgeonly, levels, kCatmark ); 
#endif

#ifdef test_catmark_edgecorner
#include "../shapes/catmark_edgecorner.h"
    total += checkMesh( "test_catmark_edgeonly", catmark_edgecorner, levels, kCatmark ); 
#endif

#ifdef test_catmark_pyramid
#include "../shapes/catmark_pyramid.h"
    total += checkMesh( "test_catmark_pyramid", catmark_pyramid, levels, kCatmark ); 
#endif

#ifdef test_catmark_pyramid_creases0
#include "../shapes/catmark_pyramid_creases0.h"
    total += checkMesh( "test_catmark_pyramid_creases0", catmark_pyramid_creases0, levels, kCatmark ); 
#endif

#ifdef test_catmark_pyramid_creases1
#include "../shapes/catmark_pyramid_creases1.h"
    total += checkMesh( "test_catmark_pyramid_creases1", catmark_pyramid_creases1, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube
#include "../shapes/catmark_cube.h"
    total += checkMesh( "test_catmark_cube", catmark_cube, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube_creases0
#include "../shapes/catmark_cube_creases0.h"
    total += checkMesh( "test_catmark_cube_creases0", catmark_cube_creases0, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube_creases1
#include "../shapes/catmark_cube_creases1.h"
    total += checkMesh( "test_catmark_cube_creases1", catmark_cube_creases1, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube_corner0
#include "../shapes/catmark_cube_corner0.h"
    total += checkMesh( "test_catmark_cube_corner0", catmark_cube_corner0, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube_corner1
#include "../shapes/catmark_cube_corner1.h"
    total += checkMesh( "test_catmark_cube_corner1", catmark_cube_corner1, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube_corner2
#include "../shapes/catmark_cube_corner2.h"
    total += checkMesh( "test_catmark_cube_corner2", catmark_cube_corner2, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube_corner3
#include "../shapes/catmark_cube_corner3.h"
    total += checkMesh( "test_catmark_cube_corner3", catmark_cube_corner3, levels, kCatmark ); 
#endif

#ifdef test_catmark_cube_corner4
#include "../shapes/catmark_cube_corner4.h"
    total += checkMesh( "test_catmark_cube_corner4", catmark_cube_corner4, levels, kCatmark ); 
#endif

#ifdef test_catmark_dart_edgecorner
#include "../shapes/catmark_dart_edgecorner.h"
    total += checkMesh( "test_catmark_dart_edgecorner", catmark_dart_edgecorner, levels, kCatmark ); 
#endif

#ifdef test_catmark_dart_edgeonly
#include "../shapes/catmark_dart_edgeonly.h"
    total += checkMesh( "test_catmark_dart_edgeonly", catmark_dart_edgeonly, levels, kCatmark ); 
#endif

#ifdef test_catmark_tent
#include "../shapes/catmark_tent.h"
    total += checkMesh( "test_catmark_tent", catmark_tent, levels, kCatmark ); 
#endif

#ifdef test_catmark_tent_creases0
#include "../shapes/catmark_tent_creases0.h"
    total += checkMesh( "test_catmark_tent_creases0", catmark_tent_creases0, levels ); 
#endif

#ifdef test_catmark_tent_creases1
#include "../shapes/catmark_tent_creases1.h"
    total += checkMesh( "test_catmark_tent_creases1", catmark_tent_creases1, levels ); 
#endif



#ifdef test_loop_triangle_edgeonly
#include "../shapes/loop_triangle_edgeonly.h"
    total += checkMesh( "test_loop_triangle_edgeonly", loop_triangle_edgeonly, levels, kLoop ); 
#endif

#ifdef test_loop_triangle_edgecorner
#include "../shapes/loop_triangle_edgecorner.h"
    total += checkMesh( "test_loop_triangle_edgecorner", loop_triangle_edgecorner, levels, kLoop ); 
#endif

#ifdef test_loop_saddle_edgeonly
#include "../shapes/loop_saddle_edgeonly.h"
    total += checkMesh( "test_loop_saddle_edgeonly", loop_saddle_edgeonly, levels, kLoop ); 
#endif

#ifdef test_loop_saddle_edgecorner
#include "../shapes/loop_saddle_edgecorner.h"
    total += checkMesh( "test_loop_saddle_edgecorner", loop_saddle_edgecorner, levels, kLoop ); 
#endif

#ifdef test_loop_icosahedron
#include "../shapes/loop_icosahedron.h"
    total += checkMesh( "test_loop_icosahedron", loop_icosahedron, levels, kLoop ); 
#endif

#ifdef test_loop_cube
#include "../shapes/loop_cube.h"
    total += checkMesh( "test_loop_cube", loop_cube, levels, kLoop ); 
#endif

#ifdef test_loop_cube_creases0
#include "../shapes/loop_cube_creases0.h"
    total += checkMesh( "test_loop_cube_creases0", loop_cube_creases0,levels, kLoop ); 
#endif

#ifdef test_loop_cube_creases1
#include "../shapes/loop_cube_creases1.h"
    total += checkMesh( "test_loop_cube_creases1", loop_cube_creases1, levels, kLoop ); 
#endif



#ifdef test_bilinear_cube
#include "../shapes/bilinear_cube.h"
    total += checkMesh( "test_bilinear_cube", bilinear_cube, levels, kBilinear ); 
#endif

    if (total==0)
      printf("All tests passed.\n");
    else
      printf("Total failures : %d\n", total);
}
