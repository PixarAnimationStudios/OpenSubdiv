//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#include <stdio.h>

#include <far/meshFactory.h>
#include <far/dispatcher.h>

#include "../common/shape_utils.h"

//
// Regression testing matching Far to Hbr (default CPU implementation)
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

    xyzVV( float x, float y, float z ) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    xyzVV( const xyzVV & src ) { _pos[0]=src._pos[0]; _pos[1]=src._pos[1]; _pos[2]=src._pos[2]; }

   ~xyzVV( ) { }

    void AddWithWeight(const xyzVV& src, float weight, void * =0 ) { 
        _pos[0]+=weight*src._pos[0]; 
        _pos[1]+=weight*src._pos[1]; 
        _pos[2]+=weight*src._pos[2]; 
    }

    void AddVaryingWithWeight(const xyzVV& , float, void * =0 ) { }

    void Clear( void * =0 ) { _pos[0]=_pos[1]=_pos[2]=0.0f; }

    void SetPosition(float x, float y, float z) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    void ApplyVertexEdit(const OpenSubdiv::HbrVertexEdit<xyzVV> & edit) {
        const float *src = edit.GetEdit();
        switch(edit.GetOperation()) {
          case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Set:
            _pos[0] = src[0];
            _pos[1] = src[1];
            _pos[2] = src[2];
            break;
          case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Add:
            _pos[0] += src[0];
            _pos[1] += src[1];
            _pos[2] += src[2];
            break;
          case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Subtract:
            _pos[0] -= src[0];
            _pos[1] -= src[1];
            _pos[2] -= src[2];
            break;
        }
    }

    void ApplyVertexEdit(OpenSubdiv::FarVertexEdit const & edit) {
        const float *src = edit.GetEdit();
        switch(edit.GetOperation()) {
          case OpenSubdiv::FarVertexEdit::Set:
            _pos[0] = src[0];
            _pos[1] = src[1];
            _pos[2] = src[2];
            break;
          case OpenSubdiv::FarVertexEdit::Add:
            _pos[0] += src[0];
            _pos[1] += src[1];
            _pos[2] += src[2];
            break;
        }
    }
    
    void ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<xyzVV> &) { }

    const float * GetPos() const { return _pos; }

private:
    float _pos[3];
};

//------------------------------------------------------------------------------
class xyzFV;
typedef OpenSubdiv::HbrMesh<xyzVV>           xyzmesh;
typedef OpenSubdiv::HbrFace<xyzVV>           xyzface;
typedef OpenSubdiv::HbrVertex<xyzVV>         xyzvertex;
typedef OpenSubdiv::HbrHalfedge<xyzVV>       xyzhalfedge;
typedef OpenSubdiv::HbrFaceOperator<xyzVV>   xyzFaceOperator;
typedef OpenSubdiv::HbrVertexOperator<xyzVV> xyzVertexOperator;

typedef OpenSubdiv::FarMesh<xyzVV>              fMesh;
typedef OpenSubdiv::FarMeshFactory<xyzVV>       fMeshFactory;
typedef OpenSubdiv::FarSubdivisionTables<xyzVV> fSubdivision;
typedef OpenSubdiv::FarPatchTables              fPatches;

static bool g_debugmode = false;
static bool g_dumphbr = false;

//------------------------------------------------------------------------------
// visual debugging using Maya
// python dictionary dump - requires the script createMesh.py to read into Maya
// format is : [ { 'verts':[(1, 0, 0),(2, 0, 0)],
//                 'faces':[[1 2 3 4],[5,6,7,8]] }, ... ]
//------------------------------------------------------------------------------
static void dumpVerts( xyzmesh * mesh, int level ) {
    printf("\t'verts':[\t");
    for (int i=0, counter=0; i<mesh->GetNumVertices(); ++i) {
        xyzvertex * v = mesh->GetVertex(i);
        if ( v->GetFace()->GetDepth()==level) {
            printf("(%10f, %10f, %10f), ",v->GetData().GetPos()[0],
                                          v->GetData().GetPos()[1],
                                          v->GetData().GetPos()[2] );
           counter++;
        }
        if (counter!=0 and (counter+1)%6==0) printf("\n\t\t\t");
    }
    printf("],\n");
}

//------------------------------------------------------------------------------
static void dumpFaces( xyzmesh * mesh, int level ) {
    int vertofs = 0;
    for (int i=0; i<mesh->GetNumVertices(); ++i)
        if (mesh->GetVertex(i)->GetFace()->GetDepth()==level) {
        vertofs = i;
        break;
    }

    printf("\t'faces':[\t");
    int nfaces = mesh->GetNumFaces();

    for (int i=0, counter=0; i<nfaces; ++i) {
        xyzface * f = mesh->GetFace(i);
        if (f->IsHole())
            continue;
        if (f->GetDepth()==level) {
            if (f->GetNumVertices()==4)
                printf("[%6d, %6d, %6d, %6d], ", f->GetVertex(0)->GetID()-vertofs,
                                                 f->GetVertex(1)->GetID()-vertofs,
                                                 f->GetVertex(2)->GetID()-vertofs,
                                                 f->GetVertex(3)->GetID()-vertofs );
            else if (f->GetNumVertices()==3)
                printf("[%6d, %6d, %6d], ", f->GetVertex(0)->GetID()-vertofs,
                                            f->GetVertex(1)->GetID()-vertofs,
                                            f->GetVertex(2)->GetID()-vertofs );

            ++counter;
            if (counter!=0 and (counter+4)%32==0)
                printf("\n\t\t\t");
        }
    }
    printf("]\n");
}

//------------------------------------------------------------------------------
// dump an Hbr mesh to console
static void dumpXYZMesh( xyzmesh * mesh, int level, Scheme /* loop */ =kCatmark ) {
    printf("{ ");
    dumpVerts(mesh, level);
    dumpFaces(mesh, level);
    printf("},\n");
}

//------------------------------------------------------------------------------
static void dumpVerts( fMesh * mesh, int level ) {
    std::vector<xyzVV> & verts = mesh->GetVertices();

    int firstvert = mesh->GetSubdivisionTables()->GetFirstVertexOffset(level),
         numverts = mesh->GetSubdivisionTables()->GetNumVertices(level);

    printf("\t'verts':[\t");
    for (int i=firstvert; i<(firstvert+numverts); ++i) {
        printf("(%10f, %10f, %10f), ",verts[i].GetPos()[0],
                                      verts[i].GetPos()[1],
                                      verts[i].GetPos()[2] );
        if (i!=0 and (i+1)%6==0)
            printf("\n\t\t\t");
    }
    printf("],\n");
}

//------------------------------------------------------------------------------
static void dumpQuadFaces( fMesh * mesh, int level ) {
    
    unsigned int const * fverts = mesh->GetPatchTables()->GetFaceVertices(level);

    int nverts = mesh->GetPatchTables()->GetNumFaces(level) * 4;

    int ofs = mesh->GetSubdivisionTables()->GetFirstVertexOffset(level);

    printf("\t'faces':[\t");
    for (int i=0; i<nverts; i+=4) {
        printf("[%6d, %6d, %6d, %6d], ", fverts[i  ]-ofs,
                                         fverts[i+1]-ofs,
                                         fverts[i+2]-ofs,
                                         fverts[i+3]-ofs );
        if (i!=0 and (i+4)%32==0)
            printf("\n\t\t\t");
    }
    printf("]\n");
}

//------------------------------------------------------------------------------
static void dumpTriFaces( fMesh * mesh, int level ) {

    unsigned int const * fverts = mesh->GetPatchTables()->GetFaceVertices(level);

    int nverts = mesh->GetPatchTables()->GetNumFaces(level) * 3;

    int ofs = mesh->GetSubdivisionTables()->GetFirstVertexOffset(level);

    printf("\t'faces':[\t");
    for (int i=0; i<nverts; i+=3) {
        printf("[%6d, %6d, %6d], ", fverts[i]-ofs, fverts[i+1]-ofs, fverts[i+2]-ofs );
        if (i!=0 and (i+4)%32==0)
            printf("\n\t\t\t");
    }
    printf("]\n");
}

//------------------------------------------------------------------------------
static void dumpMesh( fMesh * mesh, int level, Scheme scheme=kCatmark ) {
    printf("{ ");
    dumpVerts(mesh,level);
    switch (scheme) {
    case kLoop : dumpTriFaces(mesh,level); break;
    case kBilinear :
    case kCatmark : dumpQuadFaces(mesh, level); break;
    }
    printf("},\n");
}

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
int checkMesh( char const * msg, xyzmesh * hmesh, int levels, Scheme scheme=kCatmark ) {

    assert(msg);

    int count=0;
    float deltaAvg[3] = {0.0f, 0.0f, 0.0f},
          deltaCnt[3] = {0.0f, 0.0f, 0.0f};

    fMeshFactory fact( hmesh, levels );
    fMesh * m = fact.Create( );
    OpenSubdiv::FarComputeController<xyzVV>::_DefaultController.Refine(m);

    if (g_debugmode) {
        for (int i=1; i<=levels; ++i)
            if (g_dumphbr)
                dumpXYZMesh( hmesh, i, scheme );
            else
                dumpMesh( m, i, scheme );
    } else
        printf("- %s (scheme=%d)\n", msg, scheme);

    std::vector<int> const & remap = fact.GetRemappingTable();

    int nverts = m->GetNumVertices();

    // compare vertex results (only position for now - we need to expand w/ some vertex data)
    for (int i=1; i<nverts; ++i) {

        xyzvertex * hv = hmesh->GetVertex(i);
        xyzVV & nv = m->GetVertex( remap[hv->GetID()] );

        // boundary interpolation rules set to "none" produce "undefined" vertices on
        // boundary vertices : far does not match hbr for those, so skip comparison.
        if ( hmesh->GetInterpolateBoundaryMethod()==xyzmesh::k_InterpolateBoundaryNone and
             VertexOnBoundary(hv) )
             continue;


        if ( hv->GetData().GetPos()[0] != nv.GetPos()[0] )
            deltaCnt[0]++;
        if ( hv->GetData().GetPos()[1] != nv.GetPos()[1] )
            deltaCnt[1]++;
        if ( hv->GetData().GetPos()[2] != nv.GetPos()[2] )
            deltaCnt[2]++;

        float delta[3] = { hv->GetData().GetPos()[0] - nv.GetPos()[0],
                           hv->GetData().GetPos()[1] - nv.GetPos()[1],
                           hv->GetData().GetPos()[2] - nv.GetPos()[2] };

        deltaAvg[0]+=delta[0];
        deltaAvg[1]+=delta[1];
        deltaAvg[2]+=delta[2];

        float dist = sqrtf( delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
        if ( dist > PRECISION ) {
            if (not g_debugmode)
                printf("// HbrVertex<T> %d fails : dist=%.10f (%.10f %.10f %.10f)"
                       " (%.10f %.10f %.10f)\n", i, dist, hv->GetData().GetPos()[0],
                                                          hv->GetData().GetPos()[1],
                                                          hv->GetData().GetPos()[2],
                                                          nv.GetPos()[0],
                                                          nv.GetPos()[1],
                                                          nv.GetPos()[2] );
           count++;
        }
    }

    if (deltaCnt[0])
        deltaAvg[0]/=deltaCnt[0];
    if (deltaCnt[1])
        deltaAvg[1]/=deltaCnt[1];
    if (deltaCnt[2])
        deltaAvg[2]/=deltaCnt[2];

    if (not g_debugmode) {
        printf("  delta ratio : (%d/%d %d/%d %d/%d)\n", (int)deltaCnt[0], nverts,
                                                        (int)deltaCnt[1], nverts,
                                                        (int)deltaCnt[2], nverts );
        printf("  average delta : (%.10f %.10f %.10f)\n", deltaAvg[0],
                                                          deltaAvg[1],
                                                          deltaAvg[2] );
        if (count==0)
            printf("  success !\n");
    }

    delete hmesh;
    delete m;

    return count;
}

//------------------------------------------------------------------------------
static void parseArgs(int argc, char ** argv) {
    if (argc>1) {
        for (int i=1; i<argc; ++i) {
            if (strcmp(argv[i],"-debug")==0)
                g_debugmode=true;
            else if (strcmp(argv[i],"-dumphbr")==0) {
                g_debugmode=true;
                g_dumphbr=true;
            } else {
                printf("Unknown argument \"%s\". Valid arguments are [\"-debug\", \"-dumphbr\"].\n", argv[i]);
                exit(1);
            }
        }
    }
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    int levels=5, total=0;

    parseArgs(argc, argv);

#define test_catmark_edgeonly
#define test_catmark_edgecorner
#define test_catmark_flap
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
#define test_catmark_square_hedit0
#define test_catmark_square_hedit1
#define test_catmark_square_hedit2
#define test_catmark_square_hedit3

#define test_loop_triangle_edgeonly
#define test_loop_triangle_edgecorner
#define test_loop_icosahedron
#define test_loop_cube
#define test_loop_cube_creases0
#define test_loop_cube_creases1

#define test_bilinear_cube

  if (g_debugmode)
      printf("[ ");
  else
      printf("precision : %f\n",PRECISION);

#ifdef test_catmark_edgeonly
#include "../shapes/catmark_edgeonly.h"
    total += checkMesh( "test_catmark_edgeonly", simpleHbr<xyzVV>(catmark_edgeonly.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_edgecorner
#include "../shapes/catmark_edgecorner.h"
    total += checkMesh( "test_catmark_edgeonly", simpleHbr<xyzVV>(catmark_edgecorner.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_pyramid
#include "../shapes/catmark_pyramid.h"
    total += checkMesh( "test_catmark_pyramid", simpleHbr<xyzVV>(catmark_pyramid.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_pyramid_creases0
#include "../shapes/catmark_pyramid_creases0.h"
    total += checkMesh( "test_catmark_pyramid_creases0", simpleHbr<xyzVV>(catmark_pyramid_creases0.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_pyramid_creases1
#include "../shapes/catmark_pyramid_creases1.h"
    total += checkMesh( "test_catmark_pyramid_creases1", simpleHbr<xyzVV>(catmark_pyramid_creases1.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube
#include "../shapes/catmark_cube.h"
    total += checkMesh( "test_catmark_cube", simpleHbr<xyzVV>(catmark_cube.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube_creases0
#include "../shapes/catmark_cube_creases0.h"
    total += checkMesh( "test_catmark_cube_creases0", simpleHbr<xyzVV>(catmark_cube_creases0.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube_creases1
#include "../shapes/catmark_cube_creases1.h"
    total += checkMesh( "test_catmark_cube_creases1", simpleHbr<xyzVV>(catmark_cube_creases1.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube_corner0
#include "../shapes/catmark_cube_corner0.h"
    total += checkMesh( "test_catmark_cube_corner0", simpleHbr<xyzVV>(catmark_cube_corner0.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube_corner1
#include "../shapes/catmark_cube_corner1.h"
    total += checkMesh( "test_catmark_cube_corner1", simpleHbr<xyzVV>(catmark_cube_corner1.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube_corner2
#include "../shapes/catmark_cube_corner2.h"
    total += checkMesh( "test_catmark_cube_corner2", simpleHbr<xyzVV>(catmark_cube_corner2.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube_corner3
#include "../shapes/catmark_cube_corner3.h"
    total += checkMesh( "test_catmark_cube_corner3", simpleHbr<xyzVV>(catmark_cube_corner3.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_cube_corner4
#include "../shapes/catmark_cube_corner4.h"
    total += checkMesh( "test_catmark_cube_corner4", simpleHbr<xyzVV>(catmark_cube_corner4.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_dart_edgecorner
#include "../shapes/catmark_dart_edgecorner.h"
    total += checkMesh( "test_catmark_dart_edgecorner", simpleHbr<xyzVV>(catmark_dart_edgecorner.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_dart_edgeonly
#include "../shapes/catmark_dart_edgeonly.h"
    total += checkMesh( "test_catmark_dart_edgeonly", simpleHbr<xyzVV>(catmark_dart_edgeonly.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_flap
#include "../shapes/catmark_flap.h"
    total += checkMesh( "test_catmark_flap", simpleHbr<xyzVV>(catmark_flap.c_str(), kCatmark, 0), levels);
#endif

#ifdef test_catmark_tent
#include "../shapes/catmark_tent.h"
    total += checkMesh( "test_catmark_tent", simpleHbr<xyzVV>(catmark_tent.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_tent_creases0
#include "../shapes/catmark_tent_creases0.h"
    total += checkMesh( "test_catmark_tent_creases0", simpleHbr<xyzVV>(catmark_tent_creases0.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_tent_creases1
#include "../shapes/catmark_tent_creases1.h"
    total += checkMesh( "test_catmark_tent_creases1", simpleHbr<xyzVV>(catmark_tent_creases1.c_str(), kCatmark, NULL), levels );
#endif

#ifdef test_catmark_square_hedit0
#include "../shapes/catmark_square_hedit0.h"
    total += checkMesh( "test_catmark_square_hedit0", simpleHbr<xyzVV>(catmark_square_hedit0.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_square_hedit1
#include "../shapes/catmark_square_hedit1.h"
    total += checkMesh( "test_catmark_square_hedit1", simpleHbr<xyzVV>(catmark_square_hedit1.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_square_hedit2
#include "../shapes/catmark_square_hedit2.h"
    total += checkMesh( "test_catmark_square_hedit2", simpleHbr<xyzVV>(catmark_square_hedit2.c_str(), kCatmark, 0), levels );
#endif

#ifdef test_catmark_square_hedit3
#include "../shapes/catmark_square_hedit3.h"
    total += checkMesh( "test_catmark_square_hedit3", simpleHbr<xyzVV>(catmark_square_hedit3.c_str(), kCatmark, 0), levels );
#endif



#ifdef test_loop_triangle_edgeonly
#include "../shapes/loop_triangle_edgeonly.h"
    total += checkMesh( "test_loop_triangle_edgeonly", simpleHbr<xyzVV>(loop_triangle_edgeonly.c_str(), kLoop, 0), levels, kLoop );
#endif

#ifdef test_loop_triangle_edgecorner
#include "../shapes/loop_triangle_edgecorner.h"
    total += checkMesh( "test_loop_triangle_edgecorner", simpleHbr<xyzVV>(loop_triangle_edgecorner.c_str(), kLoop, 0), levels, kLoop );
#endif

#ifdef test_loop_saddle_edgeonly
#include "../shapes/loop_saddle_edgeonly.h"
    total += checkMesh( "test_loop_saddle_edgeonly", simpleHbr<xyzVV>(loop_saddle_edgeonly.c_str(), kLoop, 0), levels, kLoop );
#endif

#ifdef test_loop_saddle_edgecorner
#include "../shapes/loop_saddle_edgecorner.h"
    total += checkMesh( "test_loop_saddle_edgecorner", simpleHbr<xyzVV>(loop_saddle_edgecorner.c_str(), kLoop, 0), levels, kLoop );
#endif

#ifdef test_loop_icosahedron
#include "../shapes/loop_icosahedron.h"
    total += checkMesh( "test_loop_icosahedron", simpleHbr<xyzVV>(loop_icosahedron.c_str(), kLoop, 0), levels, kLoop );
#endif

#ifdef test_loop_cube
#include "../shapes/loop_cube.h"
    total += checkMesh( "test_loop_cube", simpleHbr<xyzVV>(loop_cube.c_str(), kLoop, 0), levels, kLoop );
#endif

#ifdef test_loop_cube_creases0
#include "../shapes/loop_cube_creases0.h"
    total += checkMesh( "test_loop_cube_creases0", simpleHbr<xyzVV>(loop_cube_creases0.c_str(), kLoop, 0), levels, kLoop );
#endif

#ifdef test_loop_cube_creases1
#include "../shapes/loop_cube_creases1.h"
    total += checkMesh( "test_loop_cube_creases1", simpleHbr<xyzVV>(loop_cube_creases1.c_str(), kLoop, 0), levels, kLoop );
#endif



#ifdef test_bilinear_cube
#include "../shapes/bilinear_cube.h"
    total += checkMesh( "test_bilinear_cube", simpleHbr<xyzVV>(bilinear_cube.c_str(), kBilinear, 0), levels, kBilinear );
#endif


    if (g_debugmode)
        printf("]\n");
    else {
        if (total==0)
          printf("All tests passed.\n");
        else
          printf("Total failures : %d\n", total);
    }
}

//------------------------------------------------------------------------------
