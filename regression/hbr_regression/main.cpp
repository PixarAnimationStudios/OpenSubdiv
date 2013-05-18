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

#include <hbr/mesh.h>
#include <hbr/face.h>
#include <hbr/vertex.h>
#include <hbr/halfedge.h>
#include <hbr/bilinear.h>
#include <hbr/catmark.h>
#include <hbr/loop.h>

#include "../common/shape_utils.h"

//
// Regression testing matching Hbr to a pre-generated data-set
//

// Precision is currently held at bit-wise identical
#define PRECISION 0

//------------------------------------------------------------------------------
// Vertex class implementation
struct xyzVV {

    xyzVV() { }

    xyzVV( int /*i*/ ) { }

    xyzVV( float x, float y, float z ) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    xyzVV( const xyzVV & src ) { _pos[0]=src._pos[0]; _pos[1]=src._pos[1]; _pos[2]=src._pos[2]; }

   ~xyzVV( ) { }

    void     AddWithWeight(const xyzVV& src, float weight, void * =0 ) { 
        _pos[0]+=weight*src._pos[0]; 
        _pos[1]+=weight*src._pos[1]; 
        _pos[2]+=weight*src._pos[2]; 
    }

    void     AddVaryingWithWeight(const xyzVV& , float, void * =0 ) { }

    void     Clear( void * =0 ) { _pos[0]=_pos[1]=_pos[2]=0.0f; }

    void     SetPosition(float x, float y, float z) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    void     ApplyVertexEdit(const OpenSubdiv::HbrVertexEdit<xyzVV> & edit) {
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

    void     ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<xyzVV> &) { }

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

//------------------------------------------------------------------------------

#include "./init_shapes.h"

static shape * readShape( char const * fname ) {

    FILE * handle = fopen( fname, "rt" );
    if (not handle) {
        printf("Could not open \"%s\" - aborting.\n", fname);
        exit(0);
    }

    fseek( handle, 0, SEEK_END );
    size_t size = ftell(handle);
    fseek( handle, 0, SEEK_SET );

    char * shapeStr = new char[size+1];

    if ( fread( shapeStr, size, 1, handle)!=1 ) {
        printf("Error reading \"%s\" - aborting.\n", fname);
        exit(0);
    }

    fclose(handle);
    
    shapeStr[size]='\0';
    
    return shape::parseShape( shapeStr, 1 );
}

#define STR(x) x

#ifdef  HBR_BASELINE_DIR  
    std::string g_baseline_path = STR(HBR_BASELINE_DIR);
#else
    std::string g_baseline_path;
#endif

//------------------------------------------------------------------------------
static int checkMesh( shaperec const & r, int levels ) {

    int count=0;
    float deltaAvg[3] = {0.0f, 0.0f, 0.0f},
          deltaCnt[3] = {0.0f, 0.0f, 0.0f};

    xyzmesh * mesh = simpleHbr<xyzVV>(r.data.c_str(), r.scheme, 0);

    int firstface=0, lastface=mesh->GetNumFaces(),
        firstvert=0, lastvert=mesh->GetNumVertices(), nverts;    
    
    printf("- %s (scheme=%d)\n", r.name.c_str(), r.scheme);
    
    for (int l=0; l<levels; ++l ) {


        std::stringstream fname;
        
        fname << g_baseline_path <<  r.name << "_level" << l << ".obj";
    
        
        shape * sh = readShape( fname.str().c_str() );
        assert(sh);
        
        // subdivide up to current level
        for (int i=firstface; i<lastface; ++i) {
            xyzface * f = mesh->GetFace(i);
            f->Refine();
        }
        
        firstface = lastface;
        lastface = mesh->GetNumFaces();
        //nfaces = lastface - firstface;

        firstvert = lastvert;
        lastvert = mesh->GetNumVertices();
        nverts = lastvert - firstvert;
        
        for (int i=firstvert; i<lastvert; ++i) {
            const float * apos = mesh->GetVertex(i)->GetData().GetPos(),
                        * bpos = &sh->verts[(i-firstvert)*3];

            if ( apos[0] != bpos[0] )
                deltaCnt[0]++;
            if ( apos[1] != bpos[1] )
                deltaCnt[1]++;
            if ( apos[2] != bpos[2] )
                deltaCnt[2]++;

            float delta[3] = { apos[0] - bpos[0],
                               apos[1] - bpos[1],
                               apos[2] - bpos[2] };

            deltaAvg[0]+=delta[0];
            deltaAvg[1]+=delta[1];
            deltaAvg[2]+=delta[2];
            
            float dist = sqrtf( delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
            if ( dist > PRECISION ) {
                printf("// HbrVertex<T> %d fails : dist=%.10f (%.10f %.10f %.10f)"
                       " (%.10f %.10f %.10f)\n", i, dist, apos[0],
                                                          apos[1],
                                                          apos[2],
                                                          bpos[0],
                                                          bpos[1],
                                                          bpos[2] );
                count++;
            }
        }
        delete sh;
    }
    
    if (deltaCnt[0])
        deltaAvg[0]/=deltaCnt[0];
    if (deltaCnt[1])
        deltaAvg[1]/=deltaCnt[1];
    if (deltaCnt[2])
        deltaAvg[2]/=deltaCnt[2];

    printf("  delta ratio : (%d/%d %d/%d %d/%d)\n", (int)deltaCnt[0], nverts,
                                                    (int)deltaCnt[1], nverts,
                                                    (int)deltaCnt[2], nverts );
    printf("  average delta : (%.10f %.10f %.10f)\n", deltaAvg[0],
                                                      deltaAvg[1],
                                                      deltaAvg[2] );

    if (count==0)
        printf("  success !\n");

    delete mesh;

    return count;
}

//------------------------------------------------------------------------------
int main(int /* argc */, char ** /* argv */) {

    int levels=5, total=0;

    initShapes();

    printf("Baseline Path : \"%s\"\n", g_baseline_path.c_str());

    for (int i=0; i<(int)g_shapes.size(); ++i)
        total+=checkMesh( g_shapes[i], levels );

    if (total==0)
      printf("All tests passed.\n");
    else
      printf("Total failures : %d\n", total);
}
