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
