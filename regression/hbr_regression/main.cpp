//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include <stdio.h>

#include "../../regression/common/hbr_utils.h"

//
// Regression testing matching Hbr to a pre-generated data-set
//

// Precision is currently held at bit-wise identical
static bool g_allowWeakRegression=true,
            g_strictRegressionFailure=false,
            g_verbose=false;

#define STRICT_PRECISION 0
#define WEAK_PRECISION 1e-6

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

static Shape * readShape( char const * fname, Scheme scheme ) {

    FILE * handle = fopen( fname, "rt" );
    if (! handle) {
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

    return Shape::parseObj( shapeStr, scheme );
}

#define STR(x) x

#ifdef  HBR_BASELINE_DIR
    std::string g_baseline_path = STR(HBR_BASELINE_DIR);
#else
    std::string g_baseline_path;
#endif

//------------------------------------------------------------------------------
static void writeObj( const char * fname, xyzmesh const * mesh,
    int firstface, int lastface, int firstvert, int lastvert ) {

    FILE * handle = fopen( fname, "w" );
    if (! handle) {
        printf("Could not open \"%s\" - aborting.\n", fname);
        exit(0);
    }

    fprintf(handle, "# This file uses centimeters as units for non-parametric coordinates.\n");

    for (int i=firstvert; i<lastvert; ++i) {
        const float * pos = mesh->GetVertex(i)->GetData().GetPos();
        fprintf(handle, "v  %.*g %.*g %.*g\n", 9, pos[0], 9, pos[1], 9, pos[2]);
    }

    fprintf(handle, "s off\n");

    for (int i=firstface; i<lastface; ++i) {
        xyzface * f = mesh->GetFace(i);

        fprintf(handle, "f ");
        for (int j=0; j<f->GetNumVertices();) {

            int vert = f->GetVertex(j)->GetID()-firstvert+1;

            fprintf(handle, "%d", vert);

            if (++j<f->GetNumVertices())
                fprintf(handle, " ");
        }
        fprintf(handle, "\n");
    }

    fclose(handle);
}

//------------------------------------------------------------------------------
static int checkMesh( ShapeDesc const & r, int levels ) {

    int count=0;

    float deltaAvg[3] = {0.0f, 0.0f, 0.0f},
          deltaCnt[3] = {0.0f, 0.0f, 0.0f};

    xyzmesh * mesh = simpleHbr<xyzVV>(r.data.c_str(), r.scheme, 0);

    int firstface=0, lastface=mesh->GetNumFaces(),
        firstvert=0, lastvert=mesh->GetNumVertices(), nverts;

    static char const * schemes[] = { "Bilinear", "Catmark", "Loop" };

    printf("- %-25s ( %-8s ): ", r.name.c_str(), schemes[r.scheme]);

    for (int l=0; l<levels; ++l ) {

        int errcount=0;

        std::stringstream fname;

        fname << g_baseline_path <<  r.name << "_level" << l << ".obj";


        Shape * sh = readShape( fname.str().c_str(), r.scheme );
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
            if ( dist > STRICT_PRECISION ) {
                if(dist < WEAK_PRECISION && g_allowWeakRegression) {
                    g_strictRegressionFailure=true;
                } else {
                    if (g_verbose) {
                        printf("\n// HbrVertex<T> %d fails : dist=%.10f "
                            "(%.10f %.10f %.10f) (%.10f %.10f %.10f)", i, dist,
                                    apos[0], apos[1], apos[2],
                                        bpos[0], bpos[1], bpos[2] );
                    }
                    ++errcount;
                }
            }
        }

        if (errcount) {

            std::stringstream errfile;
            errfile << r.name << "_level" << l << "_error.obj";

            writeObj(errfile.str().c_str(), mesh,
                firstface, lastface, firstvert, lastvert);

            printf("\n  wrote: %s\n", errfile.str().c_str());
        }

        delete sh;
        count += errcount;
    }

    if (deltaCnt[0])
        deltaAvg[0]/=deltaCnt[0];
    if (deltaCnt[1])
        deltaAvg[1]/=deltaCnt[1];
    if (deltaCnt[2])
        deltaAvg[2]/=deltaCnt[2];

    if (g_verbose) {
        printf("\n  delta ratio : (%d/%d %d/%d %d/%d)", (int)deltaCnt[0], nverts,
                                                        (int)deltaCnt[1], nverts,
                                                        (int)deltaCnt[2], nverts );
        printf("\n  average delta : (%.10f %.10f %.10f)", deltaAvg[0],
                                                          deltaAvg[1],
                                                          deltaAvg[2] );
    }

    if (count==0) {
        printf(" success !\n");
    } else
        printf(" failed !\n");

    delete mesh;

    return count;
}

//------------------------------------------------------------------------------
static void usage(char const * appname) {
    printf("Usage : %s [options]\n", appname);
    printf("    -s | -strict  : strict bitwise comparisons\n");
    printf("    -v | -verbose : verbose output\n");
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    int levels=5, total=0;

    for (int i=1; i<argc; ++i) {
        if ((! strcmp(argv[i],"-s")) || (! strcmp(argv[i],"-strict"))) {
            g_allowWeakRegression=false;
        } else if ((! strcmp(argv[i],"-v")) || (! strcmp(argv[i],"-verbose"))) {
            g_verbose=true;
        } else {
            usage( argv[1] );
            return 1;
        }
    }

    initShapes();

    printf("Baseline Path : \"%s\"\n", g_baseline_path.c_str());

    for (int i=0; i<(int)g_shapes.size(); ++i)
        total+=checkMesh( g_shapes[i], levels );

    if (total==0) {
        printf("All tests passed.\n");
        if(g_strictRegressionFailure)
            printf("Some tests were not bit-wise accurate.\n"
                   "Rerun with -s for strict regression\n");
        }
    else
      printf("Total failures : %d\n", total);

}
