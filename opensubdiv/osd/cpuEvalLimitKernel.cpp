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

#include "../osd/cpuEvalLimitKernel.h"

#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <algorithm>
#include <vector>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
evalBilinear(float u, float v,
             unsigned int const * vertexIndices,
             OsdVertexBufferDescriptor const & inDesc,
             float const * inQ,
             OsdVertexBufferDescriptor const & outDesc,
             float * outQ) {

    assert( inDesc.length <= (outDesc.stride-outDesc.offset) );

    float const * inOffset = inQ + inDesc.offset;

    float * Q = outQ + outDesc.offset;
    
    memset(Q, 0, inDesc.length*sizeof(float));

    float ou = 1.0f - u, 
          ov = 1.0f - v,
          w[4] = { ov*ou, v*ou, v*u, ov*u };

    for (int i=0; i<4; ++i) {

        float const * in = inOffset + vertexIndices[i]*inDesc.stride;

        for (int k=0; k<inDesc.length; ++k) {
            Q[k] += w[i] * in[k];
        }
    }
}
        

inline void
evalCubicBSpline(float u, float B[4], float BU[4]) {
    float t = u;
    float s = 1.0f - u;

    float A0 =                      s * (0.5f * s);
    float A1 = t * (s + 0.5f * t) + s * (0.5f * s + t);
    float A2 = t * (    0.5f * t);

    B[0] =                                     1.f/3.f * s                * A0;
    B[1] = (2.f/3.f * s +           t) * A0 + (2.f/3.f * s + 1.f/3.f * t) * A1;
    B[2] = (1.f/3.f * s + 2.f/3.f * t) * A1 + (          s + 2.f/3.f * t) * A2;
    B[3] =                1.f/3.f * t  * A2;

    if (BU) {
        BU[0] =    - A0;
        BU[1] = A0 - A1;
        BU[2] = A1 - A2;
        BU[3] = A2;
    }
}



void
evalBSpline(float u, float v, 
            unsigned int const * vertexIndices,
            OsdVertexBufferDescriptor const & inDesc,
            float const * inQ, 
            OsdVertexBufferDescriptor const & outDesc,
            float * outQ, 
            float * outDQU,
            float * outDQV ) {

    // make sure that we have enough space to store results
    assert( inDesc.length <= (outDesc.stride-outDesc.offset) );

    bool evalDeriv = (outDQU or outDQV);

    float B[4], D[4],
          *BU=(float*)alloca(inDesc.length*4*sizeof(float)),
          *DU=(float*)alloca(inDesc.length*4*sizeof(float));
    
    memset(BU, 0, inDesc.length*4*sizeof(float));
    memset(DU, 0, inDesc.length*4*sizeof(float));

    evalCubicBSpline(u, B, evalDeriv ? D : 0);

    float const * inOffset = inQ + inDesc.offset;

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
        
            float const * in = inOffset + vertexIndices[i+j*4]*inDesc.stride;
            
            for (int k=0; k<inDesc.length; ++k) {
            
                BU[i*inDesc.length+k] += in[k] * B[j];
                
                if (evalDeriv)
                    DU[i*inDesc.length+k] += in[k] * D[j];                
            }
        }
    }

    evalCubicBSpline(v, B, evalDeriv ? D : 0);

    float * Q = outQ + outDesc.offset,
          * dQU = outDQU + outDesc.offset,
          * dQV = outDQV + outDesc.offset;

    // clear result 
    memset(Q, 0, inDesc.length*sizeof(float));
    if (evalDeriv) {
        memset(dQU, 0, inDesc.length*sizeof(float));
        memset(dQV, 0, inDesc.length*sizeof(float));
    }

    for (int i=0; i<4; ++i) {
        for (int k=0; k<inDesc.length; ++k) {
            Q[k] += BU[inDesc.length*i+k] * B[i];
            
            if (evalDeriv) {
                dQU[k] += DU[inDesc.length*i+k] * B[i];
                dQV[k] += BU[inDesc.length*i+k] * D[i];
            }
        }
    }    
}             



void
evalBoundary(float u, float v, 
             unsigned int const * vertexIndices,
             OsdVertexBufferDescriptor const & inDesc,
             float const * inQ,
             OsdVertexBufferDescriptor const & outDesc,
             float * outQ,
             float * outDQU,
             float * outDQV ) {

    assert( inDesc.length <= (outDesc.stride-outDesc.offset) );

    bool evalDeriv = (outDQU or outDQV);

    float B[4], D[4],
          *BU=(float*)alloca(inDesc.length*4*sizeof(float)),
          *DU=(float*)alloca(inDesc.length*4*sizeof(float));
    
    memset(BU, 0, inDesc.length*4*sizeof(float));
    memset(DU, 0, inDesc.length*4*sizeof(float));

    evalCubicBSpline(u, B, evalDeriv ? D : 0);

    float const * inOffset = inQ + inDesc.offset;


    // mirror the missing vertices (M)
    //
    //  M0 -- M1 -- M2 -- M3 (corner)
    //   |     |     |     |
    //   |     |     |     |
    //  v0 -- v1 -- v2 -- v3    M : mirrored
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v4 -- v5 -- v6 -- v7    v : original Cv
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v8 -- v9 -- v10-- v11
    
    float *M = (float*)alloca(inDesc.length*4*sizeof(float));

    float const *v0 = inOffset + vertexIndices[0]*inDesc.stride,
                *v1 = inOffset + vertexIndices[1]*inDesc.stride,
                *v2 = inOffset + vertexIndices[2]*inDesc.stride,
                *v3 = inOffset + vertexIndices[3]*inDesc.stride,
                *v4 = inOffset + vertexIndices[4]*inDesc.stride,
                *v5 = inOffset + vertexIndices[5]*inDesc.stride,
                *v6 = inOffset + vertexIndices[6]*inDesc.stride,
                *v7 = inOffset + vertexIndices[7]*inDesc.stride;

    for (int k=0; k<inDesc.stride; ++k) {
        M[0*inDesc.length+k] = 2.0f*v0[k] - v4[k];  // M0 = 2*v0 - v3
        M[1*inDesc.length+k] = 2.0f*v1[k] - v5[k];  // M0 = 2*v1 - v4
        M[2*inDesc.length+k] = 2.0f*v2[k] - v6[k];  // M1 = 2*v2 - v5
        M[3*inDesc.length+k] = 2.0f*v3[k] - v7[k];  // M4 = 2*v2 - v1
    }
    
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
        
            // swap the missing row of verts with our mirrored ones
            float const * in = j==0 ? &M[i*inDesc.stride] :
                inOffset + vertexIndices[i+(j-1)*4]*inDesc.stride;
            
            for (int k=0; k<inDesc.length; ++k) {
            
                BU[i*inDesc.length+k] += in[k] * B[j];
                
                if (evalDeriv)
                    DU[i*inDesc.length+k] += in[k] * D[j];                
            }
        }
    }

    evalCubicBSpline(v, B, evalDeriv ? D : 0);

    float * Q = outQ + outDesc.offset,
          * dQU = outDQU + outDesc.offset,
          * dQV = outDQV + outDesc.offset;

    // clear result 
    memset(Q, 0, inDesc.length*sizeof(float));
    if (evalDeriv) {
        memset(dQU, 0, inDesc.length*sizeof(float));
        memset(dQV, 0, inDesc.length*sizeof(float));
    }

    for (int i=0; i<4; ++i) {
        for (int k=0; k<inDesc.length; ++k) {
            Q[k] += BU[inDesc.length*i+k] * B[i];
            
            if (evalDeriv) {
                dQU[k] += DU[inDesc.length*i+k] * B[i];
                dQV[k] += BU[inDesc.length*i+k] * D[i];
            }
        }
    }    
}



void
evalCorner(float u, float v, 
           unsigned int const * vertexIndices,
           OsdVertexBufferDescriptor const & inDesc,
           float const * inQ,
           OsdVertexBufferDescriptor const & outDesc,
           float * outQ,
           float * outDQU,
           float * outDQV ) {

    assert( inDesc.length <= (outDesc.stride-outDesc.offset) );

    int length = inDesc.length;

    bool evalDeriv = (outDQU or outDQV);

    float B[4], D[4],
          *BU=(float*)alloca(length*4*sizeof(float)),
          *DU=(float*)alloca(length*4*sizeof(float));
    
    memset(BU, 0, length*4*sizeof(float));
    memset(DU, 0, length*4*sizeof(float));


    evalCubicBSpline(u, B, evalDeriv ? D : 0);

    float const *inOffset = inQ + inDesc.offset;

    // mirror the missing vertices (M)
    //
    //  M0 -- M1 -- M2 -- M3 (corner)
    //   |     |     |     |
    //   |     |     |     |
    //  v0 -- v1 -- v2 -- M4    M : mirrored
    //   |.....|.....|     |
    //   |.....|.....|     |
    //  v3.--.v4.--.v5 -- M5    v : original Cv
    //   |.....|.....|     |
    //   |.....|.....|     |
    //  v6 -- v7 -- v8 -- M6
    
    float *M = (float*)alloca(length*7*sizeof(float));

    float const *v0 = inOffset + vertexIndices[0]*inDesc.stride,
                *v1 = inOffset + vertexIndices[1]*inDesc.stride,
                *v2 = inOffset + vertexIndices[2]*inDesc.stride,
                *v3 = inOffset + vertexIndices[3]*inDesc.stride,
                *v4 = inOffset + vertexIndices[4]*inDesc.stride,
                *v5 = inOffset + vertexIndices[5]*inDesc.stride,
                *v7 = inOffset + vertexIndices[7]*inDesc.stride,
                *v8 = inOffset + vertexIndices[8]*inDesc.stride;

    for (int k=0; k<inDesc.stride; ++k) {
        M[0*length+k] = 2.0f*v0[k] - v3[k];  // M0 = 2*v0 - v3
        M[1*length+k] = 2.0f*v1[k] - v4[k];  // M0 = 2*v1 - v4
        M[2*length+k] = 2.0f*v2[k] - v5[k];  // M1 = 2*v2 - v5

        M[4*length+k] = 2.0f*v2[k] - v1[k];  // M4 = 2*v2 - v1
        M[5*length+k] = 2.0f*v5[k] - v4[k];  // M5 = 2*v5 - v4
        M[6*length+k] = 2.0f*v8[k] - v7[k];  // M6 = 2*v8 - v7
        
        // M3 = 2*M2 - M1
        M[3*length+k] = 2.0f*M[2*length+k] - M[1*length+k];
    }

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
        
            float const * in = NULL;

            if (j==0) { // (2)
                in = &M[i*inDesc.stride];
            } else if (i==3) {
                in = &M[(j+3)*inDesc.stride];
            } else {
                in = inOffset + vertexIndices[i+(j-1)*3]*inDesc.stride;
            }

            assert(in);
                        
            for (int k=0; k<length; ++k) {
            
                BU[i*length+k] += in[k] * B[j];
                
                if (evalDeriv)
                    DU[i*length+k] += in[k] * D[j];                
            }
        }
    }

    evalCubicBSpline(v, B, evalDeriv ? D : 0);

    float * Q = outQ + outDesc.offset,
          * dQU = outDQU + outDesc.offset,
          * dQV = outDQV + outDesc.offset;

    // clear result 
    memset(Q, 0, length*sizeof(float));
    if (evalDeriv) {
        memset(dQU, 0, length*sizeof(float));
        memset(dQV, 0, length*sizeof(float));
    }

    for (int i=0; i<4; ++i) {
        for (int k=0; k<length; ++k) {
            Q[k] += BU[length*i+k] * B[i];
            
            if (evalDeriv) {
                dQU[k] += DU[length*i+k] * B[i];
                dQV[k] += BU[length*i+k] * D[i];
            }
        }
    }    
}


static float ef_small[7] = {
    0.813008f, 0.500000f, 0.363636f, 0.287505f,
    0.238692f, 0.204549f, 0.179211f };
/*
static float ef_large[27] = {
    0.812816f, 0.500000f, 0.363644f, 0.287514f,
    0.238688f, 0.204544f, 0.179229f, 0.159657f,
    0.144042f, 0.131276f, 0.120632f, 0.111614f,
    0.103872f, 0.09715f, 0.0912559f, 0.0860444f,
    0.0814022f, 0.0772401f, 0.0734867f, 0.0700842f,
    0.0669851f, 0.0641504f, 0.0615475f, 0.0591488f,
    0.0569311f, 0.0548745f, 0.0529621f
};
*/

inline void
univar4x4(float u, float B[4], float D[4])
{
    float t = u;
    float s = 1.0f - u;

    float A0 = s * s;
    float A1 = 2 * s * t;
    float A2 = t * t;

    B[0] = s * A0;
    B[1] = t * A0 + s * A1;
    B[2] = t * A1 + s * A2;
    B[3] = t * A2;

    if (D) {
        D[0] =    - A0;
        D[1] = A0 - A1;
        D[2] = A1 - A2;
        D[3] = A2;
    }
}

inline float 
csf(unsigned int n, unsigned int j)
{
    if (j%2 == 0) {
        return cosf((2.0f * float(M_PI) * float(float(j-0)/2.0f))/(float(n)+3.0f));
    } else {
        return sinf((2.0f * float(M_PI) * float(float(j-1)/2.0f))/(float(n)+3.0f));
    }
}


void
evalGregory(float u, float v,
            unsigned int const * vertexIndices,
            int const * vertexValenceBuffer,
            unsigned int const  * quadOffsetBuffer,
            int maxValence,
            OsdVertexBufferDescriptor const & inDesc,
            float const * inQ, 
            OsdVertexBufferDescriptor const & outDesc,
            float * outQ, 
            float * outDQU,
            float * outDQV )
{
    // vertex

    // make sure that we have enough space to store results
    assert( inDesc.length <= (outDesc.stride-outDesc.offset) );

    bool evalDeriv = (outDQU or outDQV);

    int valences[4], length=inDesc.length;
    
    float const * inOffset = inQ + inDesc.offset;
    
    float  *r  = (float*)alloca((maxValence+2)*4*length*sizeof(float)), *rp,
           *e0 = r + maxValence*4*length,
           *e1 = e0 + 4*length;
    memset(r, 0, (maxValence+2)*4*length*sizeof(float));
          
    float *f=(float*)alloca(maxValence*length*sizeof(float)),
          *pos=(float*)alloca(length*sizeof(float)),
          *opos=(float*)alloca(length*4*sizeof(float));
    memset(opos, 0, length*4*sizeof(float));
    
    for (int vid=0; vid < 4; ++vid) {
    
        int vertexID = vertexIndices[vid];
        
        const int *valenceTable = vertexValenceBuffer + vertexID * (2*maxValence+1);
        int valence = abs(*valenceTable);
        assert(valence<=maxValence);
        valences[vid] = valence;
        
        memcpy(pos, inOffset + vertexID*inDesc.stride, length*sizeof(float));
        
        rp=r+vid*maxValence*length;
        
        int vofs = vid*length;
        
        for (int i=0; i<valence; ++i) {
            unsigned int im = (i+valence-1)%valence,
                         ip = (i+1)%valence;
            
            int idx_neighbor   = valenceTable[2*i  + 0 + 1];
            int idx_diagonal   = valenceTable[2*i  + 1 + 1];
            int idx_neighbor_p = valenceTable[2*ip + 0 + 1];
            int idx_neighbor_m = valenceTable[2*im + 0 + 1];
            int idx_diagonal_m = valenceTable[2*im + 1 + 1];

            float const * neighbor   = inOffset + idx_neighbor   * inDesc.stride;
            float const * diagonal   = inOffset + idx_diagonal   * inDesc.stride;
            float const * neighbor_p = inOffset + idx_neighbor_p * inDesc.stride;
            float const * neighbor_m = inOffset + idx_neighbor_m * inDesc.stride;
            float const * diagonal_m = inOffset + idx_diagonal_m * inDesc.stride;
            
            float  *fp = f+i*length;
        
            for (int k=0; k<length; ++k) {
                fp[k] = (pos[k]*float(valence) + (neighbor_p[k]+neighbor[k])*2.0f + diagonal[k])/(float(valence)+5.0f);
                
                opos[vofs+k] += fp[k];
                rp[i*length+k] =(neighbor_p[k]-neighbor_m[k])/3.0f + (diagonal[k]-diagonal_m[k])/6.0f;
            }
            
        }
        
        for (int k=0; k<length; ++k) {
            opos[vofs+k] /= valence;
        }
        
        for (int i=0; i<valence; ++i) {
            int im = (i+valence-1)%valence;
            for (int k=0; k<length; ++k) {
                float e = 0.5f*(f[i*length+k]+f[im*length+k]);
                e0[vofs+k] += csf(valence-3, 2*i) * e;
                e1[vofs+k] += csf(valence-3, 2*i+1) * e;
            }
        }
        
        for (int k=0; k<length; ++k) {
            e0[vofs+k] *= ef_small[valence-3];
            e1[vofs+k] *= ef_small[valence-3];
        }       
    }
    
    // tess control
    
    // Control Vertices based on : 
    // "Approximating Subdivision Surfaces with Gregory Patches for Hardware Tessellation" 
    // Loop, Schaefer, Ni, Castafio (ACM ToG Siggraph Asia 2009)
    //
    //  P3         e3-      e2+         E2
    //     O--------O--------O--------O
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |        O        O        |
    // e3+ O------O            O------O e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- O------O            O------O e1+
    //     |        O        O        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------O--------O--------O
    //  P0         e0+      e1-         E1
    //

    float *Ep=(float*)alloca(length*4*sizeof(float)), 
          *Em=(float*)alloca(length*4*sizeof(float)), 
          *Fp=(float*)alloca(length*4*sizeof(float)), 
          *Fm=(float*)alloca(length*4*sizeof(float));

    for (int vid=0; vid<4; ++vid) {
    
        int ip = (vid+1)%4;
        int im = (vid+3)%4;
        int n = valences[vid];
        unsigned int const *quadOffsets = quadOffsetBuffer;

        int start = quadOffsets[vid] & 0x00ff;
        int prev = (quadOffsets[vid] & 0xff00) / 256;

        for (int k=0, ofs=vid*length; k<length; ++k, ++ofs) {
        
            Ep[ofs] = opos[ofs] + e0[ofs] * csf(n-3, 2*start) + e1[ofs]*csf(n-3, 2*start +1);
            Em[ofs] = opos[ofs] + e0[ofs] * csf(n-3, 2*prev ) + e1[ofs]*csf(n-3, 2*prev + 1);
        }
        
        unsigned int np = valences[ip],
                     nm = valences[im];

        unsigned int prev_p = (quadOffsets[ip] & 0xff00) / 256,
                    start_m = quadOffsets[im] & 0x00ff;
                    
        float *Em_ip=(float*)alloca(length*sizeof(float)), 
              *Ep_im=(float*)alloca(length*sizeof(float));
        
        for (int k=0, ipofs=ip*length, imofs=im*length; k<length; ++k, ++ipofs, ++imofs) {
            Em_ip[k] = opos[ipofs] + e0[ipofs]*csf(np-3, 2*prev_p)  + e1[ipofs]*csf(np-3, 2*prev_p+1);
            Ep_im[k] = opos[imofs] + e0[imofs]*csf(nm-3, 2*start_m) + e1[imofs]*csf(nm-3, 2*start_m+1);
        }

        float s1 = 3.0f - 2.0f*csf(n-3,2)-csf(np-3,2),
              s2 = 2.0f*csf(n-3,2),
              s3 = 3.0f -2.0f*cosf(2.0f*float(M_PI)/float(n)) - cosf(2.0f*float(M_PI)/float(nm));

        rp = r + vid*maxValence*length;
        for (int k=0, ofs=vid*length; k<length; ++k, ++ofs) {
            Fp[ofs] = (csf(np-3,2)*opos[ofs] + s1*Ep[ofs] + s2*Em_ip[k] + rp[start*length+k])/3.0f;
            Fm[ofs] = (csf(nm-3,2)*opos[ofs] + s3*Em[ofs] + s2*Ep_im[k] - rp[prev*length+k])/3.0f;
        }
    }

    float * p[20];    
    for (int i=0, ofs=0; i<4; ++i, ofs+=length) {    
        p[i*5+0] = opos + ofs;
        p[i*5+1] =   Ep + ofs;
        p[i*5+2] =   Em + ofs;
        p[i*5+3] =   Fp + ofs;
        p[i*5+4] =   Fm + ofs;
    }    

    float U = 1-u, V=1-v;
    float d11 = u+v; if(u+v==0.0f) d11 = 1.0f;
    float d12 = U+v; if(U+v==0.0f) d12 = 1.0f;
    float d21 = u+V; if(u+V==0.0f) d21 = 1.0f;
    float d22 = U+V; if(U+V==0.0f) d22 = 1.0f;
    
    float *q=(float*)alloca(length*16*sizeof(float));
    for (int k=0; k<length; ++k) {
        q[ 5*length+k] = (u*p[ 3][k] + v*p[ 4][k])/d11;
        q[ 6*length+k] = (U*p[ 9][k] + v*p[ 8][k])/d12;
        q[ 9*length+k] = (u*p[19][k] + V*p[18][k])/d21;
        q[10*length+k] = (U*p[13][k] + V*p[14][k])/d22;        
    }

    memcpy(q+ 0*length, p[ 0], length*sizeof(float));
    memcpy(q+ 1*length, p[ 1], length*sizeof(float));
    memcpy(q+ 2*length, p[ 7], length*sizeof(float));
    memcpy(q+ 3*length, p[ 5], length*sizeof(float));
    memcpy(q+ 4*length, p[ 2], length*sizeof(float));
    memcpy(q+ 7*length, p[ 6], length*sizeof(float));
    memcpy(q+ 8*length, p[16], length*sizeof(float));
    memcpy(q+11*length, p[12], length*sizeof(float));
    memcpy(q+12*length, p[15], length*sizeof(float));
    memcpy(q+13*length, p[17], length*sizeof(float));
    memcpy(q+14*length, p[11], length*sizeof(float));
    memcpy(q+15*length, p[10], length*sizeof(float));

    float B[4], D[4], 
          *BU=(float*)alloca(inDesc.length*4*sizeof(float)), 
          *DU=(float*)alloca(inDesc.length*4*sizeof(float));
    memset(BU, 0, inDesc.length*4*sizeof(float));
    memset(DU, 0, inDesc.length*4*sizeof(float));

    univar4x4(u, B, evalDeriv ? D : 0);

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
        
            float const * in = q + (i+j*4)*length;
            
            for (int k=0; k<inDesc.length; ++k) {
            
                BU[i*inDesc.length+k] += in[k] * B[j];
                
                if (evalDeriv)
                    DU[i*inDesc.length+k] += in[k] * D[j];                
            }
        }
    }

    univar4x4(v, B, evalDeriv ? D : 0);

    float * Q = outQ + outDesc.offset;
    float * dQU = outDQU + outDesc.offset;
    float * dQV = outDQV + outDesc.offset;

    // clear result 
    memset(Q, 0, outDesc.length*sizeof(float));
    if (evalDeriv) {
        memset(dQU, 0, outDesc.length*sizeof(float));
        memset(dQV, 0, outDesc.length*sizeof(float));
    }

    for (int i=0; i<4; ++i) {
        for (int k=0; k<inDesc.length; ++k) {
            Q[k] += BU[inDesc.length*i+k] * B[i];
            
            if (evalDeriv) {
                dQU[k] += DU[inDesc.length*i+k] * B[i];
                dQV[k] += BU[inDesc.length*i+k] * D[i];
            }
        }
    }    
}


void
evalGregoryBoundary(float u, float v,
                    unsigned int const * vertexIndices,
                    int const * vertexValenceBuffer,
                    unsigned int const  * quadOffsetBuffer,
                    int maxValence,
                    OsdVertexBufferDescriptor const & inDesc,
                    float const * inQ,
                    OsdVertexBufferDescriptor const & outDesc,
                    float * outQ,
                    float * outDQU,
                    float * outDQV )
{    
    // vertex

    // make sure that we have enough space to store results
    assert( inDesc.length <= (outDesc.stride-outDesc.offset) );

    bool evalDeriv = (outDQU or outDQV);

    int valences[4], zerothNeighbors[4], length=inDesc.length;

    float const * inOffset = inQ + inDesc.offset;

    float  *r  = (float*)alloca((maxValence+2)*4*length*sizeof(float)), *rp,
           *e0 = r + maxValence*4*length,
           *e1 = e0 + 4*length;
    memset(r, 0, (maxValence+2)*4*length*sizeof(float));
          
    float *f=(float*)alloca(maxValence*length*sizeof(float)),
          *org=(float*)alloca(length*4*sizeof(float)),
          *opos=(float*)alloca(length*4*sizeof(float));

    memset(opos, 0, length*4*sizeof(float));

    for (int vid=0; vid < 4; ++vid) {

        int vertexID = vertexIndices[vid];

        const int *valenceTable = vertexValenceBuffer + vertexID * (2*maxValence+1);
        int valence = *valenceTable,
            ivalence = abs(valence);

        assert(ivalence<=maxValence);
        valences[vid] = valence;

        int vofs = vid * length;

        float *pos=org + vofs;
        memcpy(pos, inOffset + vertexID*inDesc.stride, length*sizeof(float));

        int boundaryEdgeNeighbors[2];
        unsigned int currNeighbor = 0,
                     ibefore=0, 
                     zerothNeighbor=0;

        rp=r+vid*maxValence*length;

        for (int i=0; i<ivalence; ++i) {
            unsigned int im = (i+ivalence-1)%ivalence,
                         ip = (i+1)%ivalence;

            int idx_neighbor   = valenceTable[2*i  + 0 + 1];
            int idx_diagonal   = valenceTable[2*i  + 1 + 1];
            int idx_neighbor_p = valenceTable[2*ip + 0 + 1];
            int idx_neighbor_m = valenceTable[2*im + 0 + 1];
            int idx_diagonal_m = valenceTable[2*im + 1 + 1];

            int valenceNeighbor = vertexValenceBuffer[idx_neighbor * (2*maxValence+1)]; 
            if (valenceNeighbor < 0) {
                boundaryEdgeNeighbors[currNeighbor++] = idx_neighbor;
                if (currNeighbor == 1)    {
                    ibefore = i;
                    zerothNeighbor = i;
                } else {
                    if (i-ibefore == 1) {
                        int tmp = boundaryEdgeNeighbors[0];
                        boundaryEdgeNeighbors[0] = boundaryEdgeNeighbors[1];
                        boundaryEdgeNeighbors[1] = tmp;
                        zerothNeighbor = i;
                    } 
                }
            }

            float const * neighbor   = inOffset + idx_neighbor   * inDesc.stride;
            float const * diagonal   = inOffset + idx_diagonal   * inDesc.stride;
            float const * neighbor_p = inOffset + idx_neighbor_p * inDesc.stride;
            float const * neighbor_m = inOffset + idx_neighbor_m * inDesc.stride;
            float const * diagonal_m = inOffset + idx_diagonal_m * inDesc.stride;

            float *fp = f+i*length;

            for (int k=0; k<length; ++k) {
                fp[k] = (pos[k]*float(ivalence) + (neighbor_p[k]+neighbor[k])*2.0f + diagonal[k])/(float(ivalence)+5.0f);

                opos[vofs+k] += fp[k];
                rp[i*length+k] =(neighbor_p[k]-neighbor_m[k])/3.0f + (diagonal[k]-diagonal_m[k])/6.0f;
            }
        }
        
        for (int k=0; k<length; ++k) {
            opos[vofs+k] /= ivalence;
        }

        zerothNeighbors[vid] = zerothNeighbor;

        if (currNeighbor == 1) {
            boundaryEdgeNeighbors[1] = boundaryEdgeNeighbors[0];
        }

        for (int i=0; i<ivalence; ++i) {
            unsigned int im = (i+ivalence-1)%ivalence;
            for (int k=0; k<length; ++k) {
                float e = 0.5f*(f[i*length+k]+f[im*length+k]);
                e0[vofs+k] += csf(ivalence-3, 2*i  ) * e;
                e1[vofs+k] += csf(ivalence-3, 2*i+1) * e;
            }
        }

        for (int k=0; k<length; ++k) {
            e0[vofs+k] *= ef_small[ivalence-3];
            e1[vofs+k] *= ef_small[ivalence-3];
        }

        if (valence<0) {
            if (ivalence>2) {
                for (int k=0; k<length; ++k) {
                    opos[vofs+k] = (inOffset[boundaryEdgeNeighbors[0]*inDesc.stride+k] + 
                                    inOffset[boundaryEdgeNeighbors[1]*inDesc.stride+k] + 4.0f*pos[k])/6.0f;
                }
            } else {
                memcpy(opos, pos, length*sizeof(float));
            }

            float k = float(float(ivalence) - 1.0f);    //k is the number of faces
            float c = cosf(float(M_PI)/k);
            float s = sinf(float(M_PI)/k);
            float gamma = -(4.0f*s)/(3.0f*k+c);
            float alpha_0k = -((1.0f+2.0f*c)*sqrtf(1.0f+c))/((3.0f*k+c)*sqrtf(1.0f-c));
            float beta_0 = s/(3.0f*k + c);

            int idx_diagonal = valenceTable[2*zerothNeighbor + 1 + 1];
            assert(idx_diagonal>0);
            float const * diagonal = inOffset + idx_diagonal * inDesc.stride;

            for (int k=0; k<length; ++k) {
                e0[vofs+k] = (inOffset[boundaryEdgeNeighbors[0]*inDesc.stride+k] - 
                              inOffset[boundaryEdgeNeighbors[1]*inDesc.stride+k])/6.0f;

                e1[vofs+k] = gamma * pos[k] + beta_0 * diagonal[k] +                    
                            (inOffset[boundaryEdgeNeighbors[0]*inDesc.stride+k] +
                             inOffset[boundaryEdgeNeighbors[1]*inDesc.stride+k]) * alpha_0k;

            }

            for (int x=1; x<ivalence-1; ++x) {
                unsigned int curri = ((x + zerothNeighbor)%ivalence);
                float alpha = (4.0f*sinf((float(M_PI) * float(x))/k))/(3.0f*k+c);
                float beta = (sinf((float(M_PI) * float(x))/k) + sinf((float(M_PI) * float(x+1))/k))/(3.0f*k+c);

                int idx_neighbor = valenceTable[2*curri + 0 + 1],
                    idx_diagonal = valenceTable[2*curri + 1 + 1];
                assert( idx_neighbor>0 and idx_diagonal>0 );
                
                float const * neighbor = inOffset + idx_neighbor * inDesc.stride,
                            * diagonal = inOffset + idx_diagonal * inDesc.stride;

                for (int k=0; k<length; ++k) {
                    e1[vofs+k] += alpha*neighbor[k] + beta*diagonal[k];
                }
            }

            for (int k=0; k<length; ++k) {
                e1[vofs+k] /= 3.0f;
            }
        }
    }

    // tess control

    // Control Vertices based on : 
    // "Approximating Subdivision Surfaces with Gregory Patches for Hardware Tessellation" 
    // Loop, Schaefer, Ni, Castafio (ACM ToG Siggraph Asia 2009)
    //
    //  P3         e3-      e2+         E2
    //     O--------O--------O--------O
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |        O        O        |
    // e3+ O------O            O------O e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- O------O            O------O e1+
    //     |        O        O        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------O--------O--------O
    //  P0         e0+      e1-         E1
    //

    float *Ep=(float*)alloca(length*4*sizeof(float)), 
          *Em=(float*)alloca(length*4*sizeof(float)), 
          *Fp=(float*)alloca(length*4*sizeof(float)), 
          *Fm=(float*)alloca(length*4*sizeof(float));

    for (int vid=0; vid<4; ++vid) {

        unsigned int ip = (vid+1)%4,
                     im = (vid+3)%4,
                     n = abs(valences[vid]),
                     ivalence = n;

        const unsigned int *quadOffsets = quadOffsetBuffer;

        int vofs = vid * length;

        unsigned int   start =  quadOffsets[vid] & 0x00ff,
                        prev = (quadOffsets[vid] & 0xff00) / 256,
                          np = abs(valences[ip]),
                          nm = abs(valences[im]),
                     start_m =  quadOffsets[im] & 0x00ff,
                      prev_p = (quadOffsets[ip] & 0xff00) / 256;

        float *Em_ip=(float*)alloca(length*sizeof(float)), 
              *Ep_im=(float*)alloca(length*sizeof(float));

        if (valences[ip]<-2) {
            unsigned int j = (np + prev_p - zerothNeighbors[ip]) % np;
            for (int k=0, ipofs=ip*length; k<length; ++k, ++ipofs) {
                Em_ip[k] = opos[ipofs] + cosf((float(M_PI)*j)/float(np-1))*e0[ipofs] + sinf((float(M_PI)*j)/float(np-1))*e1[ipofs];
            }
        } else {
            for (int k=0, ipofs=ip*length; k<length; ++k, ++ipofs) {
                Em_ip[k] = opos[ipofs] + e0[ipofs]*csf(np-3,2*prev_p)  + e1[ipofs]*csf(np-3,2*prev_p+1);
            }
        } 

        if (valences[im]<-2) {
            unsigned int j = (nm + start_m - zerothNeighbors[im]) % nm;
            for (int k=0, imofs=im*length; k<length; ++k, ++imofs) {
                Ep_im[k] = opos[imofs] + cosf((float(M_PI)*j)/float(nm-1))*e0[imofs] + sinf((float(M_PI)*j)/float(nm-1))*e1[imofs];
            }
        } else {
            for (int k=0, imofs=im*length; k<length; ++k, ++imofs) {
                Ep_im[k] = opos[imofs] + e0[imofs]*csf(nm-3,2*start_m) + e1[imofs]*csf(nm-3,2*start_m+1);
            }
        }

        if (valences[vid] < 0) {
            n = (n-1)*2;
        }
        if (valences[im] < 0) {
            nm = (nm-1)*2;
        }  
        if (valences[ip] < 0) {
            np = (np-1)*2;
        }

        rp=r+vid*maxValence*length;

        if (valences[vid] > 2) {
           float s1 = 3.0f - 2.0f*csf(n-3,2)-csf(np-3,2),
                 s2 = 2.0f*csf(n-3,2),
                 s3 = 3.0f -2.0f*cosf(2.0f*float(M_PI)/float(n)) - cosf(2.0f*float(M_PI)/float(nm));

            for (int k=0, ofs=vofs; k<length; ++k, ++ofs) {
                Ep[ofs] = opos[ofs] + e0[ofs] * csf(n-3, 2*start) + e1[ofs]*csf(n-3, 2*start +1);
                Em[ofs] = opos[ofs] + e0[ofs] * csf(n-3, 2*prev ) + e1[ofs]*csf(n-3, 2*prev + 1);
                Fp[ofs] = (csf(np-3,2)*opos[ofs] + s1*Ep[ofs] + s2*Em_ip[k] + rp[start*length+k])/3.0f;
                Fm[ofs] = (csf(nm-3,2)*opos[ofs] + s3*Em[ofs] + s2*Ep_im[k] - rp[prev*length+k])/3.0f;
            }
        } else if (valences[vid] < -2) {
            unsigned int jp = (ivalence + start - zerothNeighbors[vid]) % ivalence,
                         jm = (ivalence + prev  - zerothNeighbors[vid]) % ivalence;

            float s1 = 3-2*csf(n-3,2)-csf(np-3,2),
                  s2 = 2*csf(n-3,2),
                  s3 = 3.0f-2.0f*cosf(2.0f*float(M_PI)/n)-cosf(2.0f*float(M_PI)/nm);

            for (int k=0, ofs=vofs; k<length; ++k, ++ofs) {
                Ep[ofs] = opos[ofs] + cosf((float(M_PI)*jp)/float(ivalence-1))*e0[ofs] + sinf((float(M_PI)*jp)/float(ivalence-1))*e1[ofs];
                Em[ofs] = opos[ofs] + cosf((float(M_PI)*jm)/float(ivalence-1))*e0[ofs] + sinf((float(M_PI)*jm)/float(ivalence-1))*e1[ofs];
                Fp[ofs] = (csf(np-3,2)*opos[ofs] + s1*Ep[ofs] + s2*Em_ip[k] + rp[start*length+k])/3.0f;
                Fm[ofs] = (csf(nm-3,2)*opos[ofs] + s3*Em[ofs] + s2*Ep_im[k] - rp[prev*length+k])/3.0f;
            }

            if (valences[im]<0) {
                float s1=3-2*csf(n-3,2)-csf(np-3,2);
                for (int k=0, ofs=vofs; k<length; ++k, ++ofs) {
                    Fp[ofs] = Fm[ofs] = (csf(np-3,2)*opos[ofs] + s1*Ep[ofs] + s2*Em_ip[k] + rp[start*length+k])/3.0f;
                }
            } else if (valences[ip]<0) {
                float s1 = 3.0f-2.0f*cosf(2.0f*float(M_PI)/n)-cosf(2.0f*float(M_PI)/nm);
                for (int k=0, ofs=vofs; k<length; ++k, ++ofs) {
                    Fm[ofs] = Fp[ofs] = (csf(nm-3,2)*opos[ofs] + s1*Em[ofs] + s2*Ep_im[k] - rp[prev*length+k])/3.0f;
                }
            }
        } else if (valences[vid]==-2) {
            for (int k=0, ofs=vofs, ipofs=ip*length, imofs=im*length; k<length; ++k, ++ofs, ++ipofs, ++imofs) {
                Ep[ofs] = (2.0f * org[ofs] + org[ipofs])/3.0f;
                Em[ofs] = (2.0f * org[ofs] + org[imofs])/3.0f;
                Fp[ofs] = Fm[ofs] = (4.0f * org[ofs] + org[((vid+2)%n)*inDesc.stride+k] + 2.0f * org[ipofs] + 2.0f * org[imofs])/9.0f;
            }
        }
    }

    float * p[20];    
    for (int vid=0, ofs=0; vid<4; ++vid, ofs+=length) {    
        p[vid*5+0] = opos + ofs;
        p[vid*5+1] =   Ep + ofs;
        p[vid*5+2] =   Em + ofs;
        p[vid*5+3] =   Fp + ofs;
        p[vid*5+4] =   Fm + ofs;
    }

    float U = 1-u, V=1-v;
    float d11 = u+v; if(u+v==0.0f) d11 = 1.0f;
    float d12 = U+v; if(U+v==0.0f) d12 = 1.0f;
    float d21 = u+V; if(u+V==0.0f) d21 = 1.0f;
    float d22 = U+V; if(U+V==0.0f) d22 = 1.0f;

    float *q=(float*)alloca(length*16*sizeof(float));
    for (int k=0; k<length; ++k) {
        q[ 5*length+k] = (u*p[ 3][k] + v*p[ 4][k])/d11;
        q[ 6*length+k] = (U*p[ 9][k] + v*p[ 8][k])/d12;
        q[ 9*length+k] = (u*p[19][k] + V*p[18][k])/d21;
        q[10*length+k] = (U*p[13][k] + V*p[14][k])/d22;        
    }

    memcpy(q+ 0*length, p[ 0], length*sizeof(float));
    memcpy(q+ 1*length, p[ 1], length*sizeof(float));
    memcpy(q+ 2*length, p[ 7], length*sizeof(float));
    memcpy(q+ 3*length, p[ 5], length*sizeof(float));
    memcpy(q+ 4*length, p[ 2], length*sizeof(float));
    memcpy(q+ 7*length, p[ 6], length*sizeof(float));
    memcpy(q+ 8*length, p[16], length*sizeof(float));
    memcpy(q+11*length, p[12], length*sizeof(float));
    memcpy(q+12*length, p[15], length*sizeof(float));
    memcpy(q+13*length, p[17], length*sizeof(float));
    memcpy(q+14*length, p[11], length*sizeof(float));
    memcpy(q+15*length, p[10], length*sizeof(float));

    float B[4], D[4], 
          *BU=(float*)alloca(inDesc.length*4*sizeof(float)), 
          *DU=(float*)alloca(inDesc.length*4*sizeof(float));
    memset(BU, 0, inDesc.length*4*sizeof(float));
    memset(DU, 0, inDesc.length*4*sizeof(float));

    univar4x4(u, B, evalDeriv ? D : 0);

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
        
            float const * in = q + (i+j*4)*length;
            
            for (int k=0; k<inDesc.length; ++k) {
            
                BU[i*inDesc.length+k] += in[k] * B[j];
                
                if (evalDeriv)
                    DU[i*inDesc.length+k] += in[k] * D[j];                
            }
        }
    }

    univar4x4(v, B, evalDeriv ? D : 0);

    float * Q = outQ + outDesc.offset;
    float * dQU = outDQU + outDesc.offset;
    float * dQV = outDQV + outDesc.offset;

    // clear result 
    memset(Q, 0, outDesc.length*sizeof(float));
    if (evalDeriv) {
        memset(dQU, 0, outDesc.length*sizeof(float));
        memset(dQV, 0, outDesc.length*sizeof(float));
    }

    for (int i=0; i<4; ++i) {
        for (int k=0; k<inDesc.length; ++k) {
            Q[k] += BU[inDesc.length*i+k] * B[i];
            
            if (evalDeriv) {
                dQU[k] += DU[inDesc.length*i+k] * B[i];
                dQV[k] += BU[inDesc.length*i+k] * D[i];
            }
        }
    }
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
