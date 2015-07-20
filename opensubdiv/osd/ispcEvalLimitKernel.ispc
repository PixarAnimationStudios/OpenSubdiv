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


#define MAX_CHANNEL 4

struct BufferDescriptor {
    int offset;  // offset to desired element data
    int length;  // number or length of the data
    int stride;  // stride to the next element    
};

struct Point {
    float x;
    float y;
    float z;
};

inline struct Point operator+(struct Point a, struct Point b) {
    struct Point result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;        

    return result;
}

inline uniform struct Point operator+(uniform struct Point a, uniform struct Point b) {
    uniform struct Point result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;        

    return result;
}

inline struct Point operator-(struct Point a, struct Point b) {
    struct Point result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;        

    return result;
}

inline uniform struct Point operator-(uniform struct Point a, uniform struct Point b) {
    uniform struct Point result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;        

    return result;
}

inline struct Point operator*(struct Point a, float b) {
    struct Point result;
    result.x = a.x * b;
    result.y = a.y * b;
    result.z = a.z * b;        

    return result;
}

inline uniform struct Point operator*(uniform struct Point a, uniform float b) {
    uniform struct Point result;
    result.x = a.x * b;
    result.y = a.y * b;
    result.z = a.z * b;        

    return result;
} 

inline struct Point operator*(float b, struct Point a) {
    struct Point result;
    result.x = b * a.x;
    result.y = b * a.y;
    result.z = b * a.z;        

    return result;
}

inline uniform struct Point operator*(uniform float b, uniform struct Point a) {
    uniform struct Point result;
    result.x = b * a.x;
    result.y = b * a.y;
    result.z = b * a.z;        

    return result;
}

inline struct Point operator/(struct Point a, float b) {
    struct Point result;
    result.x = a.x / b;
    result.y = a.y / b;
    result.z = a.z / b;        

    return result;
}

inline uniform struct Point operator/(uniform struct Point a, uniform float b) {
    uniform struct Point result;
    result.x = a.x / b;
    result.y = a.y / b;
    result.z = a.z / b;        

    return result;
}

inline void cross(struct Point &a, struct Point &b, struct Point &c)
{
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;
}

inline uniform bool
nonQuadRoot(uniform unsigned int bitField) 
{
    return (bitField >> 3) & 0x1;
}

inline uniform unsigned int getU(uniform unsigned int bitField) 
{ 
    return (uniform unsigned int)((bitField >> 22) & 0x3ff); 
}

inline uniform unsigned int getV(uniform unsigned int bitField) 
{ 
    return (uniform unsigned int)((bitField >> 12) & 0x3ff); 
}

inline uniform unsigned int getBoundary(uniform unsigned int bitField)
{ 
    return (uniform unsigned int)((bitField >> 8) & 0xf); 
}
     
inline uniform unsigned int getDepth(uniform unsigned int bitField)
{ 
    return  (uniform unsigned int)(bitField & 0xf); 
}

inline uniform float
getParamFraction(uniform unsigned int bitField){
    if (nonQuadRoot(bitField)) {
        return 1.0f / (1 << (getDepth(bitField)-1));
    } else {
        return 1.0f / (1 << getDepth(bitField));
    }
}

inline void 
adjustBoundaryWeights(uniform unsigned int bitField,
                      float                sWeights[4], 
                      float                tWeights[4]) {

    uniform int boundary = getBoundary(bitField);

    if (boundary & 1) {
        tWeights[2] -= tWeights[0];
        tWeights[1] += 2*tWeights[0];
        tWeights[0] = 0;
    }
    if (boundary & 2) {
        sWeights[1] -= sWeights[3];
        sWeights[2] += 2*sWeights[3];
        sWeights[3] = 0;
    }
    if (boundary & 4) {
        tWeights[1] -= tWeights[3];
        tWeights[2] += 2*tWeights[3];
        tWeights[3] = 0;
    }
    if (boundary & 8) {
        sWeights[2] -= sWeights[0];
        sWeights[1] += 2*sWeights[0];
        sWeights[0] = 0;
    }
}

inline void
getBSplineWeights(float t, float point[4], float deriv[4]) {
    // The four uniform cubic B-Spline basis functions evaluated at t:
    float const one6th = 1.0f / 6.0f;

    float t2 = t * t;
    float t3 = t * t2;

    point[0] = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    point[1] = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    point[2] = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    point[3] = one6th * (                                 t3);

    // Derivatives of the above four basis functions at t:
    deriv[0] = -0.5f*t2 +      t - 0.5f;
    deriv[1] =  1.5f*t2 - 2.0f*t;
    deriv[2] = -1.5f*t2 +      t + 0.5f;
    deriv[3] =  0.5f*t2;
}

inline void
getBezierWeights(float t, float point[4], float deriv[4]) {
    // The four uniform cubic Bezier basis functions (in terms of t and its
    // complement tC) evaluated at t:
    float t2 = t*t;
    float tC = 1.0f - t;
    float tC2 = tC * tC;

    point[0] = tC2 * tC;
    point[1] = tC2 * t * 3.0f;
    point[2] = t2 * tC * 3.0f;
    point[3] = t2 * t;

    // Derivatives of the above four basis functions at t:
    deriv[0] = -3.0f * tC2;
    deriv[1] =  9.0f * t2 - 12.0f * t + 3.0f;
    deriv[2] = -9.0f * t2 +  6.0f * t;
    deriv[3] =  3.0f * t2;
}

inline void
getBSplineWeightsNoDerivative(float t, float point[4]) {
    // The four uniform cubic B-Spline basis functions evaluated at t:
    float const one6th = 1.0f / 6.0f;

    float t2 = t * t;
    float t3 = t * t2;

    point[0] = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    point[1] = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    point[2] = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    point[3] = one6th * (                                 t3);
}

inline void
getBezierWeightsNoDerivative(float t, float point[4]) {
    // The four uniform cubic Bezier basis functions (in terms of t and its
    // complement tC) evaluated at t:
    float t2 = t*t;
    float tC = 1.0f - t;
    float tC2 = tC * tC;

    point[0] = tC2 * tC;
    point[1] = tC2 * t * 3.0f;
    point[2] = t2 * tC * 3.0f;
    point[3] = t2 * t;
}

export void
evalBilinear(uniform unsigned int                  bitField,
             uniform int                           nPoint, 
             uniform const float  * uniform        u, 
             uniform const float  * uniform        v,             
             uniform const int    * uniform        vertexIndices,
             uniform const BufferDescriptor       &inDesc,
             uniform const float * uniform         inQ,
             uniform const BufferDescriptor       &outDesc,
             uniform float *uniform                outQ,
             uniform const BufferDescriptor       &duDesc,            
             uniform float *uniform                outDQU,
             uniform const BufferDescriptor       &dvDesc,            
             uniform float *uniform                outDQV)
{
    uniform int nChannel = inDesc.length / 3;
    assert(nChannel < MAX_CHANNEL);
    
    uniform Point controlVertices[MAX_CHANNEL*4];
    for(uniform int i=0; i<4; i++) {
        uniform unsigned int id = vertexIndices[i];
        uniform const float * uniform pVertex = inQ + inDesc.offset + id * inDesc.stride;
        for(uniform int c=0; c<nChannel; c++) {
            uniform int offset = c * 4 + i;
            controlVertices[offset].x = pVertex[0];
            controlVertices[offset].y = pVertex[1];
            controlVertices[offset].z = pVertex[2];
            pVertex += 3;
        }
    }        
                 
    foreach( n = 0 ... nPoint) {        
        float ou   = 1.0f - u[n];
        float ov   = 1.0f - v[n];
        float w[4] = { ov*ou, v[n]*ou, v[n]*u[n], ov*u[n] };
                
        float *pOutQ   = outQ   + outDesc.offset + n * outDesc.stride;        
        for(uniform int c=0; c<nChannel; c++) { 
            Point Q;
            Q.x = Q.y = Q.z = 0.0;
            for (uniform int i=0; i<4; ++i) {
                Q = Q + w[i] * controlVertices[c * 4 + i];              
            }    
              
            *pOutQ ++ = Q.x, *pOutQ ++ = Q.y, *pOutQ ++ = Q.z;
        }        
    }
    
    uniform Point dU[MAX_CHANNEL], dV[MAX_CHANNEL];
    for(uniform int c=0; c<nChannel; c++) { 
        dU[c] = 0.5 * (controlVertices[c * 4 + 3] - controlVertices[c * 4 + 0] +
                       controlVertices[c * 4 + 2] - controlVertices[c * 4 + 1]  );
                       
        dV[c] = 0.5 * (controlVertices[c * 4 + 1] - controlVertices[c * 4 + 0] +
                       controlVertices[c * 4 + 2] - controlVertices[c * 4 + 3]  );                       
    }    
    
    foreach( n = 0 ... nPoint) {
        float *pOutDQU = outDQU +  duDesc.offset  + n *  duDesc.stride;     
        float *pOutDQV = outDQV +  dvDesc.offset  + n *  dvDesc.stride;           
        for(uniform int c=0; c<nChannel; c++) { 
            *pOutDQU ++ = dU[c].x, *pOutDQU ++ = dU[c].y, *pOutDQU ++ = dU[c].z;
            *pOutDQV ++ = dV[c].x, *pOutDQV ++ = dV[c].y, *pOutDQV ++ = dV[c].z;            
        }
    }    
}   

export void
evalBilinearNoDerivative(uniform unsigned int                  bitField,
                         uniform int                           nPoint, 
                         uniform const float  * uniform        u, 
                         uniform const float  * uniform        v,             
                         uniform const int    * uniform        vertexIndices,
                         uniform const BufferDescriptor       &inDesc,
                         uniform const float * uniform         inQ,
                         uniform const BufferDescriptor       &outDesc,
                         uniform float *uniform                outQ)
{
    uniform int nChannel = inDesc.length / 3;
    assert(nChannel < MAX_CHANNEL);
    
    uniform Point controlVertices[MAX_CHANNEL*4];
    for(uniform int i=0; i<4; i++) {
        uniform unsigned int id = vertexIndices[i];
        uniform const float * uniform pVertex = inQ + inDesc.offset + id * inDesc.stride;
        for(uniform int c=0; c<nChannel; c++) {
            uniform int offset = c * 4 + i;
            controlVertices[offset].x = pVertex[0];
            controlVertices[offset].y = pVertex[1];
            controlVertices[offset].z = pVertex[2];
            pVertex += 3;
        }
    }        
                 
    foreach( n = 0 ... nPoint) {        
        float ou   = 1.0f - u[n];
        float ov   = 1.0f - v[n];
        float w[4] = { ov*ou, v[n]*ou, v[n]*u[n], ov*u[n] };
                
        float *pOutQ   = outQ   + outDesc.offset + n * outDesc.stride;        
        for(uniform int c=0; c<nChannel; c++) { 
            Point Q;
            Q.x = Q.y = Q.z = 0.0;
            for (uniform int i=0; i<4; ++i) {
                Q = Q + w[i] * controlVertices[c * 4 + i];              
            }    
              
            *pOutQ ++ = Q.x, *pOutQ ++ = Q.y, *pOutQ ++ = Q.z;
        }        
    }
}   

export void
evalBSpline(uniform unsigned int                  bitField,
            uniform int                           nPoint, 
            uniform const float  * uniform        u, 
            uniform const float  * uniform        v,             
            uniform const int    * uniform        vertexIndices,
            uniform const BufferDescriptor       &inDesc,
            uniform const float * uniform         inQ,
            uniform const BufferDescriptor       &outDesc,
            uniform float *uniform                outQ,
            uniform const BufferDescriptor       &duDesc,            
            uniform float *uniform                outDQU,
            uniform const BufferDescriptor       &dvDesc,            
            uniform float *uniform                outDQV)
{
    uniform int nChannel = inDesc.length / 3;
    assert(nChannel < MAX_CHANNEL);
    
    uniform Point controlVertices[MAX_CHANNEL*16];
    for(uniform int i=0; i<16; i++) {
        uniform unsigned int id = vertexIndices[i];
        uniform const float * uniform pVertex = inQ + inDesc.offset + id * inDesc.stride;
        for(uniform int c=0; c<nChannel; c++) {
            uniform int offset = c * 16 + i;
            controlVertices[offset].x = pVertex[0];
            controlVertices[offset].y = pVertex[1];
            controlVertices[offset].z = pVertex[2];
            pVertex += 3;
        }
    }

    uniform float dScale = (uniform float)(1 << getDepth(bitField));
    
    uniform float frac = getParamFraction(bitField);

    // top left corner
    uniform float pu = (uniform float)getU(bitField)*frac;
    uniform float pv = (uniform float)getV(bitField)*frac;

    foreach( n = 0 ... nPoint) {
        // normalize u,v coordinates
        float s = (u[n] - pu) / frac;
        float t = (v[n] - pv) / frac;
        
        float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4];
       
        getBSplineWeights(s, sWeights, dsWeights);
        getBSplineWeights(t, tWeights, dtWeights);      
        
        adjustBoundaryWeights(bitField,  sWeights,  tWeights);
        adjustBoundaryWeights(bitField, dsWeights, dtWeights);            
        
        float weight[16];       
        for (uniform int i = 0; i < 4; ++i) {
            for (uniform int j = 0; j < 4; ++j) {
                weight[4*i+j] = sWeights[j] * tWeights[i];
            }
        }
                       
        float *pOutQ = outQ + outDesc.offset + n * outDesc.stride;             
        for(uniform int c=0; c<nChannel; c++) { 
            uniform int offset = c * 16;
            Point Q;
            Q.x = Q.y = Q.z = 0.0;
            for (uniform int i=0; i<16; ++i) {
                Q = Q + weight[i] * controlVertices[offset + i];                            
            }    
              
            *pOutQ ++ = Q.x, *pOutQ ++ = Q.y, *pOutQ ++ = Q.z;
        }   
        
        float derivS[16], derivT[16];       
        for (uniform int i = 0; i < 4; ++i) {
            for (uniform int j = 0; j < 4; ++j) {
                derivS[4*i+j] = dsWeights[j] *  tWeights[i] * dScale;
                derivT[4*i+j] =  sWeights[j] * dtWeights[i] * dScale;                
            }
        }
                       
        float *pOutDQU = outDQU + duDesc.offset + n * duDesc.stride;
        float *pOutDQV = outDQV + dvDesc.offset + n * dvDesc.stride;                                  
        for(uniform int c=0; c<nChannel; c++) { 
            uniform int offset = c * 16;
            Point DQU, DQV;
            DQU.x = DQU.y = DQU.z = 0.0;
            DQV.x = DQV.y = DQV.z = 0.0;            
            for (uniform int i=0; i<16; ++i) {
                DQU = DQU + derivS[i] * controlVertices[offset + i];
                DQV = DQV + derivT[i] * controlVertices[offset + i];                                            
            }    
              
            *pOutDQU ++ = DQU.x, *pOutDQU ++ = DQU.y, *pOutDQU ++ = DQU.z;
            *pOutDQV ++ = DQV.x, *pOutDQV ++ = DQV.y, *pOutDQV ++ = DQV.z;            
        }                   
    }
}  

export void
evalBSplineNoDerivative(uniform unsigned int                  bitField,
                        uniform int                           nPoint, 
                        uniform const float  * uniform        u, 
                        uniform const float  * uniform        v,             
                        uniform const int    * uniform        vertexIndices,
                        uniform const BufferDescriptor       &inDesc,
                        uniform const float * uniform         inQ,
                        uniform const BufferDescriptor       &outDesc,
                        uniform float *uniform                outQ)
{
    uniform int nChannel = inDesc.length / 3;
    assert(nChannel < MAX_CHANNEL);
    
    uniform Point controlVertices[MAX_CHANNEL*16];
    for(uniform int i=0; i<16; i++) {
        uniform unsigned int id = vertexIndices[i];
        uniform const float * uniform pVertex = inQ + inDesc.offset + id * inDesc.stride;
        for(uniform int c=0; c<nChannel; c++) {
            uniform int offset = c * 16 + i;
            controlVertices[offset].x = pVertex[0];
            controlVertices[offset].y = pVertex[1];
            controlVertices[offset].z = pVertex[2];
            pVertex += 3;
        }
    }

    uniform float frac = getParamFraction(bitField);

    // top left corner
    uniform float pu = (uniform float)getU(bitField)*frac;
    uniform float pv = (uniform float)getV(bitField)*frac;

    foreach( n = 0 ... nPoint) {
        // normalize u,v coordinates
        float s = (u[n] - pu) / frac;
        float t = (v[n] - pv) / frac;
        
        float sWeights[4], tWeights[4];
       
        getBSplineWeightsNoDerivative(s, sWeights);
        getBSplineWeightsNoDerivative(t, tWeights);      
        
        adjustBoundaryWeights(bitField, sWeights, tWeights);  
        
        float weight[16];       
        for (uniform int i = 0; i < 4; ++i) {
            for (uniform int j = 0; j < 4; ++j) {
                weight[4*i+j] = sWeights[j] * tWeights[i];
            }
        }
                       
        float *pOutQ = outQ + outDesc.offset + n * outDesc.stride;             
        for(uniform int c=0; c<nChannel; c++) { 
            uniform int offset = c * 16;
            Point Q;
            Q.x = Q.y = Q.z = 0.0;
            for (uniform int i=0; i<16; ++i) {
                Q = Q + weight[i] * controlVertices[offset + i];                            
            }    
              
            *pOutQ ++ = Q.x, *pOutQ ++ = Q.y, *pOutQ ++ = Q.z;
        }           
    }
}  

void getGregoryWeights(uniform unsigned int bitField, 
                       float s, float t, float point[20], float deriv1[20], float deriv2[20]) {
    //
    //  P3         e3-      e2+         P2
    //     15------17-------11--------10
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |       19       13        |
    // e3+ 16-----18           14-----12 e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- 2------4            8------6 e1+
    //     |        3        9        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------1--------7--------5
    //  P0         e0+      e1-         P1
    //

    //  Indices of boundary and interior points and their corresponding Bezier points
    //  (this can be reduced with more direct indexing and unrolling of loops):
    //
    static uniform int const boundaryGregory[12] = { 0, 1, 7, 5, 2, 6, 16, 12, 15, 17, 11, 10 };
    static uniform int const boundaryBezSCol[12] = { 0, 1, 2, 3, 0, 3,  0,  3,  0,  1,  2,  3 };
    static uniform int const boundaryBezTRow[12] = { 0, 0, 0, 0, 1, 1,  2,  2,  3,  3,  3,  3 };

    static uniform int const interiorGregory[8] = { 3, 4,  8, 9,  13, 14,  18, 19 };
    static uniform int const interiorBezSCol[8] = { 1, 1,  2, 2,   2,  2,   1,  1 };
    static uniform int const interiorBezTRow[8] = { 1, 1,  1, 1,   2,  2,   2,  2 };

    //
    //  Bezier basis functions are denoted with B while the rational multipliers for the
    //  interior points will be denoted G -- so we have B(s), B(t) and G(s,t):
    //
    //  Directional Bezier basis functions B at s and t:
    float Bs[4], Bds[4];
    float Bt[4], Bdt[4];

    getBezierWeights(s, Bs, Bds);
    getBezierWeights(t, Bt, Bdt);

    //  Rational multipliers G at s and t:
    float sC = 1.0f - s;
    float tC = 1.0f - t;

    //  Use <= here to avoid compiler warnings -- the sums should always be non-negative:
    float df0 = s  + t;   df0 = (df0 <= 0.0f) ? 1.0f : (1.0f / df0);
    float df1 = sC + t;   df1 = (df1 <= 0.0f) ? 1.0f : (1.0f / df1);
    float df2 = sC + tC;  df2 = (df2 <= 0.0f) ? 1.0f : (1.0f / df2);
    float df3 = s  + tC;  df3 = (df3 <= 0.0f) ? 1.0f : (1.0f / df3);

    float G[8] = { s*df0, t*df0,  t*df1, sC*df1,  sC*df2, tC*df2,  tC*df3, s*df3 };

    //  Combined weights for boundary and interior points:
    for (uniform int i = 0; i < 12; ++i) {
        point[boundaryGregory[i]] = Bs[boundaryBezSCol[i]] * Bt[boundaryBezTRow[i]];
    }
    for (uniform int i = 0; i < 8; ++i) {
        point[interiorGregory[i]] = Bs[interiorBezSCol[i]] * Bt[interiorBezTRow[i]] * G[i];
    }

    //
    //  For derivatives, the basis functions for the interior points are rational and ideally
    //  require appropriate differentiation, i.e. product rule for the combination of B and G
    //  and the quotient rule for the rational G itself.  As initially proposed by Loop et al
    //  though, the approximation using the 16 Bezier points arising from the G(s,t) has
    //  proved adequate (and is what the GPU shaders use) so we continue to use that here.
    //
    //  An implementation of the true derivatives is provided for future reference -- it is
    //  unclear if the approximations will hold up under surface analysis involving higher
    //  order differentiation.
    //

    //  Remember to include derivative scaling in all assignments below:
    uniform float dScale = (uniform float)(1 << getDepth(bitField));

    //  Combined weights for boundary points -- simple (scaled) tensor products:
    for (uniform int i = 0; i < 12; ++i) {
        uniform int iDst = boundaryGregory[i];
        uniform int tRow = boundaryBezTRow[i];
        uniform int sCol = boundaryBezSCol[i];

        deriv1[iDst] = Bds[sCol] * Bt[tRow] * dScale;
        deriv2[iDst] = Bdt[tRow] * Bs[sCol] * dScale;
    }

#define _USE_BEZIER_PSEUDO_DERIVATIVES
#ifdef _USE_BEZIER_PSEUDO_DERIVATIVES
    //  Approximation to the true Gregory derivatives by differentiating the Bezier patch
    //  unique to the given (s,t), i.e. having F = (g^+ * f^+) + (g^- * f^-) as its four
    //  interior points:
    //
    //  Combined weights for interior points -- (scaled) tensor products with G+ or G-:
    for (uniform int i = 0; i < 8; ++i) {
        uniform int iDst = interiorGregory[i];
        uniform int tRow = interiorBezTRow[i];
        uniform int sCol = interiorBezSCol[i];
        deriv1[iDst] = Bds[sCol] * Bt[tRow] * G[i] * dScale;
        deriv2[iDst] = Bdt[tRow] * Bs[sCol] * G[i] * dScale;
    }
#else
    //  True Gregory derivatives using appropriate differentiation of composite functions:
    //
    //  Note that for G(s,t) = N(s,t) / D(s,t), all N' and D' are trivial constants (which
    //  simplifies things for higher order derivatives).  And while each pair of functions
    //  G (i.e. the G+ and G- corresponding to points f+ and f-) must sum to 1 to ensure
    //  Bezier equivalence (when f+ = f-), the pairs of G' must similarly sum to 0.  So we
    //  can potentially compute only one of the pair and negate the result for the other
    //  (and with 4 or 8 computations involving these constants, this is all very SIMD
    //  friendly...) but for now we treat all 8 independently for simplicity.
    //
    //float N[8] = {   s,     t,      t,     sC,      sC,     tC,      tC,     s };
    uniform float D[8] = {   df0,   df0,    df1,    df1,     df2,    df2,     df3,   df3 };

    static uniform float const Nds[8] = { 1.0f, 0.0f,  0.0f, -1.0f, -1.0f,  0.0f,  0.0f,  1.0f };
    static uniform float const Ndt[8] = { 0.0f, 1.0f,  1.0f,  0.0f,  0.0f, -1.0f, -1.0f,  0.0f };

    static uniform float const Dds[8] = { 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f };
    static uniform float const Ddt[8] = { 1.0f, 1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f };

    //  Combined weights for interior points -- (scaled) combinations of B, B', G and G':
    for (uniform int i = 0; i < 8; ++i) {
        uniform int iDst = interiorGregory[i];
        uniform int tRow = interiorBezTRow[i];
        uniform int sCol = interiorBezSCol[i];

        //  Quotient rule for G' (re-expressed in terms of G to simplify (and D = 1/D)):
        float Gds = (Nds[i] - Dds[i] * G[i]) * D[i];
        float Gdt = (Ndt[i] - Ddt[i] * G[i]) * D[i];

        //  Product rule combining B and B' with G and G' (and scaled):
        deriv1[iDst] = (Bds[sCol] * G[i] + Bs[sCol] * Gds) * Bt[tRow] * dScale;
        deriv2[iDst] = (Bdt[tRow] * G[i] + Bt[tRow] * Gdt) * Bs[sCol] * dScale;
    }
#endif
}

void getGregoryWeightsNoDerivative(uniform unsigned int bitField, float s, float t, float point[20]) {
    //
    //  P3         e3-      e2+         P2
    //     15------17-------11--------10
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |       19       13        |
    // e3+ 16-----18           14-----12 e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- 2------4            8------6 e1+
    //     |        3        9        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------1--------7--------5
    //  P0         e0+      e1-         P1
    //

    //  Indices of boundary and interior points and their corresponding Bezier points
    //  (this can be reduced with more direct indexing and unrolling of loops):
    //
    static uniform int const boundaryGregory[12] = { 0, 1, 7, 5, 2, 6, 16, 12, 15, 17, 11, 10 };
    static uniform int const boundaryBezSCol[12] = { 0, 1, 2, 3, 0, 3,  0,  3,  0,  1,  2,  3 };
    static uniform int const boundaryBezTRow[12] = { 0, 0, 0, 0, 1, 1,  2,  2,  3,  3,  3,  3 };

    static uniform int const interiorGregory[8] = { 3, 4,  8, 9,  13, 14,  18, 19 };
    static uniform int const interiorBezSCol[8] = { 1, 1,  2, 2,   2,  2,   1,  1 };
    static uniform int const interiorBezTRow[8] = { 1, 1,  1, 1,   2,  2,   2,  2 };

    //
    //  Bezier basis functions are denoted with B while the rational multipliers for the
    //  interior points will be denoted G -- so we have B(s), B(t) and G(s,t):
    //
    //  Directional Bezier basis functions B at s and t:
    float Bs[4];
    float Bt[4];

    getBezierWeightsNoDerivative(s, Bs);
    getBezierWeightsNoDerivative(t, Bt);

    //  Rational multipliers G at s and t:
    float sC = 1.0f - s;
    float tC = 1.0f - t;

    //  Use <= here to avoid compiler warnings -- the sums should always be non-negative:
    float df0 = s  + t;   df0 = (df0 <= 0.0f) ? 1.0f : (1.0f / df0);
    float df1 = sC + t;   df1 = (df1 <= 0.0f) ? 1.0f : (1.0f / df1);
    float df2 = sC + tC;  df2 = (df2 <= 0.0f) ? 1.0f : (1.0f / df2);
    float df3 = s  + tC;  df3 = (df3 <= 0.0f) ? 1.0f : (1.0f / df3);

    float G[8] = { s*df0, t*df0,  t*df1, sC*df1,  sC*df2, tC*df2,  tC*df3, s*df3 };

    //  Combined weights for boundary and interior points:
    for (uniform int i = 0; i < 12; ++i) {
        point[boundaryGregory[i]] = Bs[boundaryBezSCol[i]] * Bt[boundaryBezTRow[i]];
    }
    for (uniform int i = 0; i < 8; ++i) {
        point[interiorGregory[i]] = Bs[interiorBezSCol[i]] * Bt[interiorBezTRow[i]] * G[i];
    }
}

export void
evalGregory(uniform   unsigned int            bitField,
            uniform   int                     nPoint, 
            uniform   float                   u[], 
            uniform   float                   v[],                    
            uniform   const unsigned int      vertexIndices[],
            uniform   const BufferDescriptor &inDesc,
            uniform   const float             inQ[], 
            uniform   const BufferDescriptor &outDesc,
            uniform   float                   outQ[], 
            uniform   const BufferDescriptor &duDesc,
            uniform   float                   outDQU[],
            uniform   const BufferDescriptor &dvDesc,            
            uniform   float                   outDQV[])
{
    uniform int nChannel = inDesc.length / 3;
    assert(nChannel < MAX_CHANNEL);
    
    uniform Point controlVertices[MAX_CHANNEL*20];
    for(uniform int i=0; i<20; i++) {
        uniform unsigned int id = vertexIndices[i];
        uniform const float * uniform pVertex = inQ + inDesc.offset + id * inDesc.stride;
        for(uniform int c=0; c<nChannel; c++) {
            uniform int offset = c * 20 + i;
            controlVertices[offset].x = pVertex[0];
            controlVertices[offset].y = pVertex[1];
            controlVertices[offset].z = pVertex[2];
            pVertex += 3;
        }
    }

    uniform float frac = getParamFraction(bitField);

    // top left corner
    uniform float pu = (uniform float)getU(bitField)*frac;
    uniform float pv = (uniform float)getV(bitField)*frac;

    foreach( n = 0 ... nPoint) {
        // normalize u,v coordinates
        float s = (u[n] - pu) / frac;
        float t = (v[n] - pv) / frac;
        
        float point[20], deriv1[20], deriv2[20];
        getGregoryWeights(bitField, s, t, point, deriv1, deriv2);
        
        float *pOutQ = outQ + outDesc.offset + n * outDesc.stride;             
        for(uniform int c=0; c<nChannel; c++) { 
            uniform int offset = c * 16;
            Point Q;
            Q.x = Q.y = Q.z = 0.0;
            for (uniform int i=0; i<16; ++i) {
                Q = Q + point[i] * controlVertices[offset + i];                            
            }    
              
            *pOutQ ++ = Q.x, *pOutQ ++ = Q.y, *pOutQ ++ = Q.z;
        }   
                              
        float *pOutDQU = outDQU + duDesc.offset + n * duDesc.stride;
        float *pOutDQV = outDQV + dvDesc.offset + n * dvDesc.stride;                                  
        for(uniform int c=0; c<nChannel; c++) { 
            uniform int offset = c * 20;
            Point DQU, DQV;
            DQU.x = DQU.y = DQU.z = 0.0;
            DQV.x = DQV.y = DQV.z = 0.0;            
            for (uniform int i=0; i<20; ++i) {
                DQU = DQU + deriv1[i] * controlVertices[offset + i];
                DQV = DQV + deriv2[i] * controlVertices[offset + i];                                            
            }    
              
            *pOutDQU ++ = DQU.x, *pOutDQU ++ = DQU.y, *pOutDQU ++ = DQU.z;
            *pOutDQV ++ = DQV.x, *pOutDQV ++ = DQV.y, *pOutDQV ++ = DQV.z;            
        }                           
    }
}   

export void
evalGregoryNoDerivative(uniform unsigned int            bitField,
                        uniform int                     nPoint, 
                        uniform float                   u[], 
                        uniform float                   v[],                    
                        uniform const unsigned int      vertexIndices[],
                        uniform const BufferDescriptor &inDesc,
                        uniform const float             inQ[], 
                        uniform const BufferDescriptor &outDesc,
                        uniform float                   outQ[]
                       )
{
    uniform int nChannel = inDesc.length / 3;
    assert(nChannel < MAX_CHANNEL);
    
    uniform Point controlVertices[MAX_CHANNEL*20];
    for(uniform int i=0; i<20; i++) {
        uniform unsigned int id = vertexIndices[i];
        uniform const float * uniform pVertex = inQ + inDesc.offset + id * inDesc.stride;
        for(uniform int c=0; c<nChannel; c++) {
            uniform int offset = c * 20 + i;
            controlVertices[offset].x = pVertex[0];
            controlVertices[offset].y = pVertex[1];
            controlVertices[offset].z = pVertex[2];
            pVertex += 3;
        }
    }

    uniform float frac = getParamFraction(bitField);

    // top left corner
    uniform float pu = (uniform float)getU(bitField)*frac;
    uniform float pv = (uniform float)getV(bitField)*frac;

    foreach( n = 0 ... nPoint) {
        // normalize u,v coordinates
        float s = (u[n] - pu) / frac;
        float t = (v[n] - pv) / frac;
        
        float point[20];
        getGregoryWeightsNoDerivative(bitField, s, t, point);
        
        float *pOutQ = outQ + outDesc.offset + n * outDesc.stride;             
        for(uniform int c=0; c<nChannel; c++) { 
            uniform int offset = c * 20;
            Point Q;
            Q.x = Q.y = Q.z = 0.0;
            for (uniform int i=0; i<20; ++i) {
                Q = Q + point[i] * controlVertices[offset + i];                            
            }    
              
            *pOutQ ++ = Q.x, *pOutQ ++ = Q.y, *pOutQ ++ = Q.z;
        }   
    }
}    

