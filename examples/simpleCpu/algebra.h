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
#pragma once

#include <limits>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>

//
// A few basic linear algebra operations
//

//
// Make the given matrix an identity matrix
//
inline void setIdentity(float* m)
{
    m[0] = m[5] = m[10] = m[15] = 1.0f;
    m[1] = m[2] = m[3] = m[4] = m[6] = m[7] = m[8] = m[9] = m[11] = m[12] = m[13] = m[14] = 0.0f;
}

//
// Multiply A * B and store the result in D
//
inline void
multMatrix(float *d, const float *a, const float *b) 
{
    for (int i=0; i<4; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            d[i*4 + j] =
                a[i*4 + 0] * b[0*4 + j] +
                a[i*4 + 1] * b[1*4 + j] +
                a[i*4 + 2] * b[2*4 + j] +
                a[i*4 + 3] * b[3*4 + j];
        }
    }
}

//
// Create a perspective projection matrix 
//
void setPersp( float fov, float aspect, float znear, float zfar, float* m )
{
    float xymax = znear * tanf(fov * 3.141592653589793238462f / 360.f);
    float ymin = -xymax;
    float xmin = -xymax;

    float width = xymax - xmin;
    float height = xymax - ymin;

    float depth = zfar - znear;
    float q = -(zfar + znear) / depth;
    float qn = -2 * (zfar * znear) / depth;

    float w = 2 * znear / width;
    w = w / aspect;
    float h = 2 * znear / height;

    m[0]  = w;
    m[1]  = 0.f;
    m[2]  = 0.f;
    m[3]  = 0.f;

    m[4]  = 0.f;
    m[5]  = h;
    m[6]  = 0.f;
    m[7]  = 0.f;

    m[8]  = 0.f;
    m[9]  = 0.f;
    m[10] = q;
    m[11] = -1;

    m[12] = 0.f;
    m[13] = 0.f;
    m[14] = qn;
    m[15] = 0.f;   
}

//
// Apply a translation to the given matrix m
//
void 
translateMatrix(float x, float y, float z, float* m)
{
    m[0] += m[3]*x;   m[4] += m[7]*x;   m[8] += m[11]*x;   m[12] += m[15]*x;
    m[1] += m[3]*y;   m[5] += m[7]*y;   m[9] += m[11]*y;   m[13] += m[15]*y;
    m[2] += m[3]*z;   m[6] += m[7]*z;   m[10]+= m[11]*z;   m[14] += m[15]*z;
}

//
// Apply a rotation to the given matrix m
//
void 
rotateMatrix(float angle, float x, float y, float z, float* m)
{
    float rads = float((2*3.14159 / 360.) * angle);
    float c = cosf(rads);  
    float s = sinf(rads); 
    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;

    float m2[16];
    m2[0] = xx * (1 - c) + c;
    m2[4] = xy * (1 - c) - z * s;
    m2[8] = xz * (1 - c) + y * s;
    m2[12] = 0;

    m2[1] = xy * (1 - c) + z * s;
    m2[5] = yy * (1 - c) + c;
    m2[9] = yz * (1 - c) - x * s;
    m2[13] = 0;
    
    m2[2] = xz * (1 - c) - y * s;
    m2[6] = yz * (1 - c) + x * s;
    m2[10]= zz * (1 - c) + c;
    m2[14]= 0;

    m2[3]= 0;
    m2[7]= 0;
    m2[11]= 0;
    m2[15]= 1;

    float mOrig[16];
    for (int i = 0; i < 16; i++)
        mOrig[i] = m[i];

    multMatrix(m, mOrig, m2);
}

//
// Print out the matrix (as usual, column-major order is assumed)
//
inline void printMatrix(float* m) 
{
    for (int r = 0; r < 4; r++) {
        std::cout << " ";
        for (int c = 0; c < 4; c++) { 
            std::cout << std::setprecision(3) << m[c*4 + r];
            if (c != 3)
                std::cout << ",";
            else
                std::cout << std::endl;
        }
    }
}

// 
// Perform a cross-product of three points to calculate a face normal 
//
inline void
cross(float *n, const float *p0, const float *p1, const float *p2) 
{
    float a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
    float b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
    n[0] = a[1]*b[2]-a[2]*b[1];
    n[1] = a[2]*b[0]-a[0]*b[2];
    n[2] = a[0]*b[1]-a[1]*b[0];

    float rn = 1.0f/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] *= rn;
    n[1] *= rn;
    n[2] *= rn;
}

//
// Normalize the given vector 
//
inline void
normalize(float * p) 
{
    float dist = sqrtf( p[0]*p[0] + p[1]*p[1]  + p[2]*p[2] );
    p[0]/=dist;
    p[1]/=dist;
    p[2]/=dist;
}

//
// Compute the center of the list of points and the size of the bound
//
inline void
computeCenterAndSize(const std::vector<float>& positions, float* center, float* size) 
{
    float fmax = std::numeric_limits<float>().max(),
          fmin = std::numeric_limits<float>().min();
    float min[3] = { fmax, fmax, fmax};
    float max[3] = { fmin, fmin, fmin};
    for (size_t i=0; i < positions.size()/3; ++i) {
        for(int j=0; j<3; ++j) {
            float v = positions[i*3+j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }
    for (int j=0; j<3; ++j) {
        center[j] = (min[j] + max[j]) * 0.5f;
        *size += (max[j]-min[j])*(max[j]-min[j]);
    }
    *size = sqrtf(*size);
}

