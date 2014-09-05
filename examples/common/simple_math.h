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

#ifndef SIMPLE_MATH_H
#define SIMPLE_MATH_H

#include <cmath>

inline void
cross(float *n, const float *v1, const float *v2) {
    n[0] = v1[1]*v2[2]-v1[2]*v2[1];
    n[1] = v1[2]*v2[0]-v1[0]*v2[2];
    n[2] = v1[0]*v2[1]-v1[1]*v2[0];
}

inline void
cross(float *n, const float *p0, const float *p1, const float *p2) {

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

inline void
normalize(float * p) {

    float dist = sqrtf( p[0]*p[0] + p[1]*p[1]  + p[2]*p[2] );
    p[0]/=dist;
    p[1]/=dist;
    p[2]/=dist;
}

inline void
multMatrix(float *d, const float *a, const float *b) {

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

inline void
inverseMatrix(float *d, const float *m) {

    d[0] = m[ 5]*m[10]*m[15] - m[ 5]*m[11]*m[14] -
           m[ 9]*m[ 6]*m[15] + m[ 9]*m[ 7]*m[14] +
           m[13]*m[ 6]*m[11] - m[13]*m[ 7]*m[10];

    d[1] = -m[ 1]*m[10]*m[15] + m[ 1]*m[11]*m[14] +
            m[ 9]*m[ 2]*m[15] - m[ 9]*m[ 3]*m[14] -
            m[13]*m[ 2]*m[11] + m[13]*m[ 3]*m[10];

    d[2] = m[ 1]*m[ 6]*m[15] - m[ 1]*m[ 7]*m[14] -
           m[ 5]*m[ 2]*m[15] + m[ 5]*m[ 3]*m[14] +
           m[13]*m[ 2]*m[ 7] - m[13]*m[ 3]*m[ 6];

    d[3] = -m[ 1]*m[ 6]*m[11] + m[ 1]*m[ 7]*m[10] +
            m[ 5]*m[ 2]*m[11] - m[ 5]*m[ 3]*m[10] -
            m[ 9]*m[ 2]*m[ 7] + m[ 9]*m[ 3]*m[ 6];

    d[4] = -m[ 4]*m[10]*m[15] + m[ 4]*m[11]*m[14] +
            m[ 8]*m[ 6]*m[15] - m[ 8]*m[ 7]*m[14] -
            m[12]*m[ 6]*m[11] + m[12]*m[ 7]*m[10];

    d[5] = m[ 0]*m[10]*m[15] - m[ 0]*m[11]*m[14] -
           m[ 8]*m[ 2]*m[15] + m[ 8]*m[ 3]*m[14] +
           m[12]*m[ 2]*m[11] - m[12]*m[ 3]*m[10];

    d[6] = -m[ 0]*m[ 6]*m[15] + m[ 0]*m[ 7]*m[14] +
            m[ 4]*m[ 2]*m[15] - m[ 4]*m[ 3]*m[14] -
            m[12]*m[ 2]*m[ 7] + m[12]*m[ 3]*m[ 6];

    d[7] = m[ 0]*m[ 6]*m[11] - m[ 0]*m[ 7]*m[10] -
           m[ 4]*m[ 2]*m[11] + m[ 4]*m[ 3]*m[10] +
           m[ 8]*m[ 2]*m[ 7] - m[ 8]*m[ 3]*m[ 6];

    d[8] = m[ 4]*m[ 9]*m[15] - m[ 4]*m[11]*m[13] -
           m[ 8]*m[ 5]*m[15] + m[ 8]*m[ 7]*m[13] +
           m[12]*m[ 5]*m[11] - m[12]*m[ 7]*m[ 9];

    d[9] = -m[ 0]*m[ 9]*m[15] + m[ 0]*m[11]*m[13] +
            m[ 8]*m[ 1]*m[15] - m[ 8]*m[ 3]*m[13] -
            m[12]*m[ 1]*m[11] + m[12]*m[ 3]*m[ 9];

    d[10] = m[ 0]*m[ 5]*m[15] - m[ 0]*m[ 7]*m[13] -
            m[ 4]*m[ 1]*m[15] + m[ 4]*m[ 3]*m[13] +
            m[12]*m[ 1]*m[ 7] - m[12]*m[ 3]*m[ 5];

    d[11] = -m[ 0]*m[ 5]*m[11] + m[ 0]*m[ 7]*m[ 9] +
             m[ 4]*m[ 1]*m[11] - m[ 4]*m[ 3]*m[ 9] -
             m[ 8]*m[ 1]*m[ 7] + m[ 8]*m[ 3]*m[ 5];

    d[12] = -m[ 4]*m[ 9]*m[14] + m[ 4]*m[10]*m[13] +
             m[ 8]*m[ 5]*m[14] - m[ 8]*m[ 6]*m[13] -
             m[12]*m[ 5]*m[10] + m[12]*m[ 6]*m[ 9];

    d[13] = m[ 0]*m[ 9]*m[14] - m[ 0]*m[10]*m[13] -
            m[ 8]*m[ 1]*m[14] + m[ 8]*m[ 2]*m[13] +
            m[12]*m[ 1]*m[10] - m[12]*m[ 2]*m[ 9];

    d[14] = -m[ 0]*m[ 5]*m[14] + m[ 0]*m[ 6]*m[13] +
             m[ 4]*m[ 1]*m[14] - m[ 4]*m[ 2]*m[13] -
             m[12]*m[ 1]*m[ 6] + m[12]*m[ 2]*m[ 5];

    d[15] = m[ 0]*m[ 5]*m[10] - m[ 0]*m[ 6]*m[ 9] -
            m[ 4]*m[ 1]*m[10] + m[ 4]*m[ 2]*m[ 9] +
            m[ 8]*m[ 1]*m[ 6] - m[ 8]*m[ 2]*m[ 5];

    float det = m[0] * d[0] + m[1] * d[4] + m[2] * d[8] + m[3] * d[12];

    if (det == 0.0f) return;
    det = 1.0f / det;

    for (int i = 0; i < 16; i++)
        d[i] = d[i] * det;
}

inline void
perspective(float *m, float fovy, float aspect, float znear, float zfar)
{
    float r = 2 * (float)M_PI * fovy / 360.0F;
    float t = 1.0f / tanf(r*0.5f);
    m[0] = t/aspect;
    m[1] = m[2] = m[3] = 0.0;
    m[4] = 0.0;
    m[5] = t;
    m[6] = m[7] = 0.0;
    m[8] = m[9] = 0.0;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1;
    m[12] = m[13] = 0.0;
    m[14] = (2*zfar*znear)/(znear - zfar);
    m[15] = 0.0;
}

inline void
identity(float *m)
{
    m[0] = 1; m[1] = 0; m[2] = 0; m[3] = 0;
    m[4] = 0; m[5] = 1; m[6] = 0; m[7] = 0;
    m[8] = 0; m[9] = 0; m[10] = 1; m[11] = 0;
    m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
}

inline void
translate(float *m, float x, float y, float z)
{
    float t[16];
    identity(t);
    t[12] = x;
    t[13] = y;
    t[14] = z;
    float o[16];
    for(int i = 0; i < 16; i++) o[i] = m[i];
    multMatrix(m, t, o);
}

inline void
ortho(float *m, float left, float top, float right, float bottom)
{
    identity(m);
    m[0] = 2.0f / (right - left);
    m[5] = 2.0f / (top - bottom);
    m[10] = -1;
    m[12] = -(right+left)/(right-left);
    m[13] = -(top+bottom)/(top-bottom);
}

inline void
rotate(float *m, float angle, float x, float y, float z)
{
    float r = 2 * (float) M_PI * angle/360.0f;
    float c = cosf(r);
    float s = sinf(r);
    float t[16];
    t[0] = x*x*(1-c)+c;
    t[1] = y*x*(1-c)+z*s;
    t[2] = x*z*(1-c)-y*s;
    t[3] = 0;
    t[4] = x*y*(1-c)-z*s;
    t[5] = y*y*(1-c)+c;
    t[6] = y*z*(1-c)+x*s;
    t[7] = 0;
    t[8] = x*z*(1-c)+y*s;
    t[9] = y*z*(1-c)-x*s;
    t[10] = z*z*(1-c)+c;
    t[11] = 0;
    t[12] = t[13] = t[14] = 0;
    t[15] = 1;
    float o[16];
    for(int i = 0; i < 16; i++) o[i] = m[i];
    multMatrix(m, t, o);
}

inline void
scale(float *m, float sx, float sy, float sz)
{
    float t[16];
    identity(t);
    t[0] = sx;
    t[5] = sy;
    t[10] = sz;
    float o[16];
    for(int i = 0; i < 16; i++) o[i] = m[i];
    multMatrix(m, t, o);
}

inline void
transpose(float *m)
{
    std::swap(m[1], m[4]);
    std::swap(m[2], m[8]);
    std::swap(m[3], m[12]);
    std::swap(m[6], m[9]);
    std::swap(m[7], m[13]);
    std::swap(m[11],m[14]);
}

inline void
apply(float *v, const float *m)
{
    float r[4];
    r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + v[3] * m[12];
    r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + v[3] * m[13];
    r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + v[3] * m[14];
    r[3] = v[0] * m[3] + v[1] * m[7] + v[2] * m[11] + v[3] * m[15];
    v[0] = r[0];
    v[1] = r[1];
    v[2] = r[2];
    v[3] = r[3];
}

inline void
pickMatrix(float *m, float x, float y, float width, float height, const int *viewport)
{
    float sx, sy;
    float tx, ty;

    sx = viewport[2] / width;
    sy = viewport[3] / height;
    tx = (viewport[2] + 2.0f * (viewport[0] - x)) / width;
    ty = (viewport[3] + 2.0f * (viewport[1] - y)) / height;

    identity(m);
    m[0] = sx;
    m[5] = sy;
    m[12] = tx;
    m[13] = ty;
}

#endif // SIMPLE_MATH_H
