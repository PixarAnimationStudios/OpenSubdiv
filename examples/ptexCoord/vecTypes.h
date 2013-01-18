#ifndef _VEC_TYPES_H_
#define _VEC_TYPES_H_

#include <math.h>

//--------------------------------------------------------------------------------
// vec2 
//--------------------------------------------------------------------------------

struct vec2
{
    float x, y;

	vec2() {}

	explicit vec2( float a )
    {
        x = a;
        y = a;
    }

	explicit vec2( float a, float b )
    {
        x = a;
        y = b;
    }

	explicit vec2( float const * const v )
    {
        x = v[0];
        y = v[1];
    }

          float & operator [](int i) { return ((float*)this)[i]; }
    const float & operator [](int i) const { return ((float*)this)[i]; }
	
	vec2 & operator =( vec2  const & v ) { x = v.x;  y = v.y;  return *this; }
	vec2 & operator+=( float const & s ) { x += s;   y += s;   return *this; }
	vec2 & operator+=( vec2  const & v ) { x += v.x; y += v.y; return *this; }
	vec2 & operator-=( float const & s ) { x -= s;   y -= s;   return *this; }
	vec2 & operator-=( vec2  const & v ) { x -= v.x; y -= v.y; return *this; }
	vec2 & operator*=( float const & s ) { x *= s;   y *= s;   return *this; }
	vec2 & operator*=( vec2  const & v ) { x *= v.x; y *= v.y; return *this; }
	vec2 & operator/=( float const & s ) { x /= s;   y /= s;   return *this; }
	vec2 & operator/=( vec2  const & v ) { x /= v.x; y /= v.y; return *this; }

    vec2 xx() const { return vec2(x,x); }
    vec2 xy() const { return vec2(x,y); }
    vec2 yx() const { return vec2(y,x); }
    vec2 yy() const { return vec2(y,y); }
};

inline vec2 operator+( vec2  const & v, float const & s ) { return vec2( v.x + s, v.y + s ); }
inline vec2 operator+( float const & s, vec2  const & v ) { return vec2( s + v.x, s + v.y ); }
inline vec2 operator+( vec2  const & a, vec2  const & b ) { return vec2( a.x + b.x, a.y + b.y ); }
inline vec2 operator-( vec2  const & v, float const & s ) { return vec2( v.x - s, v.y - s ); }
inline vec2 operator-( float const & s, vec2  const & v ) { return vec2( s - v.x, s - v.y ); }
inline vec2 operator-( vec2  const & a, vec2  const & b ) { return vec2( a.x - b.x, a.y - b.y ); }
inline vec2 operator*( vec2  const & v, float const & s ) { return vec2( v.x * s, v.y * s ); }
inline vec2 operator*( float const & s, vec2  const & v ) { return vec2( s * v.x, s * v.y ); }
inline vec2 operator*( vec2  const & a, vec2  const & b ) { return vec2( a.x * b.x, a.y * b.y ); }
inline vec2 operator/( vec2  const & v, float const & s ) { return vec2( v.x / s, v.y / s ); }
inline vec2 operator/( float const & s, vec2  const & v ) { return vec2( s / v.x, s / v.y ); }
inline vec2 operator/( vec2  const & a, vec2  const & b ) { return vec2( a.x / b.x, a.y / b.y ); }

inline vec2 normalize( vec2 const & v )
{
    float const m2 = v.x*v.x + v.y*v.y;
    float const im = 1.0f/sqrtf( m2 );
    return vec2( v.x*im, v.y*im );
}

inline vec2 mix( vec2 const & a, vec2 const & b, float const f )
{
    return vec2( a.x*(1.0f-f) + f*b.x, 
                 a.y*(1.0f-f) + f*b.y ); 
}

inline float length( vec2 const & v )
{
    return sqrtf( v.x*v.x + v.y*v.y );
}

inline float dot( vec2 const & a, vec2 const & b )
{
    return a.x*b.x + a.y*b.y;
}

inline float distance( vec2 const & a, vec2 const & b )
{
    return length( a - b );
}


//--------------------------------------------------------------------------------
// vec3 
//--------------------------------------------------------------------------------

struct vec3
{
    float x, y, z;

	vec3() {}

	explicit vec3( float a )
    {
        x = a;
        y = a;
        z = a;
    }

	explicit vec3( float a, float b, float c )
    {
        x = a;
        y = b;
        z = c;
    }

	explicit vec3( float const * const v )
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

	explicit vec3( vec2 const & v, float s )
    {
        x = v.x;
        y = v.y;
        z = s;
    }

	explicit vec3( float s, vec2 const & v )
    {
        x = s;
        y = v.x;
        z = v.y;
    }


          float & operator [](int i) { return ((float*)this)[i]; }
    const float & operator [](int i) const { return ((float*)this)[i]; }

	vec3 & operator =( vec3  const & v ) { x = v.x;  y = v.y;  z = v.z;  return *this; }
	vec3 & operator+=( float const & s ) { x += s;   y += s;   z += s;   return *this; }
	vec3 & operator+=( vec3  const & v ) { x += v.x; y += v.y; z += v.z; return *this; }
	vec3 & operator-=( float const & s ) { x -= s;   y -= s;   z -= s;   return *this; }
	vec3 & operator-=( vec3  const & v ) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	vec3 & operator*=( float const & s ) { x *= s;   y *= s;   z *= s;   return *this; }
	vec3 & operator*=( vec3  const & v ) { x *= v.x; y *= v.y; z *= v.z; return *this; }
	vec3 & operator/=( float const & s ) { x /= s;   y /= s;   z /= s;   return *this; }
	vec3 & operator/=( vec3  const & v ) { x /= v.x; y /= v.y; z /= v.z; return *this; }

    vec2 xx() const { return vec2(x,x); }
    vec2 xy() const { return vec2(x,y); }
    vec2 xz() const { return vec2(x,z); }
    vec2 yx() const { return vec2(y,x); }
    vec2 yy() const { return vec2(y,y); }
    vec2 yz() const { return vec2(y,z); }
    vec2 zx() const { return vec2(z,x); }
    vec2 zy() const { return vec2(z,y); }
    vec2 zz() const { return vec2(z,z); }

    vec3 xxx() const { return vec3(x,x,x); }
    vec3 xxy() const { return vec3(x,x,y); }
    vec3 xxz() const { return vec3(x,x,z); }
    vec3 xyx() const { return vec3(x,y,x); }
    vec3 xyy() const { return vec3(x,y,y); }
    vec3 xyz() const { return vec3(x,y,z); }
    vec3 xzx() const { return vec3(x,z,x); }
    vec3 xzy() const { return vec3(x,z,y); }
    vec3 xzz() const { return vec3(x,z,z); }
    vec3 yxx() const { return vec3(y,x,x); }
    vec3 yxy() const { return vec3(y,x,y); }
    vec3 yxz() const { return vec3(y,x,z); }
    vec3 yyx() const { return vec3(y,y,x); }
    vec3 yyy() const { return vec3(y,y,y); }
    vec3 yyz() const { return vec3(y,y,z); }
    vec3 yzx() const { return vec3(y,z,x); }
    vec3 yzy() const { return vec3(y,z,y); }
    vec3 yzz() const { return vec3(y,z,z); }
    vec3 zxx() const { return vec3(z,x,x); }
    vec3 zxy() const { return vec3(z,x,y); }
    vec3 zxz() const { return vec3(z,x,z); }
    vec3 zyx() const { return vec3(z,y,x); }
    vec3 zyy() const { return vec3(z,y,y); }
    vec3 zyz() const { return vec3(z,y,z); }
    vec3 zzx() const { return vec3(z,z,x); }
    vec3 zzy() const { return vec3(z,z,y); }
    vec3 zzz() const { return vec3(z,z,z); }
};

inline vec3 operator+( vec3  const & v, float const & s ) { return vec3( v.x + s, v.y + s, v.z + s ); }
inline vec3 operator+( float const & s, vec3  const & v ) { return vec3( s + v.x, s + v.y, s + v.z ); }
inline vec3 operator+( vec3  const & a, vec3  const & b ) { return vec3( a.x + b.x, a.y + b.y, a.z + b.z ); }
inline vec3 operator-( vec3  const & v, float const & s ) { return vec3( v.x - s, v.y - s, v.z - s); }
inline vec3 operator-( float const & s, vec3  const & v ) { return vec3( s - v.x, s - v.y, s - v.z ); }
inline vec3 operator-( vec3  const & a, vec3  const & b ) { return vec3( a.x - b.x, a.y - b.y, a.z - b.z ); }
inline vec3 operator*( vec3  const & v, float const & s ) { return vec3( v.x * s, v.y * s, v.z * s ); }
inline vec3 operator*( float const & s, vec3  const & v ) { return vec3( s * v.x, s * v.y, s * v.z ); }
inline vec3 operator*( vec3  const & a, vec3  const & b ) { return vec3( a.x * b.x, a.y * b.y, a.z * b.z ); }
inline vec3 operator/( vec3  const & v, float const & s ) { return vec3( v.x / s, v.y / s, v.z / s ); }
inline vec3 operator/( float const & s, vec3  const & v ) { return vec3( s / v.x, s / v.y, s / v.z); }
inline vec3 operator/( vec3  const & a, vec3  const & b ) { return vec3( a.x / b.x, a.y / b.y, a.z / b.z ); }

inline vec3 normalize( vec3 const & v )
{
    float const m2 = v.x*v.x + v.y*v.y + v.z*v.z;
    float const im = 1.0f/sqrtf( m2 );
    return vec3( v.x*im, v.y*im, v.z*im );
}

inline vec3 normalizeSafe( vec3 const & v )
{
    float const m2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if( m2<0.00001f ) return vec3(0.0f);
    float const im = 1.0f/sqrtf( m2 );
    return vec3( v.x*im, v.y*im, v.z*im );
}

inline vec3 mix( vec3 const & a, vec3 const & b, float const f )
{
    return vec3( a.x*(1.0f-f) + f*b.x, 
                 a.y*(1.0f-f) + f*b.y, 
                 a.z*(1.0f-f) + f*b.z );
}

inline vec3 cross( vec3 const & a, vec3 const & b )
{
    return vec3( a.y*b.z - a.z*b.y,
                 a.z*b.x - a.x*b.z,
                 a.x*b.y - a.y*b.x );
}

inline float length( vec3 const & v )
{
    return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z );
}

inline float dot( vec3 const & a, vec3 const & b )
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float distance( vec3 const & a, vec3 const & b )
{
    return length( a - b );
}

inline void buildBase( const vec3 & n, vec3 & uu, vec3 & vv )
{
    vec3 up;
    if( fabsf(n.z)<0.9f ) { up.x=0.0f; up.y=0.0f; up.z=1.0f; }
    else                  { up.x=1.0f; up.y=0.0f; up.z=0.0f; }
    uu = normalize( cross( n, up ) );
    vv = normalize( cross( uu, n ) );
}


inline vec3 orientate( const vec3 & v, const vec3 & dir )
{
    vec3 res = v;
    const float kk = dot( dir, v );
    if( kk<0.0f ) res -= 2.0f*dir*kk;
    return res;

}

//--------------------------------------------------------------------------------
// vec4 
//--------------------------------------------------------------------------------

struct vec4
{
    float x, y, z, w;

	vec4() {}

	explicit vec4( float a, float b, float c, float d )
    {
        x = a;
        y = b;
        z = c;
        w = d;
    }

	explicit vec4( float const * const v )
    {
        x = v[0];
        y = v[1];
        z = v[2];
        w = v[3];
    }

          float & operator [](int i) { return ((float*)this)[i]; }
    const float & operator [](int i) const { return ((float*)this)[i]; }

	vec4 & operator =( vec4  const & v ) { x = v.x;  y = v.y;  z = v.z;  w = v.w;  return *this; }
	vec4 & operator+=( float const & s ) { x += s;   y += s;   z += s;   w += s;   return *this; }
	vec4 & operator+=( vec4  const & v ) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
	vec4 & operator-=( float const & s ) { x -= s;   y -= s;   z -= s;   w -= s;   return *this; }
	vec4 & operator-=( vec4  const & v ) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
	vec4 & operator*=( float const & s ) { x *= s;   y *= s;   z *= s;   w *= s;   return *this; }
	vec4 & operator*=( vec4  const & v ) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
	vec4 & operator/=( float const & s ) { x /= s;   y /= s;   z /= s;   w /= s;   return *this; }
	vec4 & operator/=( vec4  const & v ) { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *this; }

    vec2 xx() const { return vec2(x,x); }
    vec2 xy() const { return vec2(x,y); }
    vec2 xz() const { return vec2(x,z); }
    vec2 yx() const { return vec2(y,x); }
    vec2 yy() const { return vec2(y,y); }
    vec2 yz() const { return vec2(y,z); }
    vec2 zx() const { return vec2(z,x); }
    vec2 zy() const { return vec2(z,y); }
    vec2 zz() const { return vec2(z,z); }

    vec3 xxx() const { return vec3(x,x,x); }
    vec3 xxy() const { return vec3(x,x,y); }
    vec3 xxz() const { return vec3(x,x,z); }
    vec3 xyx() const { return vec3(x,y,x); }
    vec3 xyy() const { return vec3(x,y,y); }
    vec3 xyz() const { return vec3(x,y,z); }
    vec3 xzx() const { return vec3(x,z,x); }
    vec3 xzy() const { return vec3(x,z,y); }
    vec3 xzz() const { return vec3(x,z,z); }
    vec3 yxx() const { return vec3(y,x,x); }
    vec3 yxy() const { return vec3(y,x,y); }
    vec3 yxz() const { return vec3(y,x,z); }
    vec3 yyx() const { return vec3(y,y,x); }
    vec3 yyy() const { return vec3(y,y,y); }
    vec3 yyz() const { return vec3(y,y,z); }
    vec3 yzx() const { return vec3(y,z,x); }
    vec3 yzy() const { return vec3(y,z,y); }
    vec3 yzz() const { return vec3(y,z,z); }
    vec3 zxx() const { return vec3(z,x,x); }
    vec3 zxy() const { return vec3(z,x,y); }
    vec3 zxz() const { return vec3(z,x,z); }
    vec3 zyx() const { return vec3(z,y,x); }
    vec3 zyy() const { return vec3(z,y,y); }
    vec3 zyz() const { return vec3(z,y,z); }
    vec3 zzx() const { return vec3(z,z,x); }
    vec3 zzy() const { return vec3(z,z,y); }
    vec3 zzz() const { return vec3(z,z,z); }
};

inline vec4 operator+( vec4  const & v, float const & s ) { return vec4( v.x + s, v.y + s, v.z + s, v.w + s ); }
inline vec4 operator+( float const & s, vec4  const & v ) { return vec4( s + v.x, s + v.y, s + v.z, s + v.w ); }
inline vec4 operator+( vec4  const & a, vec4  const & b ) { return vec4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
inline vec4 operator-( vec4  const & v, float const & s ) { return vec4( v.x - s, v.y - s, v.z - s, v.w - s); }
inline vec4 operator-( float const & s, vec4  const & v ) { return vec4( s - v.x, s - v.y, s - v.z, s - v.w ); }
inline vec4 operator-( vec4  const & a, vec4  const & b ) { return vec4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
inline vec4 operator*( vec4  const & v, float const & s ) { return vec4( v.x * s, v.y * s, v.z * s, v.w * s ); }
inline vec4 operator*( float const & s, vec4  const & v ) { return vec4( s * v.x, s * v.y, s * v.z, s * v.w ); }
inline vec4 operator*( vec4  const & a, vec4  const & b ) { return vec4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }
inline vec4 operator/( vec4  const & v, float const & s ) { return vec4( v.x / s, v.y / s, v.z / s, v.w / s ); }
inline vec4 operator/( float const & s, vec4  const & v ) { return vec4( s / v.x, s / v.y, s / v.z, s / v.w); }
inline vec4 operator/( vec4  const & a, vec4  const & b ) { return vec4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w ); }

inline vec4 normalize( vec4 const & v )
{
    float const m2 = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    float const im = 1.0f/sqrtf( m2 );
    return vec4( v.x*im, v.y*im, v.z*im, v.w*im );
}

inline vec4 mix( vec4 const & a, vec4 const & b, float const f )
{
    return vec4( a.x*(1.0f-f) + f*b.x, 
                 a.y*(1.0f-f) + f*b.y, 
                 a.z*(1.0f-f) + f*b.z, 
                 a.w*(1.0f-f) + f*b.w );
}

inline float length( vec4 const & v )
{
    return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w );
}

inline float dot( vec4 const & a, vec4 const & b )
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline float distance( vec4 const & a, vec4 const & b )
{
    return length( a - b );
}

//--------------------------------------------------------------------------------
// mat2x2 
//--------------------------------------------------------------------------------

struct mat2x2
{
    float m[4];

	mat2x2() {}

	explicit mat2x2( float a0, float a1,
	                 float a2, float a3 )
    {
        m[0] = a0; m[1] = a1;
        m[2] = a2; m[3] = a3;
    }

          float & operator [](int i) { return m[i]; }
    const float & operator [](int i) const { return m[i]; }
};

inline vec2 operator*( mat2x2 const & m, vec2 const & v ) 
{ 
    return vec2( v.x*m[0] + v.y*m[1],
                 v.x*m[2] + v.y*m[3] );
}

inline float determinant( mat2x2 const & m )
{
    return m.m[0]*m.m[3] - m.m[1]*m.m[2];
}

//--------------------------------------------------------------------------------
// mat3x3 
//--------------------------------------------------------------------------------

struct mat3x3
{
    float m[9];

	mat3x3() {}

	explicit mat3x3( float a0, float a1, float a2,
	                 float a3, float a4, float a5,
	                 float a6, float a7, float a8 )
    {
        m[0] = a0; m[1] = a1; m[2] = a2;
        m[3] = a3; m[4] = a4; m[5] = a5;
        m[6] = a6; m[7] = a7; m[8] = a8;
    }

          float & operator [](int i) { return m[i]; }
    const float & operator [](int i) const { return m[i]; }
/*
	mat3x3 & operator =( mat3x3 const & v ) { x = v.x;  y = v.y;  z = v.z;  return *this; }
	mat3x3 & operator*=( float  const & s ) { x *= s;   y *= s;   z *= s;   return *this; }
	mat3x3 & operator*=( mat3x3 const & v ) { x *= v.x; y *= v.y; z *= v.z; return *this; }
	mat3x3 & operator/=( float  const & s ) { x /= s;   y /= s;   z /= s;   return *this; }
	mat3x3 & operator/=( mat3x3 const & v ) { x /= v.x; y /= v.y; z /= v.z; return *this; }
*/
};

inline mat3x3 operator*( mat3x3 const & a, mat3x3 const & b ) 
{ 

    return mat3x3( a[0]*b[0] + a[1]*b[3] + a[2]*b[6],
                   a[0]*b[1] + a[1]*b[4] + a[2]*b[7],
                   a[0]*b[2] + a[1]*b[5] + a[2]*b[8],

                   a[3]*b[0] + a[4]*b[3] + a[5]*b[6],
                   a[3]*b[1] + a[4]*b[4] + a[5]*b[7],
                   a[3]*b[2] + a[4]*b[5] + a[5]*b[8],

                   a[6]*b[0] + a[7]*b[3] + a[8]*b[6],
                   a[6]*b[1] + a[7]*b[4] + a[8]*b[7],
                   a[6]*b[2] + a[7]*b[5] + a[8]*b[8] ); 
}

inline vec3 operator*( mat3x3 const & m, vec3 const & v ) 
{ 
    return vec3( v.x*m[0] + v.y*m[1] + v.z*m[2],
                 v.x*m[3] + v.y*m[4] + v.z*m[5],
                 v.x*m[6] + v.y*m[7] + v.z*m[8] );
}

inline float determinant( mat3x3 const & m )
{
    return m.m[0]*m.m[4]*m.m[8] + m.m[3]*m.m[7]*m.m[2] + m.m[1]*m.m[5]*m.m[6] - m.m[2]*m.m[4]*m.m[6] - m.m[1]*m.m[3]*m.m[8] - m.m[5]*m.m[7]*m.m[0];
}


inline mat3x3 transpose( mat3x3 const & m )
{
    return mat3x3( m.m[0], m.m[3], m.m[6], 
                   m.m[1], m.m[4], m.m[7], 
                   m.m[2], m.m[5], m.m[8] );
}

inline vec3 rotate( const vec3 & v, float t, const vec3 & a )
{
    const float sint = sinf(t);
    const float cost = cosf(t);
    const float icost = 1.0f - cost;

    const mat3x3 m = mat3x3( a.x*a.x*icost + cost,
                             a.y*a.x*icost - sint*a.z,
                             a.z*a.x*icost + sint*a.y,

                             a.x*a.y*icost + sint*a.z,
                             a.y*a.y*icost + cost,
                             a.z*a.y*icost - sint*a.x,

                             a.x*a.z*icost - sint*a.y,
                             a.y*a.z*icost + sint*a.x,
                             a.z*a.z*icost + cost );
    return m * v;
}

inline vec3 rotate( const vec3 & v, float cost, float sint, const vec3 & a )
{
    const float icost = 1.0f - cost;

    const mat3x3 m = mat3x3( a.x*a.x*icost + cost,
                             a.y*a.x*icost - sint*a.z,
                             a.z*a.x*icost + sint*a.y,

                             a.x*a.y*icost + sint*a.z,
                             a.y*a.y*icost + cost,
                             a.z*a.y*icost - sint*a.x,

                             a.x*a.z*icost - sint*a.y,
                             a.y*a.z*icost + sint*a.x,
                             a.z*a.z*icost + cost );
    return m * v;
}


inline mat3x3 rotationAxisAngle( const vec3 & a, const float t )
{
    const float sint = sinf(t);
    const float cost = cosf(t);
    const float icost = 1.0f - cost;

    return mat3x3( a.x*a.x*icost + cost,
                   a.y*a.x*icost - sint*a.z,
                   a.z*a.x*icost + sint*a.y,

                   a.x*a.y*icost + sint*a.z,
                   a.y*a.y*icost + cost,
                   a.z*a.y*icost - sint*a.x,

                   a.x*a.z*icost - sint*a.y,
                   a.y*a.z*icost + sint*a.x,
                   a.z*a.z*icost + cost );
}


inline mat3x3 rotationEuler( float x, float y, float z )
{
    const float a = sinf(x); 
    const float b = cosf(x); 
    const float c = sinf(y); 
    const float d = cosf(y); 
    const float e = sinf(z); 
    const float f = cosf(z); 
    const float ac = a*c;
    const float bc = b*c;

    return mat3x3( d*f, d*e, -c,
                   ac*f-b*e, ac*e+b*f, a*d,
                   bc*f+a*e, bc*e-a*f, b*d );

}


/*
inline mat3x3 inverse( mat3x3 const & m )
{
}
*/

//--------------------------------------------------------------------------------
// mat4x4 
//--------------------------------------------------------------------------------

struct mat4x4
{
    float m[16];

	mat4x4() {}

	explicit mat4x4( float a00, float a01, float a02, float a03,
	                 float a04, float a05, float a06, float a07,
	                 float a08, float a09, float a10, float a11,
	                 float a12, float a13, float a14, float a15 )
    {
        m[ 0] = a00; m[ 1] = a01; m[ 2] = a02; m[ 3] = a03;
        m[ 4] = a04; m[ 6] = a05; m[ 6] = a06; m[ 7] = a07;
        m[ 8] = a08; m[ 9] = a09; m[10] = a10; m[11] = a11;
        m[12] = a12; m[13] = a13; m[14] = a14; m[15] = a15;
    }

          float & operator [](int i) { return m[i]; }
    const float & operator [](int i) const { return m[i]; }
/*
	mat4x4 & operator =( mat4x4 const & v ) { x = v.x;  y = v.y;  z = v.z;  return *this; }
	mat4x4 & operator*=( float  const & s ) { x *= s;   y *= s;   z *= s;   return *this; }
	mat4x4 & operator*=( mat4x4 const & v ) { x *= v.x; y *= v.y; z *= v.z; return *this; }
	mat4x4 & operator/=( float  const & s ) { x /= s;   y /= s;   z /= s;   return *this; }
	mat4x4 & operator/=( mat4x4 const & v ) { x /= v.x; y /= v.y; z /= v.z; return *this; }
*/
};

inline mat4x4 operator*( mat4x4 const & a, mat4x4 const & b ) 
{
    mat4x4 res;
    for( int i=0; i<4; i++ )
    {
        const float x = a.m[4*i+0];
        const float y = a.m[4*i+1];
        const float z = a.m[4*i+2];
        const float w = a.m[4*i+3];

        res.m[4*i+0] = x * b[ 0] + y * b[ 4] + z * b[ 8] + w * b[12];
        res.m[4*i+1] = x * b[ 1] + y * b[ 5] + z * b[ 9] + w * b[13];
        res.m[4*i+2] = x * b[ 2] + y * b[ 6] + z * b[10] + w * b[14];
        res.m[4*i+3] = x * b[ 3] + y * b[ 7] + z * b[11] + w * b[15];
    }

    return res;
}


inline vec4 operator*( mat4x4 const & m, vec4 const & v ) 
{ 
    return vec4( v.x*m[ 0] + v.y*m[ 1] + v.z*m[ 2] + v.w*m[ 3],
                 v.x*m[ 4] + v.y*m[ 5] + v.z*m[ 6] + v.w*m[ 7],
                 v.x*m[ 8] + v.y*m[ 9] + v.z*m[10] + v.w*m[11],
                 v.x*m[12] + v.y*m[13] + v.z*m[14] + v.w*m[15] );
}

inline mat4x4 transpose( mat4x4 const & m )
{
    return mat4x4( m.m[0], m.m[4], m.m[ 8], m.m[12],
                   m.m[1], m.m[5], m.m[ 9], m.m[13], 
                   m.m[2], m.m[6], m.m[10], m.m[14], 
                   m.m[3], m.m[7], m.m[11], m.m[15] );
}

inline mat4x4 invert( mat4x4 const & src, int *status=0 )
{
    int   i, j, k, swap;
    float t, temp[4][4];


    for (i=0; i<4; i++)
	for (j=0; j<4; j++)
	    temp[i][j] = src[i*4+j];

    float inv[16];
    for( i=0; i<16; i++ ) inv[i] = 0.0f;
    inv[ 0] = 1.0f;
    inv[ 5] = 1.0f;
    inv[10] = 1.0f;
    inv[15] = 1.0f;

    for( i=0; i<4; i++ ) 
    	{
		// Look for largest element in column
		swap = i;
		for (j = i + 1; j < 4; j++) 
			if (fabsf(temp[j][i]) > fabsf(temp[i][i])) 
				swap = j;

		if (swap != i)
    	    {
			// Swap rows.
			for( k=0; k<4; k++ )
    	    	{
				t = temp[i][k];
				temp[i][k] = temp[swap][k];
				temp[swap][k] = t;

				t = inv[i*4+k];
				inv[i*4+k] = inv[swap*4+k];
				inv[swap*4+k] = t;
	    		}
			}

    	// pivot==0 -> singular matrix!
		if (temp[i][i] == 0)
        {
		    if( status ) status[0] = 0;
            return mat4x4(0.0f,0.0f,0.0f,0.0f,
                          0.0f,0.0f,0.0f,0.0f,
                          0.0f,0.0f,0.0f,0.0f,
                          0.0f,0.0f,0.0f,0.0f);
        }

		t = temp[i][i];
		t = 1.0f/t;
		for (k=0; k<4; k++ )
    			{
				temp[i][k] *= t;
				inv[i*4+k] *= t;
				}

		for( j=0; j<4; j++ )
			{
			if( j != i )
    	   		{
				t = temp[j][i];
				for( k=0; k<4; k++ )
    	    		{
					temp[j][k] -= temp[i][k]*t;
					inv[j*4+k] -= inv[i*4+k]*t;
					}
	    		}
			}
		}

    if( status ) status[0] = 1;

    return mat4x4( inv[ 0], inv[ 1], inv[ 2], inv[ 3],
                   inv[ 4], inv[ 5], inv[ 6], inv[ 7],
                   inv[ 8], inv[ 9], inv[10], inv[11],
                   inv[12], inv[13], inv[14], inv[15] );

}

inline mat4x4 setRotationEuler( float x, float y, float z )
{
    const float a = sinf(x); 
    const float b = cosf(x); 
    const float c = sinf(y); 
    const float d = cosf(y); 
    const float e = sinf(z); 
    const float f = cosf(z); 
    const float ac = a*c;
    const float bc = b*c;

    return mat4x4( d*f,      d*e,       -c,  0.0f, 
                   ac*f-b*e, ac*e+b*f, a*d,  0.0f,
                   bc*f+a*e, bc*e-a*f, b*d,  0.0f,
                   0.0f,     0.0f,     0.0f, 1.0f );
}


inline mat4x4 setTranslation( float x, float y, float z )
{
    return mat4x4( 1.0f, 0.0f, 0.0f, x,
                   0.0f, 1.0f, 0.0f, y,
                   0.0f, 0.0f, 1.0f, z,
                   0.0f, 0.0f, 0.0f, 1.0f );
}

inline mat4x4 setScale( float x, float y, float z )
{
    return mat4x4( x,    0.0f, 0.0f, 0.0f,
                   0.0f, y,    0.0f, 0.0f,
                   0.0f, 0.0f, z,    0.0f,
                   0.0f, 0.0f, 0.0f, 1.0f );
}

//--------------------------------------------------------------------------------
// ivec2 
//--------------------------------------------------------------------------------

struct ivec2
{
    int x, y;

	ivec2() {}

	explicit ivec2( int a, int b )
    {
        x = a;
        y = b;
    }

          int & operator [](int i) { return ((int*)this)[i]; }
    const int & operator [](int i) const { return ((int*)this)[i]; }
};

//--------------------------------------------------------------------------------
// ivec3 
//--------------------------------------------------------------------------------

struct ivec3
{
    int x, y, z;

	ivec3() {}

	explicit ivec3( int a, int b, int c )
    {
        x = a;
        y = b;
        z = c;
    }

          int & operator [](int i) { return ((int*)this)[i]; }
    const int & operator [](int i) const { return ((int*)this)[i]; }
};

//--------------------------------------------------------------------------------
// bound3 
//--------------------------------------------------------------------------------

struct bound3
{
    float mMinX;
    float mMaxX;
    float mMinY;
    float mMaxY;
    float mMinZ;
    float mMaxZ;

	bound3() {}

	explicit bound3( float mix, float max, float miy, float may, float miz, float maz )
    {
        mMinX = mix;
        mMaxX = max;
        mMinY = miy;
        mMaxY = may;
        mMinZ = miz;
        mMaxZ = maz;
    }

	explicit bound3( float const * const v )
    {
        mMinX = v[0];
        mMaxX = v[1];
        mMinY = v[2];
        mMaxY = v[3];
        mMinZ = v[4];
        mMaxZ = v[5];
    }

          float & operator [](int i) { return ((float*)this)[i]; }
    const float & operator [](int i) const { return ((float*)this)[i]; }

};

inline bound3 expand( bound3 const & a, bound3 const & b )
{
    return bound3( a.mMinX+b.mMinX, a.mMaxX+b.mMaxX, 
                   a.mMinY+b.mMinY, a.mMaxY+b.mMaxY, 
                   a.mMinZ+b.mMinZ, a.mMaxZ+b.mMaxZ );
}

inline bound3 expand( bound3 const & a, float const b )
{
    return bound3( a.mMinX-b, a.mMaxX+b, 
                   a.mMinY-b, a.mMaxY+b, 
                   a.mMinZ-b, a.mMaxZ+b );
}

inline bound3 include( bound3 const & a, vec3 const & p )
{
    bound3 res = bound3(
        (p.x<a.mMinX) ? p.x : a.mMinX,
        (p.x>a.mMaxX) ? p.x : a.mMaxX,
        (p.y<a.mMinY) ? p.y : a.mMinY,
        (p.y>a.mMaxY) ? p.y : a.mMaxY,
        (p.z<a.mMinZ) ? p.z : a.mMinZ,
        (p.z>a.mMaxZ) ? p.z : a.mMaxZ );

    return res;
}
#if 0
inline int containsp( bound3 const & b, vec3 const & p )
{
    if( p.x < b.mMinX ) return 0;
    if( p.y < b.mMinY ) return 0;
    if( p.z < b.mMinZ ) return 0;
    if( p.x > b.mMaxX ) return 0;
    if( p.y > b.mMaxY ) return 0;
    if( p.z > b.mMaxZ ) return 0;
    return 1;
}

// 0 = a and b are disjoint
// 1 = a and b intersect
// 2 = b is fully contained in a
// 3 = a is fully contained in b
inline int contains( bound3 const & a, bound3 const & b )
{
    int nBinA = containsp( a, vec3(b.mMinX,b.mMinY,b.mMinZ) ) +
                containsp( a, vec3(b.mMaxX,b.mMinY,b.mMinZ) ) +
                containsp( a, vec3(b.mMinX,b.mMaxY,b.mMinZ) ) +
                containsp( a, vec3(b.mMaxX,b.mMaxY,b.mMinZ) ) +
                containsp( a, vec3(b.mMinX,b.mMinY,b.mMaxZ) ) +
                containsp( a, vec3(b.mMaxX,b.mMinY,b.mMaxZ) ) +
                containsp( a, vec3(b.mMinX,b.mMaxY,b.mMaxZ) ) +
                containsp( a, vec3(b.mMaxX,b.mMaxY,b.mMaxZ) );

    int nAinB = containsp( b, vec3(a.mMinX,a.mMinY,a.mMinZ) ) +
                containsp( b, vec3(a.mMaxX,a.mMinY,a.mMinZ) ) +
                containsp( b, vec3(a.mMinX,a.mMaxY,a.mMinZ) ) +
                containsp( b, vec3(a.mMaxX,a.mMaxY,a.mMinZ) ) +
                containsp( b, vec3(a.mMinX,a.mMinY,a.mMaxZ) ) +
                containsp( b, vec3(a.mMaxX,a.mMinY,a.mMaxZ) ) +
                containsp( b, vec3(a.mMinX,a.mMaxY,a.mMaxZ) ) +
                containsp( b, vec3(a.mMaxX,a.mMaxY,a.mMaxZ) );

    if( nAinB==0 && nBinA==0 )   return 0;
    if( nAinB!=0 && nBinA!=0 )   return 1;
    if( nAinB==0 && nBinA!=0 )   return 2;
  /*if( nAinB!=0 && nBinA==0 )*/ return 3;
}
#endif

// 0 if they are disjoint
// 1 if they intersect 
inline int overlap( bound3 const & a, bound3 const & b )
{
    if( a.mMaxX < b.mMinX ) return 0;
    if( a.mMinX > b.mMaxX ) return 0;
    if( a.mMaxY < b.mMinY ) return 0;
    if( a.mMinY > b.mMaxY ) return 0;
    if( a.mMaxZ < b.mMinZ ) return 0;
    if( a.mMinZ > b.mMaxZ ) return 0;
    return 1;
}

inline bound3 compute( const vec3 * const p, const int num )
{
    bound3 res = bound3( p[0].x, p[0].x, 
                         p[0].y, p[0].y,
                         p[0].z, p[0].z );

    for( int k=1; k<num; k++ ) 
    {
        res.mMinX = (p[k].x<res.mMinX) ? p[k].x : res.mMinX;
        res.mMaxX = (p[k].x>res.mMaxX) ? p[k].x : res.mMaxX;
        res.mMinY = (p[k].y<res.mMinY) ? p[k].y : res.mMinY;
        res.mMaxY = (p[k].y>res.mMaxY) ? p[k].y : res.mMaxY;
        res.mMinZ = (p[k].z<res.mMinZ) ? p[k].z : res.mMinZ;
        res.mMaxZ = (p[k].z>res.mMaxZ) ? p[k].z : res.mMaxZ;
    }

    return res;
}

inline float diagonal( bound3 const & bbox )
{
    const float dx = bbox.mMaxX - bbox.mMinX;
    const float dy = bbox.mMaxY - bbox.mMinY;
    const float dz = bbox.mMaxZ - bbox.mMinZ;
    return sqrtf( dx*dx + dy*dy + dz*dz );
}

inline float volume( bound3 const & bbox )
{
    const float dx = bbox.mMaxX - bbox.mMinX;
    const float dy = bbox.mMaxY - bbox.mMinY;
    const float dz = bbox.mMaxZ - bbox.mMinZ;
    return dx*dy*dz;
}


#endif

