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

//----------------------------------------------------------
// Patches.TessVertexBSpline
//----------------------------------------------------------
#ifdef OSD_PATCH_VERTEX_BSPLINE_SHADER

layout(location = 0) in vec4 position;
OSD_USER_VARYING_ATTRIBUTE_DECLARE

out block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    outpt.v.position = OsdModelViewMatrix() * position;
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(position);
    OSD_USER_VARYING_PER_VERTEX();
}

#endif

//----------------------------------------------------------
// Patches.TessControlBSpline
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER

// Regular
uniform mat4 Q = mat4(
    1.f/6.f, 4.f/6.f, 1.f/6.f, 0.f,
    0.f,     4.f/6.f, 2.f/6.f, 0.f,
    0.f,     2.f/6.f, 4.f/6.f, 0.f,
    0.f,     1.f/6.f, 4.f/6.f, 1.f/6.f
);

// Infinite sharp
uniform mat4 Mi = mat4(
    1.f/6.f, 4.f/6.f, 1.f/6.f, 0.f,
    0.f,     4.f/6.f, 2.f/6.f, 0.f,
    0.f,     2.f/6.f, 4.f/6.f, 0.f,
    0.f,     0.f,     1.f,     0.f
);

// Boundary / Corner
uniform mat4x3 B = mat4x3( 
    1.f,     0.f,     0.f,
    4.f/6.f, 2.f/6.f, 0.f,
    2.f/6.f, 4.f/6.f, 0.f,
    1.f/6.f, 4.f/6.f, 1.f/6.f
);

layout(vertices = 16) out;

in block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    ControlVertex v;
#if defined OSD_PATCH_SINGLE_CREASE
    vec4 P1;
    vec4 P2;
    float sharpness;
#endif
    OSD_USER_VARYING_DECLARE
} outpt[];

#define ID gl_InvocationID

// compute single-crease patch matrix
mat4
ComputeMatrixSimplified(float sharpness)
{
    float s = pow(2.0f, sharpness);
    float s2 = s*s;
    float s3 = s2*s;

    mat4 m = mat4(
        0, s + 1 + 3*s2 - s3, 7*s - 2 - 6*s2 + 2*s3, (1-s)*(s-1)*(s-1),
        0,       (1+s)*(1+s),        6*s - 2 - 2*s2,       (s-1)*(s-1),
        0,               1+s,               6*s - 2,               1-s,
        0,                 1,               6*s - 2,                 1);

    m /= (s*6.0);
    m[0][0] = 1.0/6.0;

    return m;
}

void main()
{
    int i = ID%4;
    int j = ID/4;

#if defined OSD_PATCH_BOUNDARY
    vec3 H[3];
    for (int l=0; l<3; ++l) {
        H[l] = vec3(0,0,0);
        for (int k=0; k<4; ++k) {
            H[l] += Q[i][k] * inpt[l*4 + k].v.position.xyz;
        }
    }

    vec3 pos = vec3(0,0,0);
    for (int k=0; k<3; ++k) {
        pos += B[j][k]*H[k];
    }

    outpt[ID].v.position = vec4(pos, 1.0);

#elif defined OSD_PATCH_CORNER
    vec3 H[3];
    for (int l=0; l<3; ++l) {
        H[l] = vec3(0,0,0);
        for (int k=0; k<3; ++k) {
            H[l] += B[3-i][2-k] * inpt[l*3 + k].v.position.xyz;
        }
    }

    vec3 pos = vec3(0,0,0);
    for (int k=0; k<3; ++k) {
        pos += B[j][k]*H[k];
    }

    outpt[ID].v.position = vec4(pos, 1.0);

#else // not OSD_PATCH_BOUNDARY, not OSD_PATCH_CORNER
    vec3 H[4];
    for (int l=0; l<4; ++l) {
        H[l] = vec3(0,0,0);
        for (int k=0; k<4; ++k) {
            H[l] += Q[i][k] * inpt[l*4 + k].v.position.xyz;
        }
    }

#if defined OSD_PATCH_SINGLE_CREASE
    float sharpness = GetSharpness();
    float Sf = floor(sharpness);
    float Sc = ceil(sharpness);
    float Sr = fract(sharpness);
    mat4 Mf = ComputeMatrixSimplified(Sf);
    mat4 Mc = ComputeMatrixSimplified(Sc);
    mat4 Mj = (1-Sr) * Mf + Sr * Mi;
    mat4 Ms = (1-Sr) * Mf + Sr * Mc;

    vec3 P = vec3(0);
    vec3 P1 = vec3(0);
    vec3 P2 = vec3(0);
    for (int k=0; k<4; ++k) {
        P  += Mi[j][k]*H[k]; // 0 to 1-2^(-Sf)
        P1 += Mj[j][k]*H[k]; // 1-2^(-Sf) to 1-2^(-Sc)
        P2 += Ms[j][k]*H[k]; // 1-2^(-Sc) to 1
   }
    outpt[ID].v.position = vec4(P, 1.0);
    outpt[ID].P1 = vec4(P1, 1.0);
    outpt[ID].P2 = vec4(P2, 1.0);
    outpt[ID].sharpness = sharpness;

#else  // REGULAR
    vec3 pos = vec3(0,0,0);
    for (int k=0; k<4; ++k) {
        pos += Q[j][k]*H[k];
    }
    outpt[ID].v.position = vec4(pos, 1.0);
#endif

#endif


#if defined OSD_PATCH_BOUNDARY
    const int p[16] = int[]( 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 );
#elif defined OSD_PATCH_CORNER
    const int p[16] = int[]( 0, 1, 2, 2, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8 );
#else
    const int p[16] = int[]( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
#endif

#if OSD_TRANSITION_ROTATE == 0
    const int r[16] = int[]( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
#elif OSD_TRANSITION_ROTATE == 1
    const int r[16] = int[]( 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 );
#elif OSD_TRANSITION_ROTATE == 2
    const int r[16] = int[]( 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 );
#elif OSD_TRANSITION_ROTATE == 3
    const int r[16] = int[]( 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 );
#endif

    OSD_USER_VARYING_PER_CONTROL_POINT(ID, p[r[ID]]);

    int patchLevel = GetPatchLevel();

    // +0.5 to avoid interpolation error of integer value
    outpt[ID].v.patchCoord = vec4(0, 0,
                                  patchLevel+0.5,
                                  GetPrimitiveID()+0.5);

    OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER;

    if (ID == 0) {
        OSD_PATCH_CULL(OSD_PATCH_INPUT_SIZE);

#ifdef OSD_PATCH_TRANSITION
        vec3 cp[OSD_PATCH_INPUT_SIZE];
        for(int k = 0; k < OSD_PATCH_INPUT_SIZE; ++k) cp[k] = inpt[k].v.position.xyz;
        SetTransitionTessLevels(cp, patchLevel);
#else
    #if defined OSD_PATCH_BOUNDARY
        const int p[4] = int[]( 1, 2, 5, 6 );
    #elif defined OSD_PATCH_CORNER
        const int p[4] = int[]( 1, 2, 4, 5 );
    #else
        const int p[4] = int[]( 5, 6, 9, 10 );
    #endif

    #ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
        gl_TessLevelOuter[0] = TessAdaptive(inpt[p[0]].v.position.xyz, inpt[p[2]].v.position.xyz);
        gl_TessLevelOuter[1] = TessAdaptive(inpt[p[0]].v.position.xyz, inpt[p[1]].v.position.xyz);
        gl_TessLevelOuter[2] = TessAdaptive(inpt[p[1]].v.position.xyz, inpt[p[3]].v.position.xyz);
        gl_TessLevelOuter[3] = TessAdaptive(inpt[p[2]].v.position.xyz, inpt[p[3]].v.position.xyz);
        gl_TessLevelInner[0] = max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] = max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #else
        gl_TessLevelInner[0] = GetTessLevel(patchLevel);
        gl_TessLevelInner[1] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[0] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[1] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[2] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[3] = GetTessLevel(patchLevel);
    #endif
#endif
    }
}

#endif

//----------------------------------------------------------
// Patches.TessEvalBSpline
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_EVAL_BSPLINE_SHADER

#ifdef OSD_TRANSITION_TRIANGLE_SUBPATCH
    layout(triangles) in;
#else
    layout(quads) in;
#endif

#if defined OSD_FRACTIONAL_ODD_SPACING
    layout(fractional_odd_spacing) in;
#elif defined OSD_FRACTIONAL_EVEN_SPACING
    layout(fractional_even_spacing) in;
#endif

in block {
    ControlVertex v;
#if defined OSD_PATCH_SINGLE_CREASE
    vec4 P1;
    vec4 P2;
    float sharpness;
#endif
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    OutputVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
#ifdef OSD_PATCH_TRANSITION
    vec2 UV = GetTransitionSubpatchUV();
#else
    vec2 UV = gl_TessCoord.xy;
#endif

#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
    float B[4], D[4], C[4];
    vec3 BUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0)),
         DUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0)),
         CUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0));
    Univar4x4(UV.x, B, D, C);
#else
    float B[4], D[4];
    vec3 BUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0)),
         DUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0));
    Univar4x4(UV.x, B, D);
#endif

#if defined OSD_PATCH_SINGLE_CREASE
    float sharpness = inpt[0].sharpness;
    float s0 = 1.0 - pow(2.0f, -floor(sharpness));
    float s1 = 1.0 - pow(2.0f, -ceil(sharpness));
#endif

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
#if defined OSD_PATCH_SINGLE_CREASE
#if OSD_TRANSITION_ROTATE == 1
            int k = 4*(3-j) + i;
            float s = 1-UV.x;
#elif OSD_TRANSITION_ROTATE == 2
            int k = 4*(3-i) + (3-j);
            float s = 1-UV.y;
#elif OSD_TRANSITION_ROTATE == 3
            int k = 4*j + (3-i);
            float s = UV.x;
#else // ROTATE=0 or non-transition
            int k = 4*i + j;
            float s = UV.y;
#endif
            vec3 A = (s < s0) ?
                 inpt[k].v.position.xyz :
                 ((s < s1) ?
                  inpt[k].P1.xyz :
                  inpt[k].P2.xyz);

#else // !SINGLE_CREASE
#if OSD_TRANSITION_ROTATE == 1
            vec3 A = inpt[4*(3-j) + i].v.position.xyz;
#elif OSD_TRANSITION_ROTATE == 2
            vec3 A = inpt[4*(3-i) + (3-j)].v.position.xyz;
#elif OSD_TRANSITION_ROTATE == 3
            vec3 A = inpt[4*j + (3-i)].v.position.xyz;
#else // OSD_TRANSITION_ROTATE == 0, or non-transition patch
            vec3 A = inpt[4*i + j].v.position.xyz;
#endif
#endif
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
            CUCP[i] += A * C[j];
#endif
        }
    }

    vec3 WorldPos  = vec3(0);
    vec3 Tangent   = vec3(0);
    vec3 BiTangent = vec3(0);

#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
    // used for weingarten term
    Univar4x4(UV.y, B, D, C);

    vec3 dUU = vec3(0);
    vec3 dVV = vec3(0);
    vec3 dUV = vec3(0);

    for (int k=0; k<4; ++k) {
        WorldPos  += B[k] * BUCP[k];
        Tangent   += B[k] * DUCP[k];
        BiTangent += D[k] * BUCP[k];

        dUU += B[k] * CUCP[k];
        dVV += C[k] * BUCP[k];
        dUV += D[k] * DUCP[k];
    }

    int level = int(inpt[0].v.ptexInfo.z);
    Tangent *= 3 * level;
    BiTangent *= 3 * level;
    dUU *= 6 * level;
    dVV *= 6 * level;
    dUV *= 9 * level;

    vec3 n = cross(Tangent, BiTangent);
    vec3 normal = normalize(n);

    float E = dot(Tangent, Tangent);
    float F = dot(Tangent, BiTangent);
    float G = dot(BiTangent, BiTangent);
    float e = dot(normal, dUU);
    float f = dot(normal, dUV);
    float g = dot(normal, dVV);

    vec3 Nu = (f*F-e*G)/(E*G-F*F) * Tangent + (e*F-f*E)/(E*G-F*F) * BiTangent;
    vec3 Nv = (g*F-f*G)/(E*G-F*F) * Tangent + (f*F-g*E)/(E*G-F*F) * BiTangent;

    Nu = Nu/length(n) - n * (dot(Nu,n)/pow(dot(n,n), 1.5));
    Nv = Nv/length(n) - n * (dot(Nv,n)/pow(dot(n,n), 1.5));

    OSD_COMPUTE_PTEX_COMPATIBLE_DERIVATIVES(OSD_TRANSITION_ROTATE);
#else
    Univar4x4(UV.y, B, D);

    for (int k=0; k<4; ++k) {
        WorldPos  += B[k] * BUCP[k];
        Tangent   += B[k] * DUCP[k];
        BiTangent += D[k] * BUCP[k];
    }
    int level = int(inpt[0].v.ptexInfo.z);
    Tangent *= 3 * level;
    BiTangent *= 3 * level;

    vec3 normal = normalize(cross(Tangent, BiTangent));

    OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(OSD_TRANSITION_ROTATE);
#endif

    outpt.v.position = vec4(WorldPos, 1.0f);
    outpt.v.normal = normal;

    OSD_USER_VARYING_PER_EVAL_POINT(UV, 5, 6, 9, 10);

    outpt.v.patchCoord = inpt[0].v.patchCoord;

#if OSD_TRANSITION_ROTATE == 1
    outpt.v.patchCoord.xy = vec2(UV.y, 1.0-UV.x);
#elif OSD_TRANSITION_ROTATE == 2
    outpt.v.patchCoord.xy = vec2(1.0-UV.x, 1.0-UV.y);
#elif OSD_TRANSITION_ROTATE == 3
    outpt.v.patchCoord.xy = vec2(1.0-UV.y, UV.x);
#else // OSD_TRANNSITION_ROTATE == 0, or non-transition patch
    outpt.v.patchCoord.xy = vec2(UV.x, UV.y);
#endif

    OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER;

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = (OsdProjectionMatrix() * vec4(WorldPos, 1.0f));
}

#endif
