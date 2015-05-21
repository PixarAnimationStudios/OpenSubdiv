//
//   Copyright 2015 Pixar
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
// Patches.VertexGregoryBasis
//----------------------------------------------------------
#ifdef OSD_PATCH_VERTEX_GREGORY_BASIS_SHADER

layout(location = 0) in vec4 position;
OSD_USER_VARYING_ATTRIBUTE_DECLARE

out block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    outpt.v.position = OsdModelViewMatrix() * position;
    outpt.v.patchCoord = ivec4(0);
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(position);
    OSD_USER_VARYING_PER_VERTEX();
}

#endif

//----------------------------------------------------------
// Patches.TessControlGregoryBasis
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_CONTROL_GREGORY_BASIS_SHADER

layout(vertices = 20) out;

in block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} outpt[];

#define ID gl_InvocationID

void main()
{
    outpt[ID].v = inpt[ID].v;
    OSD_USER_VARYING_PER_CONTROL_POINT(ID, ID);

    ivec3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(gl_PrimitiveID));

    outpt[ID].v.patchCoord = OsdGetPatchCoord(patchParam);

    if (ID == 0) {
        OSD_PATCH_CULL(OSD_PATCH_INPUT_SIZE);

        vec4 tessLevelOuter = vec4(0);
        vec2 tessLevelInner = vec2(0);

        OsdGetTessLevels(inpt[0].v.position.xyz, inpt[15].v.position.xyz,
                         inpt[10].v.position.xyz, inpt[5].v.position.xyz,
                         patchParam, tessLevelOuter, tessLevelInner);

        gl_TessLevelOuter[0] = tessLevelOuter[0];
        gl_TessLevelOuter[1] = tessLevelOuter[1];
        gl_TessLevelOuter[2] = tessLevelOuter[2];
        gl_TessLevelOuter[3] = tessLevelOuter[3];

        gl_TessLevelInner[0] = tessLevelInner[0];
        gl_TessLevelInner[1] = tessLevelInner[1];
    }
}

#endif

//----------------------------------------------------------
// Patches.TessEvalGregory
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_EVAL_GREGORY_BASIS_SHADER

layout(quads) in;

#if defined OSD_FRACTIONAL_ODD_SPACING
    layout(fractional_odd_spacing) in;
#elif defined OSD_FRACTIONAL_EVEN_SPACING
    layout(fractional_even_spacing) in;
#endif

in block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    OutputVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    float u = gl_TessCoord.x,
          v = gl_TessCoord.y;

    vec3 p[20];

    for (int i = 0; i < 20; ++i) {
        p[i] = inpt[i].v.position.xyz;
    }
    vec3 q[16];

    float U = 1-u, V=1-v;

    float d11 = u+v; if(u+v==0.0f) d11 = 1.0f;
    float d12 = U+v; if(U+v==0.0f) d12 = 1.0f;
    float d21 = u+V; if(u+V==0.0f) d21 = 1.0f;
    float d22 = U+V; if(U+V==0.0f) d22 = 1.0f;

#if 1
    q[ 5] = (u*p[3] + v*p[4])/d11;
    q[ 6] = (U*p[9] + v*p[8])/d12;
    q[ 9] = (u*p[19] + V*p[18])/d21;
    q[10] = (U*p[13] + V*p[14])/d22;
#else
    q[ 5] = (p[3] + p[4])/2.0;
    q[ 6] = (p[9] + p[8])/2.0;
    q[ 9] = (p[19] + p[18])/2.0;
    q[10] = (p[13] + p[14])/2.0;
#endif

    q[ 0] = p[0];
    q[ 1] = p[1];
    q[ 2] = p[7];
    q[ 3] = p[5];
    q[ 4] = p[2];
    q[ 7] = p[6];
    q[ 8] = p[16];
    q[11] = p[12];
    q[12] = p[15];
    q[13] = p[17];
    q[14] = p[11];
    q[15] = p[10];

    vec3 WorldPos  = vec3(0, 0, 0);
    vec3 Tangent   = vec3(0, 0, 0);
    vec3 BiTangent = vec3(0, 0, 0);

#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
    float B[4], D[4], C[4];
    vec3 BUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0)),
         DUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0)),
         CUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0));
    vec3 dUU = vec3(0);
    vec3 dVV = vec3(0);
    vec3 dUV = vec3(0);

    Univar4x4(u, B, D, C);

    for (int i=0; i<4; ++i) {
        for (uint j=0; j<4; ++j) {
            vec3 A = q[4*i + j];
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
            CUCP[i] += A * C[j];
        }
    }

    Univar4x4(v, B, D, C);

    for (int i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
        dUU += B[i] * CUCP[i];
        dVV += C[i] * BUCP[i];
        dUV += D[i] * DUCP[i];
    }

    int level = inpt[0].v.patchCoord.z;
    BiTangent *= 3 * level;
    Tangent *= 3 * level;
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

    outpt.v.Nu = Nu;
    outpt.v.Nv = Nv;

#else
    float B[4], D[4];
    vec3 BUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0)),
         DUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0));

    Univar4x4(u, B, D);

    for (int i=0; i<4; ++i) {
        for (uint j=0; j<4; ++j) {
            vec3 A = q[4*i + j];
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    Univar4x4(v, B, D);

    for (int i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
    }
    int level = inpt[0].v.patchCoord.z;
    BiTangent *= 3 * level;
    Tangent *= 3 * level;

    vec3 normal = normalize(cross(Tangent, BiTangent));

#endif
    outpt.v.position = vec4(WorldPos, 1.0f);
    outpt.v.normal = normal;
    outpt.v.tangent = Tangent;
    outpt.v.bitangent = BiTangent;

    vec2 UV = vec2(u, v);

    OSD_USER_VARYING_PER_EVAL_POINT(UV, 0, 5, 15, 10);

    outpt.v.tessCoord = UV;
    outpt.v.patchCoord = OsdInterpolatePatchCoord(UV, inpt[0].v.patchCoord);

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = OsdProjectionMatrix() * outpt.v.position;
}

#endif
