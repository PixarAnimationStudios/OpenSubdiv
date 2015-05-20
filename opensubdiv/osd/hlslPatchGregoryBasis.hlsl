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

#if defined OSD_FRACTIONAL_ODD_SPACING
    #define HS_PARTITION "fractional_odd"
#elif defined OSD_FRACTIONAL_EVEN_SPACING
    #define HS_PARTITION "fractional_even"
#else
    #define HS_PARTITION "integer"
#endif

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------

void vs_main_patches( in InputVertex input,
                      out HullVertex output )
{
    output.position = mul(OsdModelViewMatrix(), input.position);
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(input.position);
}

//----------------------------------------------------------
// Patches.HullGregoryBasis
//----------------------------------------------------------

HS_CONSTANT_FUNC_OUT
HSConstFunc(
    InputPatch<HullVertex, 20> patch,
    uint primitiveID : SV_PrimitiveID)
{
    HS_CONSTANT_FUNC_OUT output;

    int3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(primitiveID));

    OSD_PATCH_CULL(20);

    float4 tessLevelOuter = float4(0,0,0,0);
    float4 tessLevelInner = float4(0,0,0,0);
    float4 tessOuterLo = float4(0,0,0,0);
    float4 tessOuterHi = float4(0,0,0,0);

    OsdGetTessLevels(patch[0].position.xyz, patch[5].position.xyz,
                     patch[10].position.xyz, patch[15].position.xyz,
                     patchParam, tessLevelOuter, tessLevelInner);

    output.tessLevelOuter[0] = tessLevelOuter[0];
    output.tessLevelOuter[1] = tessLevelOuter[1];
    output.tessLevelOuter[2] = tessLevelOuter[2];
    output.tessLevelOuter[3] = tessLevelOuter[3];

    output.tessLevelInner[0] = tessLevelInner[0];
    output.tessLevelInner[1] = tessLevelInner[1];

    return output;
}

[domain("quad")]
[partitioning(HS_PARTITION)]
[outputtopology("triangle_cw")]
[outputcontrolpoints(20)]
[patchconstantfunc("HSConstFunc")]
HullVertex hs_main_patches(
    in InputPatch<HullVertex, 20> patch,
    uint primitiveID : SV_PrimitiveID,
    in uint ID : SV_OutputControlPointID )
{
    int3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(primitiveID));
    HullVertex output;

    output.position = float4(patch[ID].position.xyz, 1.0);
    output.patchCoord = OsdGetPatchCoord(patchParam);

    return output;
}

//----------------------------------------------------------
// Patches.DomainGregory
//----------------------------------------------------------

[domain("quad")]
void ds_main_patches(
    in HS_CONSTANT_FUNC_OUT input,
    in OutputPatch<HullVertex, 20> patch,
    in float2 uv : SV_DomainLocation,
    out OutputVertex output )
{
    float u = uv.x,
          v = uv.y;

    float3 p[20];
    for (int i = 0; i < 20; ++i) {
        p[i] = patch[i].position.xyz;
    }
    float3 q[16];

    float U = 1-u, V=1-v;

    float d11 = u+v; if(u+v==0.0f) d11 = 1.0f;
    float d12 = U+v; if(U+v==0.0f) d12 = 1.0f;
    float d21 = u+V; if(u+V==0.0f) d21 = 1.0f;
    float d22 = U+V; if(U+V==0.0f) d22 = 1.0f;

    q[ 5] = (u*p[3] + v*p[4])/d11;
    q[ 6] = (U*p[9] + v*p[8])/d12;
    q[ 9] = (u*p[19] + V*p[18])/d21;
    q[10] = (U*p[13] + V*p[14])/d22;

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

    float3 WorldPos  = float3(0, 0, 0);
    float3 Tangent   = float3(0, 0, 0);
    float3 BiTangent = float3(0, 0, 0);

#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
    float B[4], D[4], C[4];

    float3 BUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)},
           DUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)},
           CUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)};

    float3 dUU = float3(0, 0, 0);
    float3 dVV = float3(0, 0, 0);
    float3 dUV = float3(0, 0, 0);

    Univar4x4(u, B, D, C);

    for (int i=0; i<4; ++i) {
        for (uint j=0; j<4; ++j) {
            float3 A = q[4*i + j];
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

    int level = patch[0].patchCoord.z;
    BiTangent *= 3 * level;
    Tangent *= 3 * level;
    dUU *= 6 * level;
    dVV *= 6 * level;
    dUV *= 9 * level;

    float3 n = cross(Tangent, BiTangent);
    float3 normal = normalize(n);

    float E = dot(Tangent, Tangent);
    float F = dot(Tangent, BiTangent);
    float G = dot(BiTangent, BiTangent);
    float e = dot(normal, dUU);
    float f = dot(normal, dUV);
    float g = dot(normal, dVV);

    float3 Nu = (f*F-e*G)/(E*G-F*F) * Tangent + (e*F-f*E)/(E*G-F*F) * BiTangent;
    float3 Nv = (g*F-f*G)/(E*G-F*F) * Tangent + (f*F-g*E)/(E*G-F*F) * BiTangent;

    Nu = Nu/length(n) - n * (dot(Nu,n)/pow(dot(n,n), 1.5));
    Nv = Nv/length(n) - n * (dot(Nv,n)/pow(dot(n,n), 1.5));

    output.Nu = Nu;
    output.Nv = Nv;

#else
    float B[4], D[4];
    float3 BUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)},
           DUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)};

    Univar4x4(uv.x, B, D);

    for (int i=0; i<4; ++i) {
        for (uint j=0; j<4; ++j) {
            float3 A = q[4*i + j];
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    Univar4x4(uv.y, B, D);

    for (uint i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
    }
    int level = patch[0].patchCoord.z;
    BiTangent *= 3 * level;
    Tangent *= 3 * level;

    float3 normal = normalize(cross(Tangent, BiTangent));

#endif

    output.position = float4(WorldPos, 1.0f);
    output.normal = normal;
    output.tangent = Tangent;
    output.bitangent = BiTangent;

    output.edgeDistance = 0;

    float2 UV = float2(u, v);
    output.patchCoord = OsdInterpolatePatchCoord(UV, patch[0].patchCoord);

    OSD_DISPLACEMENT_CALLBACK;

    output.positionOut = mul(OsdProjectionMatrix(), output.position);
}
