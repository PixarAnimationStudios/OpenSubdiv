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
    output.position = input.position;
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(input.position);
}

//----------------------------------------------------------
// Patches.HullBSpline
//----------------------------------------------------------

// Regular
static float4x4 Q = {
    1.f/6.f, 4.f/6.f, 1.f/6.f, 0.f,
    0.f,     4.f/6.f, 2.f/6.f, 0.f,
    0.f,     2.f/6.f, 4.f/6.f, 0.f,
    0.f,     1.f/6.f, 4.f/6.f, 1.f/6.f
};

// Infinite sharp
static float4x4 Mi = {
    1.f/6.f, 4.f/6.f, 1.f/6.f, 0.f,
    0.f,     4.f/6.f, 2.f/6.f, 0.f,
    0.f,     2.f/6.f, 4.f/6.f, 0.f,
    0.f,     0.f,     1.f,     0.f
};

// compute single-crease patch matrix
float4x4
ComputeMatrixSimplified(float sharpness)
{
    float s = pow(2.0f, sharpness);
    float s2 = s*s;
    float s3 = s2*s;

    float4x4 m ={
        0, s + 1 + 3*s2 - s3, 7*s - 2 - 6*s2 + 2*s3, (1-s)*(s-1)*(s-1),
        0,       (1+s)*(1+s),        6*s - 2 - 2*s2,       (s-1)*(s-1),
        0,               1+s,               6*s - 2,               1-s,
        0,                 1,               6*s - 2,                 1 };

    m /= (s*6.0);
    m[0][0] = 1.0/6.0;

    return m;
}

[domain("quad")]
[partitioning(HS_PARTITION)]
[outputtopology("triangle_cw")]
[outputcontrolpoints(16)]
[patchconstantfunc("HSConstFunc")]
HullVertex hs_main_patches(
    in InputPatch<HullVertex, 16> patch,
    uint primitiveID : SV_PrimitiveID,
    in uint ID : SV_OutputControlPointID )
{
    int i = ID%4;
    int j = ID/4;

    float3 position[16];
    for (int p=0; p<16; ++p) {
        position[p] = patch[p].position.xyz;
    }

    int3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(primitiveID));

    OsdComputeBSplineBoundaryPoints(position, patchParam);

    float3 H[4];
    for (int l=0; l<4; ++l) {
        H[l] = float3(0,0,0);
        for(int k=0; k<4; ++k) {
            H[l] += Q[i][k] * position[l*4 + k];
        }
    }

    HullVertex output;
#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    float sharpness = OsdGetPatchSharpness(patchParam);
    if (sharpness > 0) {
        float Sf = floor(sharpness);
        float Sc = ceil(sharpness);
        float Sr = frac(sharpness);
        float4x4 Mf = ComputeMatrixSimplified(Sf);
        float4x4 Mc = ComputeMatrixSimplified(Sc);
        float4x4 Mj = (1-Sr) * Mf + Sr * Mi;
        float4x4 Ms = (1-Sr) * Mf + Sr * Mc;

        float3 pos = float3(0,0,0);
        float3 P1 = float3(0,0,0);
        float3 P2 = float3(0,0,0);
        for (int k=0; k<4; ++k) {
            pos += Mi[j][k]*H[k]; // 0 to 1-2^(-Sf)
            P1  += Mj[j][k]*H[k]; // 1-2^(-Sf) to 1-2^(-Sc)
            P2  += Ms[j][k]*H[k]; // 1-2^(-Sc) to 1
        }
        output.position = float4(pos, 1.0);
        output.P1 = float4(P1, 1.0);
        output.P2 = float4(P2, 1.0);
        output.sharpness = sharpness;
    } else {
        float3 pos = float3(0,0,0);
        for (int k=0; k<4; ++k){
            pos += Q[j][k]*H[k];
        }
        output.position = float4(pos, 1.0);
        output.sharpness = 0;
    }
#else
    {
        float3 pos = float3(0,0,0);
        for (int k=0; k<4; ++k){
            pos += Q[j][k]*H[k];
        }
        output.position = float4(pos, 1.0);
    }
#endif

    output.patchCoord = OsdGetPatchCoord(patchParam);

    return output;
}

HS_CONSTANT_FUNC_OUT
HSConstFunc(
    InputPatch<HullVertex, 16> patch,
    OutputPatch<HullVertex, 16> bezierPatch,
    uint primitiveID : SV_PrimitiveID)
{
    HS_CONSTANT_FUNC_OUT output;

    float3 position[16];
    for (int p=0; p<16; ++p) {
        position[p] = bezierPatch[p].position.xyz;
    }

    int3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(primitiveID));

    OsdComputeBSplineBoundaryPoints(position, patchParam);

    OSD_PATCH_CULL(16);

    float4 tessLevelOuter = float4(0,0,0,0);
    float4 tessLevelInner = float4(0,0,0,0);
    float4 tessOuterLo = float4(0,0,0,0);
    float4 tessOuterHi = float4(0,0,0,0);

    OsdGetTessLevels(position, patchParam,
                     tessLevelOuter, tessLevelInner,
                     tessOuterLo, tessOuterHi);

    output.tessLevelOuter[0] = tessLevelOuter[0];
    output.tessLevelOuter[1] = tessLevelOuter[1];
    output.tessLevelOuter[2] = tessLevelOuter[2];
    output.tessLevelOuter[3] = tessLevelOuter[3];

    output.tessLevelInner[0] = tessLevelInner[0];
    output.tessLevelInner[1] = tessLevelInner[1];

    output.tessOuterLo = tessOuterLo;
    output.tessOuterHi = tessOuterHi;

    return output;
}

//----------------------------------------------------------
// Patches.DomainBSpline
//----------------------------------------------------------

[domain("quad")]
void ds_main_patches(
    in HS_CONSTANT_FUNC_OUT input,
    in OutputPatch<HullVertex, 16> patch,
    in float2 domainCoord : SV_DomainLocation,
    out OutputVertex output )
{
    float2 UV = OsdGetTessParameterization(domainCoord,
                                           input.tessOuterLo,
                                           input.tessOuterHi);

#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
    float B[4], D[4], C[4];
    float3 BUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)},
           DUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)},
           CUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)};
    Univar4x4(UV.x, B, D, C);
#else
    float B[4], D[4];
    float3 BUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)},
           DUCP[4] = {float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)};
    Univar4x4(UV.x, B, D);
#endif

    // ----------------------------------------------------------------
#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    // sharpness
    float sharpness = patch[0].sharpness;
    if (sharpness != 0) {
        float s0 = 1.0 - pow(2.0f, -floor(sharpness));
        float s1 = 1.0 - pow(2.0f, -ceil(sharpness));

        for (int i=0; i<4; ++i) {
            for (int j=0; j<4; ++j) {
                int k = 4*i + j;
                float s = UV.y;

                float3 A = (s < s0) ?
                     patch[k].position.xyz :
                     ((s < s1) ?
                      patch[k].P1.xyz :
                      patch[k].P2.xyz);

                BUCP[i] += A * B[j];
                DUCP[i] += A * D[j];
#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
                CUCP[i] += A * C[j];
#endif
            }
        }
        output.sharpness = sharpness;
    } else {
        for (int i=0; i<4; ++i) {
            for (int j=0; j<4; ++j) {
                float3 A = patch[4*i + j].position.xyz;
                BUCP[i] += A * B[j];
                DUCP[i] += A * D[j];
#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
                CUCP[i] += A * C[j];
#endif
            }
        }
        output.sharpness = 0;
    }
#else
    // ----------------------------------------------------------------
        for (int i=0; i<4; ++i) {
            for (int j=0; j<4; ++j) {
                float3 A = patch[4*i + j].position.xyz;
                BUCP[i] += A * B[j];
                DUCP[i] += A * D[j];
#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
                CUCP[i] += A * C[j];
#endif
            }
        }
#endif
    // ----------------------------------------------------------------

    float3 position = float3(0,0,0);
    float3 uTangent = float3(0,0,0);
    float3 vTangent = float3(0,0,0);

#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
    // used for weingarten term
    Univar4x4(UV.y, B, D, C);

    float3 dUU = float3(0,0,0);
    float3 dVV = float3(0,0,0);
    float3 dUV = float3(0,0,0);

    for (int k=0; k<4; ++k) {
        position += B[k] * BUCP[k];
        uTangent += B[k] * DUCP[k];
        vTangent += D[k] * BUCP[k];

        dUU += B[k] * CUCP[k];
        dVV += C[k] * BUCP[k];
        dUV += D[k] * DUCP[k];
    }

    int level = patch[0].patchCoord.z;
    uTangent *= 3 * level;
    vTangent *= 3 * level;
    dUU *= 6 * level;
    dVV *= 6 * level;
    dUV *= 9 * level;

    float3 n = cross(uTangent, vTangent);
    float3 normal = normalize(n);

    float E = dot(uTangent, uTangent);
    float F = dot(uTangent, vTangent);
    float G = dot(vTangent, vTangent);
    float e = dot(normal, dUU);
    float f = dot(normal, dUV);
    float g = dot(normal, dVV);

    float3 Nu = (f*F-e*G)/(E*G-F*F) * uTangent + (e*F-f*E)/(E*G-F*F) * vTangent;
    float3 Nv = (g*F-f*G)/(E*G-F*F) * uTangent + (f*F-g*E)/(E*G-F*F) * vTangent;

    Nu = Nu/length(n) - n * (dot(Nu,n)/pow(dot(n,n), 1.5));
    Nv = Nv/length(n) - n * (dot(Nv,n)/pow(dot(n,n), 1.5));

    output.Nu = Nu;
    output.Nv = Nv;
#else
    Univar4x4(UV.y, B, D);

    for (int k=0; k<4; ++k) {
        position += B[k] * BUCP[k];
        uTangent += B[k] * DUCP[k];
        vTangent += D[k] * BUCP[k];
    }
    int level = patch[0].patchCoord.z;
    uTangent *= 3 * level;
    vTangent *= 3 * level;

    float3 normal = normalize(cross(uTangent, vTangent));
#endif

    output.position = mul(OsdModelViewMatrix(), float4(position, 1.0f));
    output.normal = mul(OsdModelViewMatrix(), float4(normal, 0.0f)).xyz;
    output.tangent = mul(OsdModelViewMatrix(), float4(uTangent, 0.0f)).xyz;
    output.bitangent = mul(OsdModelViewMatrix(), float4(vTangent, 0.0f)).xyz;

    output.patchCoord = OsdInterpolatePatchCoord(UV, patch[0].patchCoord);

    OSD_DISPLACEMENT_CALLBACK;

    output.positionOut = mul(OsdProjectionMatrix(), output.position);
    output.edgeDistance = 0;
}
