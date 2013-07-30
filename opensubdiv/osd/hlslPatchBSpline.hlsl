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

#ifdef OSD_TRANSITION_TRIANGLE_SUBPATCH
    #define HS_DOMAIN "tri"
#else
    #define HS_DOMAIN "quad"
#endif

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------

void vs_main_patches( in InputVertex input,
                      out HullVertex output )
{
    output.position = mul(ModelViewMatrix, input.position);
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

// Boundary / Corner
static float4x3 B = {
    1.f,     0.f,     0.f,
    4.f/6.f, 2.f/6.f, 0.f,
    2.f/6.f, 4.f/6.f, 0.f,
    1.f/6.f, 4.f/6.f, 1.f/6.f
};

#ifdef OSD_PATCH_TRANSITION
    HS_CONSTANT_TRANSITION_FUNC_OUT
#else
    HS_CONSTANT_FUNC_OUT
#endif
HSConstFunc(
    InputPatch<HullVertex, OSD_PATCH_INPUT_SIZE> patch,
    uint primitiveID : SV_PrimitiveID)
{
#ifdef OSD_PATCH_TRANSITION
    HS_CONSTANT_TRANSITION_FUNC_OUT output;
#else
    HS_CONSTANT_FUNC_OUT output;
#endif
    int patchLevel = GetPatchLevel(primitiveID);

#ifdef OSD_TRANSITION_TRIANGLE_SUBPATCH
    OSD_PATCH_CULL_TRIANGLE(OSD_PATCH_INPUT_SIZE);
#else
    OSD_PATCH_CULL(OSD_PATCH_INPUT_SIZE);
#endif

#ifdef OSD_PATCH_TRANSITION
    float3 cp[OSD_PATCH_INPUT_SIZE];
    for(int k = 0; k < OSD_PATCH_INPUT_SIZE; ++k) cp[k] = patch[k].position.xyz;
    SetTransitionTessLevels(output, cp, patchLevel);
#else
    #if defined OSD_PATCH_BOUNDARY
        const int p[4] = { 1, 2, 5, 6 };
    #elif defined OSD_PATCH_CORNER
        const int p[4] = { 1, 2, 4, 5 };
    #else
        const int p[4] = { 5, 6, 9, 10 };
    #endif

    #ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
        output.tessLevelOuter[0] = TessAdaptive(patch[p[0]].position.xyz, patch[p[2]].position.xyz);
        output.tessLevelOuter[1] = TessAdaptive(patch[p[0]].position.xyz, patch[p[1]].position.xyz);
        output.tessLevelOuter[2] = TessAdaptive(patch[p[1]].position.xyz, patch[p[3]].position.xyz);
        output.tessLevelOuter[3] = TessAdaptive(patch[p[2]].position.xyz, patch[p[3]].position.xyz);
        output.tessLevelInner[0] = max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] = max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #else
        output.tessLevelInner[0] = GetTessLevel(patchLevel);
        output.tessLevelInner[1] = GetTessLevel(patchLevel);
        output.tessLevelOuter[0] = GetTessLevel(patchLevel);
        output.tessLevelOuter[1] = GetTessLevel(patchLevel);
        output.tessLevelOuter[2] = GetTessLevel(patchLevel);
        output.tessLevelOuter[3] = GetTessLevel(patchLevel);
    #endif
#endif

    return output;
}

[domain(HS_DOMAIN)]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(16)]
[patchconstantfunc("HSConstFunc")]
HullVertex hs_main_patches(
    in InputPatch<HullVertex, OSD_PATCH_INPUT_SIZE> patch,
    uint primitiveID : SV_PrimitiveID,
    in uint ID : SV_OutputControlPointID )
{
    int i = ID%4;
    int j = ID/4;

#if defined OSD_PATCH_BOUNDARY
    float3 H[3];
    for (int l=0; l<3; ++l) {
        H[l] = float3(0,0,0);
        for (int k=0; k<4; ++k) {
            H[l] += Q[i][k] * patch[l*4 + k].position.xyz;
        }
    }

    float3 pos = float3(0,0,0);
    for (int k=0; k<3; ++k) {
        pos += B[j][k]*H[k];
    }

#elif defined OSD_PATCH_CORNER
    float3 H[3];
    for (int l=0; l<3; ++l) {
        H[l] = float3(0,0,0);
        for (int k=0; k<3; ++k) {
            H[l] += B[3-i][2-k] * patch[l*3 + k].position.xyz;
        }
    }

    float3 pos = float3(0,0,0);
    for (int k=0; k<3; ++k) {
        pos += B[j][k]*H[k];
    }

#else // not OSD_PATCH_BOUNDARY, not OSD_PATCH_CORNER
    float3 H[4];
    for (int l=0; l<4; ++l) {
        H[l] = float3(0,0,0);
        for(int k=0; k<4; ++k) {
            H[l] += Q[i][k] * patch[l*4 + k].position.xyz;
        }
    }

    float3 pos = float3(0,0,0);
    for (int k=0; k<4; ++k){
        pos += Q[j][k]*H[k];
    }

#endif

    HullVertex output;
    output.position = float4(pos, 1.0);

    int patchLevel = GetPatchLevel(primitiveID);

    // +0.5 to avoid interpolation error of integer value
    output.patchCoord = float4(0, 0,
                               patchLevel+0.5,
                               primitiveID+PrimitiveIdBase+0.5);

    OSD_COMPUTE_PTEX_COORD_HULL_SHADER;

    return output;
}

//----------------------------------------------------------
// Patches.DomainBSpline
//----------------------------------------------------------

// B-spline basis evaluation via deBoor pyramid...
void
EvalCubicBSpline(in float u, out float B[4], out float BU[4])
{
    float t = u;
    float s = 1.0 - u;

    float C0 =                     s * (0.5 * s);
    float C1 = t * (s + 0.5 * t) + s * (0.5 * s + t);
    float C2 = t * (    0.5 * t);

    B[0] =                                     1.f/3.f * s                * C0;
    B[1] = (2.f/3.f * s +           t) * C0 + (2.f/3.f * s + 1.f/3.f * t) * C1;
    B[2] = (1.f/3.f * s + 2.f/3.f * t) * C1 + (          s + 2.f/3.f * t) * C2;
    B[3] =                1.f/3.f * t  * C2;

    BU[0] =    - C0;
    BU[1] = C0 - C1;
    BU[2] = C1 - C2;
    BU[3] = C2;
}

void
Univar4x4(in float u, out float B[4], out float D[4])
{
    float t = u;
    float s = 1.0 - u;

    float A0 = s * s;
    float A1 = 2 * s * t;
    float A2 = t * t;

    B[0] = s * A0;
    B[1] = t * A0 + s * A1;
    B[2] = t * A1 + s * A2;
    B[3] = t * A2;

    D[0] =    - A0;
    D[1] = A0 - A1;
    D[2] = A1 - A2;
    D[3] = A2;
}

[domain(HS_DOMAIN)]
void ds_main_patches(
#ifdef OSD_PATCH_TRANSITION
    in HS_CONSTANT_TRANSITION_FUNC_OUT input,
#else
    in HS_CONSTANT_FUNC_OUT input,
#endif
    in OutputPatch<HullVertex, 16> patch,
#ifdef OSD_TRANSITION_TRIANGLE_SUBPATCH
    in float3 domainCoord : SV_DomainLocation,
#else
    in float2 domainCoord : SV_DomainLocation,
#endif
    out OutputVertex output )
{
#ifdef OSD_PATCH_TRANSITION
    float2 UV = GetTransitionSubpatchUV(domainCoord);
#else
    float2 UV = domainCoord;
#endif

    float B[4], D[4];

    Univar4x4(UV.x, B, D);

    float3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] = float3(0,0,0);
        DUCP[i] = float3(0,0,0);

        for (int j=0; j<4; ++j) {
#if OSD_TRANSITION_ROTATE == 1
            float3 A = patch[4*(3-j) + i].position.xyz;
#elif OSD_TRANSITION_ROTATE == 2
            float3 A = patch[4*(3-i) + (3-j)].position.xyz;
#elif OSD_TRANSITION_ROTATE == 3
            float3 A = patch[4*j + (3-i)].position.xyz;
#else // OSD_TRANSITION_ROTATE == 0, or non-transition patch
            float3 A = patch[4*i + j].position.xyz;
#endif
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    float3 WorldPos  = float3(0,0,0);
    float3 Tangent   = float3(0,0,0);
    float3 BiTangent = float3(0,0,0);

    Univar4x4(UV.y, B, D);

    for (int k=0; k<4; ++k) {
        WorldPos  += B[k] * BUCP[k];
        Tangent   += B[k] * DUCP[k];
        BiTangent += D[k] * BUCP[k];
    }

    float3 normal = normalize(cross(Tangent, BiTangent));

    output.position = float4(WorldPos, 1.0f);
    output.normal = normal;
    output.tangent = normalize(Tangent);

    output.patchCoord = patch[0].patchCoord;

#if OSD_TRANSITION_ROTATE == 1
    output.patchCoord.xy = float2(UV.y, 1.0-UV.x);
#elif OSD_TRANSITION_ROTATE == 2
    output.patchCoord.xy = float2(1.0-UV.x, 1.0-UV.y);
#elif OSD_TRANSITION_ROTATE == 3
    output.patchCoord.xy = float2(1.0-UV.y, UV.x);
#else // OSD_TRANNSITION_ROTATE == 0, or non-transition patch
    output.patchCoord.xy = float2(UV.x, UV.y);
#endif

    OSD_COMPUTE_PTEX_COORD_DOMAIN_SHADER;

    OSD_DISPLACEMENT_CALLBACK;

    output.positionOut = mul(ProjectionMatrix, float4(WorldPos, 1.0f));
}
