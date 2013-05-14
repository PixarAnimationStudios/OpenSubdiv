//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

//----------------------------------------------------------
// Patches.Coefficients
//----------------------------------------------------------

static float4x4 Q = {
    1.f/6.f, 2.f/3.f, 1.f/6.f, 0.f,
    0.f,     2.f/3.f, 1.f/3.f, 0.f,
    0.f,     1.f/3.f, 2.f/3.f, 0.f,
    0.f,     1.f/6.f, 2.f/3.f, 1.f/6.f
};

// Boundary
static float4x3 B = {
    1.0f,    0.0f,    0.0f,
    2.f/3.f, 1.f/3.f, 0.0f,
    1.f/3.f, 2.f/3.f, 0.0f,
    1.f/6.f, 2.f/3.f, 1.f/6.f
};

// Corner
static float4x4 R = {
    1.f/6.f, 2.f/3.f, 1.f/6.f, 0.0f,
    0.0f,    2.f/3.f, 1.f/3.f, 0.0f,
    0.0f,    1.f/3.f, 2.f/3.f, 0.0f,
    0.0f,    0.0f,    1.0f,    0.0f
};

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------

void vs_main_patches( in InputVertex input,
                      out HullVertex output )
{
    output.position = mul(ModelViewMatrix, input.position);
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(input.position);

#if OSD_NUM_VARYINGS > 0
    for (int i = 0; i< OSD_NUM_VARYINGS; ++i)
        output.varyings[i] = input.varyings[i];
#endif
}

//----------------------------------------------------------
// Patches.HullRegular
//----------------------------------------------------------

HS_CONSTANT_FUNC_OUT HSConstFunc(
    InputPatch<HullVertex, 9> patch,
    uint primitiveID : SV_PrimitiveID)
{
    HS_CONSTANT_FUNC_OUT output;
    int patchLevel = GetPatchLevel(primitiveID);

    OSD_PATCH_CULL(9);

#if OSD_ENABLE_SCREENSPACE_TESSELLATION
    output.tessLevelOuter[0] =
        TessAdaptive(patch[2].position.xyz, patch[5].position.xyz, patchLevel);

    output.tessLevelOuter[1] =
        TessAdaptive(patch[1].position.xyz, patch[2].position.xyz, patchLevel);

    output.tessLevelOuter[2] =
        TessAdaptive(patch[4].position.xyz, patch[5].position.xyz, patchLevel);

    output.tessLevelOuter[3] =
        TessAdaptive(patch[1].position.xyz, patch[4].position.xyz, patchLevel);

    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#else
    output.tessLevelInner[0] = GetTessLevel(patchLevel);
    output.tessLevelInner[1] = GetTessLevel(patchLevel);
    output.tessLevelOuter[0] = GetTessLevel(patchLevel);
    output.tessLevelOuter[1] = GetTessLevel(patchLevel);
    output.tessLevelOuter[2] = GetTessLevel(patchLevel);
    output.tessLevelOuter[3] = GetTessLevel(patchLevel);
#endif
    return output;
}

[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(16)]
[patchconstantfunc("HSConstFunc")]
HullVertex hs_main_patches(
    in InputPatch<HullVertex, 9> patch,
    uint primitiveID : SV_PrimitiveID,
    in uint ID : SV_OutputControlPointID )
{
    int i = ID/4;
    int j = ID%4;

    float3 H[3];
    for (int l=0; l<3; ++l) {
        H[l] = float3(0,0,0);
        for (int k=0; k<3; ++k) {
            H[l] += B[i][2-k] * patch[l*3 + k].position.xyz;
        }
    }

    float3 pos = float3(0,0,0);
    for (int k=0; k<3; ++k){
        pos += B[j][k] * H[k];
    }

    HullVertex output;
    output.position = float4(pos, 1.0);

    int patchLevel = GetPatchLevel(primitiveID);
    // +0.5 to avoid interpolation error of integer value
    output.patchCoord = float4(0, 0,
                               patchLevel+0.5,
                               primitiveID+LevelBase+0.5);

    OSD_COMPUTE_PTEX_COORD_HULL_SHADER;

    return output;
}

//----------------------------------------------------------
// Patches.DomainRegular
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

[domain("quad")]
void ds_main_patches(
    in HS_CONSTANT_FUNC_OUT input,
    in OutputPatch<HullVertex, 16> patch,
    in float2 uv : SV_DomainLocation,
    out OutputVertex output )
{
    float u = uv.x,
          v = uv.y;

    float B[4], D[4];

    Univar4x4(u, B, D);

    float3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] = float3(0,0,0);
        DUCP[i] = float3(0,0,0);

        for (int j=0; j<4; ++j) {
            float3 A = patch[4*i + j].position.xyz;

            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    float3 WorldPos  = float3(0,0,0);
    float3 Tangent   = float3(0,0,0);
    float3 BiTangent = float3(0,0,0);

    Univar4x4(v, B, D);

    for (int i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
    }

    float3 normal = -normalize(cross(BiTangent, Tangent));

    output.position = float4(WorldPos, 1.0f);
    output.normal = normal;
    output.tangent = -normalize(BiTangent);

    output.patchCoord = patch[0].patchCoord;
    output.patchCoord.xy = float2(u, v);

    OSD_COMPUTE_PTEX_COORD_DOMAIN_SHADER;

    OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(0);

    OSD_DISPLACEMENT_CALLBACK;

    output.positionOut = mul(ProjectionMatrix, float4(WorldPos, 1.0f));
}

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------

void vs_main( in InputVertex input,
              out OutputVertex output)
{
    output.positionOut = mul(ModelViewProjectionMatrix, input.position);
}

//----------------------------------------------------------
// Patches.PixelColor
//----------------------------------------------------------

cbuffer Data : register( b2 ) {
    float4 color;
};

void ps_main( in OutputVertex input,
              out float4 colorOut : SV_Target )
{
    colorOut = color;
}
