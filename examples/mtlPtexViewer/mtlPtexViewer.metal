#line 0 "examples/mtlPtexViewer/mtlPtexViewer.metal"
//
//   Copyright 2013-2019 Pixar
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

#include <metal_stdlib>
using namespace metal;

#if OSD_IS_ADAPTIVE
static_assert(!OSD_ENABLE_SCREENSPACE_TESSELLATION || !USE_PTVS_FACTORS, "USE_PTVS_FACTORS cannot be enabled if OSD_ENABLE_SCREENSPACE_TESSELLATION is enabled");
#endif

struct Config {
    float displacementScale;
    float mipmapBias;
};

struct PerFrameConstants {
    float4x4 ModelViewMatrix;
    float4x4 ProjectionMatrix;
    float4x4 ModelViewProjectionMatrix;
    float4x4 ModelViewInverseMatrix;
    float TessLevel;
};

// ---------------------------------------------------------------------------

const constant float4 patchColors[] = {
    float4(1.0f,  1.0f,  1.0f,  1.0f),   // regular
    float4(0.0f,  1.0f,  1.0f,  1.0f),   // regular pattern 0
    float4(0.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 1
    float4(0.0f,  0.5f,  0.5f,  1.0f),   // regular pattern 2
    float4(0.5f,  0.0f,  1.0f,  1.0f),   // regular pattern 3
    float4(1.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 4

    float4(1.0f,  0.5f,  0.5f,  1.0f),   // single crease
    float4(1.0f,  0.70f,  0.6f,  1.0f),  // single crease pattern 0
    float4(1.0f,  0.65f,  0.6f,  1.0f),  // single crease pattern 1
    float4(1.0f,  0.60f,  0.6f,  1.0f),  // single crease pattern 2
    float4(1.0f,  0.55f,  0.6f,  1.0f),  // single crease pattern 3
    float4(1.0f,  0.50f,  0.6f,  1.0f),  // single crease pattern 4

    float4(0.8f,  0.0f,  0.0f,  1.0f),   // boundary
    float4(0.0f,  0.0f,  0.75f, 1.0f),   // boundary pattern 0
    float4(0.0f,  0.2f,  0.75f, 1.0f),   // boundary pattern 1
    float4(0.0f,  0.4f,  0.75f, 1.0f),   // boundary pattern 2
    float4(0.0f,  0.6f,  0.75f, 1.0f),   // boundary pattern 3
    float4(0.0f,  0.8f,  0.75f, 1.0f),   // boundary pattern 4

    float4(0.0f,  1.0f,  0.0f,  1.0f),   // corner
    float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 0
    float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 1
    float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 2
    float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 3
    float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 4

    float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
    float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
    float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
    float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
    float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
    float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory

    float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
    float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
    float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
    float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
    float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
    float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary

    float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
    float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
    float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
    float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
    float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
    float4(1.0f,  0.7f,  0.3f,  1.0f)    // gregory basis
};

float4
getAdaptivePatchColor(int3 patchParam, float sharpness)
{
    int patchType = 0;

    int edgeCount = popcount(OsdGetPatchBoundaryMask(patchParam));
    if (edgeCount == 1) {
        patchType = 2; // BOUNDARY
    }
    if (edgeCount > 1) {
        patchType = 3; // CORNER
    }

#if OSD_PATCH_ENABLE_SINGLE_CREASE
    if (sharpness > 0) {
        patchType = 1;
    }
#elif OSD_PATCH_GREGORY
    patchType = 4;
#elif OSD_PATCH_GREGORY_BOUNDARY
    patchType = 5;
#elif OSD_PATCH_GREGORY_BASIS
    patchType = 6;
#elif OSD_PATCH_GREGORY_TRIANGLE
    patchType = 6;
#endif

    int pattern = popcount(OsdGetPatchTransitionMask(patchParam));

    return patchColors[6*patchType + pattern];
}

#if    DISPLACEMENT_HW_BILINEAR        \
|| DISPLACEMENT_BILINEAR           \
|| DISPLACEMENT_BIQUADRATIC        \
|| NORMAL_HW_SCREENSPACE           \
|| NORMAL_SCREENSPACE              \
|| NORMAL_BIQUADRATIC              \
|| NORMAL_BIQUADRATIC_WG
#define USE_DISPLACEMENT_RESOURCES 1
#endif

#if DISPLACEMENT_HW_BILINEAR \
|| DISPLACEMENT_BILINEAR \
|| DISPLACEMENT_BIQUADRATIC

#define USE_DISPLACEMENT 1
#undef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK              \

float3 displacement(float3 position, float3 normal, float4 patchCoord, float mipmapBias, float displacementScale
#if USE_DISPLACEMENT_RESOURCES
                    ,texture2d_array<float, access::sample> textureDisplace_Data
                    ,device ushort* textureDisplace_Packing
#endif
                    )
{
#if DISPLACEMENT_HW_BILINEAR
    float disp = PtexLookupFast(patchCoord, mipmapBias,
                                textureDisplace_Data,
                                textureDisplace_Packing).x;
#elif DISPLACEMENT_BILINEAR
    float disp = PtexMipmapLookup(patchCoord, mipmapBias,
                                  textureDisplace_Data,
                                  textureDisplace_Packing).x;
#elif DISPLACEMENT_BIQUADRATIC
    float disp = PtexMipmapLookupQuadratic(patchCoord, mipmapBias,
                                           textureDisplace_Data,
                                           textureDisplace_Packing).x;
#else
    float disp(0);
#endif
    return position + disp*normal * displacementScale;
}
#endif

float4 GeneratePatchCoord(float2 uv, int3 patchParam)  // for non-adaptive
{
    return OsdInterpolatePatchCoord(uv, patchParam);
}

#if NORMAL_HW_SCREENSPACE || NORMAL_SCREENSPACE

float3
perturbNormalFromDisplacement(float3 position, float3 normal, float4 patchCoord, float mipmapBias
                              ,texture2d_array<float, access::sample> textureDisplace_Data
                              ,device ushort* textureDisplace_Packing
                              ,float displacementScale)
{
    // by Morten S. Mikkelsen
    // http://jbit.net/~sparky/sfgrad_bump/mm_sfgrad_bump.pdf
    // slightly modified for ptex guttering
    float3 vSigmaS = dfdx(position);
    float3 vSigmaT = dfdy(position);
    float3 vN = normal;
    float3 vR1 = cross(vSigmaT, vN);
    float3 vR2 = cross(vN, vSigmaS);
    float fDet = dot(vSigmaS, vR1);
#if 0
    // not work well with ptex
    float dBs = dfdx(disp);
    float dBt = dfdy(disp);
#else
    float2 texDx = dfdx(patchCoord.xy);
    float2 texDy = dfdy(patchCoord.xy);

    // limit forward differencing to the width of ptex gutter
    const float resolution = 128.0;
    float d = min(1.0f, (0.5/resolution)/max(length(texDx), length(texDy)));

    float4 STll = patchCoord;
    float4 STlr = patchCoord + d * float4(texDx.x, texDx.y, 0, 0);
    float4 STul = patchCoord + d * float4(texDy.x, texDy.y, 0, 0);
#if NORMAL_HW_SCREENSPACE
    float Hll = PtexLookupFast(STll, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hlr = PtexLookupFast(STlr, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hul = PtexLookupFast(STul, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
#elif NORMAL_SCREENSPACE
    float Hll = PtexMipmapLookup(STll, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hlr = PtexMipmapLookup(STlr, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hul = PtexMipmapLookup(STul, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
#endif
    float dBs = (Hlr - Hll)/d;
    float dBt = (Hul - Hll)/d;
#endif

    float3 vSurfGrad = sign(fDet) * (dBs * vR1 + dBt * vR2);
    return normalize(abs(fDet) * vN - vSurfGrad);
}
#endif // NORMAL_SCREENSPACE

// ---------------------------------------------------------------------------
//  Vertex Shader
// ---------------------------------------------------------------------------


struct FragmentInput
{
    float4 positionOut [[position]];
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float4 patchCoord;
#if COLOR_PATCHTYPE
    float4 patchColor;
#endif
#if OSD_COMPUTE_NORMAL_DERIVATIVES
    float3 Nu;
    float3 Nv;
#endif
};

#if OSD_IS_ADAPTIVE
#if USE_STAGE_IN
#if OSD_PATCH_REGULAR || OSD_PATCH_BOX_SPLINE_TRIANGLE
struct ControlPoint
{
    float3 P [[attribute(0)]];
#if OSD_PATCH_ENABLE_SINGLE_CREASE
    float3 P1 [[attribute(1)]];
    float3 P2 [[attribute(2)]];
#if !USE_PTVS_SHARPNESS
    float2 vSegments [[attribute(3)]];
#endif
#endif
};
#elif OSD_PATCH_GREGORY || OSD_PATCH_GREGORY_BOUNDARY
struct ControlPoint
{
    float3 P [[attribute(0)]];
    float3 Ep [[attribute(1)]];
    float3 Em [[attribute(2)]];
    float3 Fp [[attribute(3)]];
    float3 Fm [[attribute(4)]];
};
#elif OSD_PATCH_GREGORY_BASIS || OSD_PATCH_GREGORY_TRIANGLE
struct ControlPoint
{
    float3 position [[attribute(0)]];
};
#endif

struct PatchInput
{
    patch_control_point<ControlPoint> cv;
#if !USE_PTVS_FACTORS
    float4 tessOuterLo [[attribute(5)]];
    float4 tessOuterHi [[attribute(6)]];
#endif
    int3 patchParam [[attribute(10)]];
};
#endif


#if OSD_PATCH_REGULAR || OSD_PATCH_GREGORY_BASIS || OSD_PATCH_GREGORY || OSD_PATCH_GREGORY_BOUNDARY
typedef MTLQuadTessellationFactorsHalf PatchTessFactors;
#elif OSD_PATCH_BOX_SPLINE_TRIANGLE || OSD_PATCH_GREGORY_TRIANGLE
typedef MTLTriangleTessellationFactorsHalf PatchTessFactors;
#endif


kernel void compute_main(
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    unsigned thread_position_in_grid [[thread_position_in_grid]],
    unsigned thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    unsigned threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    OsdPatchParamBufferSet osdBuffers,
    device PatchTessFactors* patchTessellationFactors [[buffer(PATCH_TESSFACTORS_INDEX)]]
#if OSD_USE_PATCH_INDEX_BUFFER
    ,device unsigned* patchIndex [[buffer(OSD_PATCH_INDEX_BUFFER_INDEX)]]
    ,device MTLDrawPatchIndirectArguments* drawIndirectCommands [[buffer(OSD_DRAWINDIRECT_BUFFER_INDEX)]]
#endif
)
{
    //----------------------------------------------------------
    // OSD Kernel Setup
    //----------------------------------------------------------

    #define PATCHES_PER_THREADGROUP (THREADS_PER_THREADGROUP / THREADS_PER_PATCH)
    int const primitiveID = thread_position_in_grid / THREADS_PER_PATCH;
    int const primitiveIDInTG = thread_position_in_threadgroup / THREADS_PER_PATCH;
    int const vertexIndex = threadgroup_position_in_grid * PATCHES_PER_THREADGROUP * CONTROL_POINTS_PER_PATCH +
                            thread_position_in_threadgroup * CONTROL_POINTS_PER_THREAD;
    int const vertexIndexInTG = thread_position_in_threadgroup * CONTROL_POINTS_PER_THREAD;
    int const invocationID = (thread_position_in_threadgroup * VERTEX_CONTROL_POINTS_PER_THREAD) % (THREADS_PER_PATCH*VERTEX_CONTROL_POINTS_PER_THREAD);

    //Contains the shared patchParam value used by all threads that act upon a single patch
    //the .z (sharpness) field is set to -1 (NAN) if that patch should be culled to signal other threads to return.
    threadgroup int3 patchParam[PATCHES_PER_THREADGROUP];
    threadgroup PatchVertexType patchVertices[PATCHES_PER_THREADGROUP * CONTROL_POINTS_PER_PATCH];

    //----------------------------------------------------------
    // OSD Vertex Transform
    //----------------------------------------------------------
    {
        patchParam[primitiveIDInTG] = OsdGetPatchParam(primitiveID, osdBuffers.patchParamBuffer);

        for (unsigned threadOffset = 0; threadOffset < CONTROL_POINTS_PER_THREAD; ++threadOffset)
        {
            if (vertexIndexInTG + threadOffset < PATCHES_PER_THREADGROUP * CONTROL_POINTS_PER_PATCH)
            {
                const auto vertexId = osdBuffers.indexBuffer[(vertexIndex + threadOffset)];
                const auto v = osdBuffers.vertexBuffer[vertexId];

                threadgroup auto& patchVertex = patchVertices[vertexIndexInTG + threadOffset];

                //----------------------------------------------------------
                // User Vertex Transform
                //----------------------------------------------------------

                OsdComputePerVertex(float4(v.position,1), patchVertex, vertexId, frameConsts.ModelViewProjectionMatrix, osdBuffers);
            }
        }
    }

#if NEEDS_BARRIER
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif

    //----------------------------------------------------------
    // OSD Patch Cull
    //----------------------------------------------------------
    {
        auto patch = patchVertices + primitiveIDInTG * CONTROL_POINTS_PER_PATCH;

        if (!OsdCullPerPatchVertex(patch, frameConsts.ModelViewMatrix))
        {
#if !OSD_USE_PATCH_INDEX_BUFFER
#if OSD_PATCH_REGULAR || OSD_PATCH_GREGORY_BASIS || OSD_PATCH_GREGORY || OSD_PATCH_GREGORY_BOUNDARY
            patchTessellationFactors[primitiveID].edgeTessellationFactor[0] = 0.0h;
            patchTessellationFactors[primitiveID].edgeTessellationFactor[1] = 0.0h;
            patchTessellationFactors[primitiveID].edgeTessellationFactor[2] = 0.0h;
            patchTessellationFactors[primitiveID].edgeTessellationFactor[3] = 0.0h;
            patchTessellationFactors[primitiveID].insideTessellationFactor[0] = 0.0h;
            patchTessellationFactors[primitiveID].insideTessellationFactor[1] = 0.0h;
#elif OSD_PATCH_BOX_SPLINE_TRIANGLE || OSD_PATCH_GREGORY_TRIANGLE
            patchTessellationFactors[primitiveID].edgeTessellationFactor[0] = 0.0h;
            patchTessellationFactors[primitiveID].edgeTessellationFactor[1] = 0.0h;
            patchTessellationFactors[primitiveID].edgeTessellationFactor[2] = 0.0h;
            patchTessellationFactors[primitiveID].insideTessellationFactor = 0.0h;
#endif
#endif

            patchParam[primitiveIDInTG].z = -1;
#if !NEEDS_BARRIER
            return;
#endif
        }
    }

#if NEEDS_BARRIER
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif

    //----------------------------------------------------------
    // OSD Patch Compute
    //----------------------------------------------------------
    if (patchParam[primitiveIDInTG].z != -1)
    {
        for (unsigned threadOffset = 0; threadOffset < VERTEX_CONTROL_POINTS_PER_THREAD; ++threadOffset)
        {
            if (invocationID + threadOffset < VERTEX_CONTROL_POINTS_PER_PATCH)
            {
                OsdComputePerPatchVertex(
                    patchParam[primitiveIDInTG],
                    invocationID + threadOffset,
                    primitiveID,
                    invocationID + threadOffset + primitiveID * VERTEX_CONTROL_POINTS_PER_PATCH,
                    patchVertices + primitiveIDInTG * CONTROL_POINTS_PER_PATCH,
                    osdBuffers
                    );
            }
        }
    }

#if NEEDS_BARRIER
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
#endif

    //----------------------------------------------------------
    // OSD Tessellation Factors
    //----------------------------------------------------------
    if (invocationID == 0)
    {

#if OSD_USE_PATCH_INDEX_BUFFER
        const auto patchId = atomic_fetch_add_explicit((device atomic_uint*)&drawIndirectCommands->patchCount, 1, memory_order_relaxed);
        patchIndex[patchId] = primitiveID;
#else
        const auto patchId = primitiveID;
#endif

        OsdComputePerPatchFactors(
            patchParam[primitiveIDInTG],
            frameConsts.TessLevel,
            primitiveID,
            frameConsts.ProjectionMatrix,
            frameConsts.ModelViewMatrix,
            osdBuffers,
            patchVertices + primitiveIDInTG * CONTROL_POINTS_PER_PATCH,
            patchTessellationFactors[patchId]
            );
    }
}

[[patch(quad, VERTEX_CONTROL_POINTS_PER_PATCH)]]
vertex FragmentInput vertex_main(
    const constant Config& config [[buffer(CONFIG_BUFFER_INDEX)]],
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
#if USE_STAGE_IN
    const PatchInput patchInput [[stage_in]],
#else
    const OsdVertexBufferSet patchInput,
#endif
    float2 position_in_patch [[position_in_patch]],
    uint patch_id [[patch_id]]
#if USE_DISPLACEMENT_RESOURCES
    ,texture2d_array<float, access::sample> textureDisplace_Data [[texture(DISPLACEMENT_TEXTURE_INDEX)]]
    ,device ushort* textureDisplace_Packing [[buffer(DISPLACEMENT_BUFFER_INDEX)]]
#endif
    )
{
    FragmentInput out;

#if USE_STAGE_IN
    int3 patchParam = patchInput.patchParam;
#else
    int3 patchParam = patchInput.patchParamBuffer[patch_id];
#endif

    int refinementLevel = OsdGetPatchRefinementLevel(patchParam);
    float tessLevel = min(frameConsts.TessLevel, (float)OSD_MAX_TESS_LEVEL) /
        exp2((float)refinementLevel - 1);

    auto patchVertex = OsdComputePatch(tessLevel, position_in_patch, patch_id, patchInput);


#if USE_DISPLACEMENT
    float3 position = displacement(patchVertex.position,
                                   patchVertex.normal,
                                   patchVertex.patchCoord,
                                   config.mipmapBias,
                                   config.displacementScale
#if USE_DISPLACEMENT_RESOURCES
                                   ,textureDisplace_Data, textureDisplace_Packing
#endif
                                   );
#else
    float3 position = patchVertex.position;
#endif


    out.positionOut = mul(frameConsts.ModelViewProjectionMatrix, float4(position, 1));
    out.position = mul(frameConsts.ModelViewMatrix, float4(position,1)).xyz;
    out.normal = mul(frameConsts.ModelViewMatrix,float4(patchVertex.normal, 0)).xyz;
    out.tangent = mul(frameConsts.ModelViewMatrix,float4(patchVertex.tangent,0)).xyz;
    out.bitangent = mul(frameConsts.ModelViewMatrix,float4(patchVertex.bitangent,0)).xyz;
    out.patchCoord = patchVertex.patchCoord;
#if COLOR_PATCHTYPE
    out.patchColor = getAdaptivePatchColor(patchParam, OsdGetPatchSharpness(patchParam));
#endif
#if OSD_COMPUTE_NORMAL_DERIVATIVES
    out.Nu = mul(frameConsts.ModelViewMatrix, float4(patchVertex.Nu, 0)).xyz;
    out.Nv = mul(frameConsts.ModelViewMatrix, float4(patchVertex.Nv, 0)).xyz;
#endif
    return out;
}
#endif

const constant float VIEWPORT_SCALE = 1024.0; // XXXdyu

// ---------------------------------------------------------------------------
//  Lighting
// ---------------------------------------------------------------------------

#define NUM_LIGHTS 1

struct LightSource {
    float4 position;
    float4 ambient;
    float4 diffuse;
    float4 specular;
};

float4
lighting(float4 texColor, float3 Peye, float3 Neye, float occ, const constant LightSource (&lightSource)[NUM_LIGHTS])
{
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    float3 n = Neye;

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        float4 Plight = lightSource[i].position;
        float3 l = (Plight.w == 0.0)
        ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        float3 h = normalize(l + float3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 64.0f);

        color += (1.0 - occ) * ((lightSource[i].ambient +
                                 d * lightSource[i].diffuse) * texColor +
                                s * lightSource[i].specular);
    }

    color.a = 1.0;
    return color;
}

// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------

float4
edgeColor(float4 Cfill, float4 edgeDistance)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float d =
    min(edgeDistance[0], min(edgeDistance[1], edgeDistance[2]));
#endif
#ifdef PRIM_QUAD
    float d =
    min(min(edgeDistance[0], edgeDistance[1]),
        min(edgeDistance[2], edgeDistance[3]));
#endif
    float4 Cedge = float4(1.0, 1.0, 0.0, 1.0);
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    Cfill.rgb = lerp(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------


#if COLOR_PTEX_NEAREST ||     \
COLOR_PTEX_HW_BILINEAR || \
COLOR_PTEX_BILINEAR ||    \
COLOR_PTEX_BIQUADRATIC
#define USE_IMAGE_RESOURCES 1
#endif

#if USE_PTEX_OCCLUSION
#define USE_OCCLUSION_RESOURCES 1
#endif

#if USE_PTEX_SPECULAR
#define USE_SPECULAR_RESOURCES 1
#endif

fragment float4 fragment_main(
    FragmentInput input [[stage_in]]
#if USE_DISPLACEMENT_RESOURCES
        ,texture2d_array<float, access::sample> textureDisplace_Data [[texture(DISPLACEMENT_TEXTURE_INDEX)]]
        ,device ushort* textureDisplace_Packing [[buffer(DISPLACEMENT_BUFFER_INDEX)]]
#endif
#if USE_IMAGE_RESOURCES
        ,texture2d_array<float, access::sample> textureImage_Data [[texture(IMAGE_TEXTURE_INDEX)]]
        ,device ushort* textureImage_Packing [[buffer(IMAGE_BUFFER_INDEX)]]
#endif
#if USE_OCCLUSION_RESOURCES
        ,texture2d_array<float, access::sample> textureOcclusion_Data [[texture(OCCLUSION_TEXTURE_INDEX)]]
        ,device ushort* textureOcclusion_Packing [[buffer(OCCLUSION_BUFFER_INDEX)]]
#endif
#if USE_SPECULAR_RESOURCES
        ,texture2d_array<float, acess::read> textureSpecular_Data [[texture(SPECULAR_TEXTURE_INDEX)]]
        ,device ushort* textureSpecular_Packing [[buffer(SPECULAR_BUFFER_INDEX)]]
#endif
        ,const constant LightSource (&lightSource [[buffer(0)]]) [NUM_LIGHTS]
        ,const constant Config& config [[buffer(1)]]
        ,const constant float4& shade [[buffer(2)]]
        )
{
    const auto displacementScale = config.displacementScale;
    const auto mipmapBias = config.mipmapBias;
    float4 outColor;
    // ------------ normal ---------------
#if NORMAL_HW_SCREENSPACE || NORMAL_SCREENSPACE
    float3 normal = perturbNormalFromDisplacement(input.position.xyz,
                                                  input.normal,
                                                  input.patchCoord,
                                                  mipmapBias,
                                                  textureDisplace_Data,
                                                  textureDisplace_Packing,
                                                  displacementScale);
#elif NORMAL_BIQUADRATIC || NORMAL_BIQUADRATIC_WG
    float4 du, dv;
    float4 disp = PtexMipmapLookupQuadratic(du, dv, input.patchCoord,
                                            mipmapBias,
                                            textureDisplace_Data,
                                            textureDisplace_Packing);

    disp *= displacementScale;
    du *= displacementScale;
    dv *= displacementScale;

    float3 n = normalize(cross(input.tangent, input.bitangent));
    float3 tangent = input.tangent + n * du.x;
    float3 bitangent = input.bitangent + n * dv.x;

#if NORMAL_BIQUADRATIC_WG
    tangent += input.Nu * disp.x;
    bitangent += input.Nv * disp.x;
#endif

    float3 normal = normalize(cross(tangent, bitangent));
#else
    float3 normal = input.normal;
#endif

    // ------------ color ---------------
#if COLOR_PTEX_NEAREST
    float4 texColor = PtexLookupNearest(input.patchCoord,
                                        textureImage_Data,
                                        textureImage_Packing);
#elif COLOR_PTEX_HW_BILINEAR
    float4 texColor = PtexLookupFast(input.patchCoord,
                                     textureImage_Data,
                                     textureImage_Packing);
#elif COLOR_PTEX_BILINEAR
    float4 texColor = PtexMipmapLookup(input.patchCoord, mipmapBias,
                                       textureImage_Data,
                                       textureImage_Packing);
#elif COLOR_PTEX_BIQUADRATIC
    float4 texColor = PtexMipmapLookupQuadratic(input.patchCoord, mipmapBias,
                                                textureImage_Data,
                                                textureImage_Packing);
#elif COLOR_PATCHTYPE
    float4 texColor = lighting(float4(input.patchColor), input.position.xyz, normal, 0, lightSource);
    outColor = max(texColor, shade);
    return outColor;
#elif COLOR_PATCHCOORD
    float4 texColor = lighting(input.patchCoord, input.position.xyz, normal, 0, lightSource);
    outColor = max(texColor, shade);
    return outColor;
#elif COLOR_NORMAL
    float4 texColor = float4(normal.x, normal.y, normal.z, 1);
    outColor = max(texColor, shade);
    return outColor;
#else // COLOR_NONE
    float4 texColor = float4(0.5, 0.5, 0.5, 1);
#endif

    // ------------ occlusion ---------------

#if USE_PTEX_OCCLUSION
    float occ = PtexMipmapLookup(input.patchCoord, mipmapBias,
                                 textureOcclusion_Data,
                                 textureOcclusion_Packing).x;
#else
    float occ = 0.0;
#endif

    // ------------ specular ---------------

#if USE_PTEX_SPECULAR
    float specular = PtexMipmapLookup(input.patchCoord, mipmapBias,
                                      textureSpecular_Data,
                                      textureSpecular_Packing).x;
#else
    float specular = 1.0;
#endif
    // ------------ lighting ---------------
    float4 Cf = lighting(texColor, input.position.xyz, normal, occ, lightSource);

    // ------------ wireframe ---------------
    outColor = max(Cf, shade);
    return outColor;
}
