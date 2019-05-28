#line 0 "examples/mtlViewer/mtlViewer.metal"

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

#include <metal_stdlib>
using namespace metal;

#if OSD_IS_ADAPTIVE
static_assert(!OSD_ENABLE_SCREENSPACE_TESSELLATION || !USE_PTVS_FACTORS, "USE_PTVS_FACTORS cannot be enabled if OSD_ENABLE_SCREENSPACE_TESSELLATION is enabled");
#endif

#define SHADING_TYPE_MATERIAL 0
#define SHADING_TYPE_FACE_VARYING_COLOR 1
#define SHADING_TYPE_PATCH_TYPE 2
#define SHADING_TYPE_PATCH_DEPTH 3
#define SHADING_TYPE_PATCH_COORD 4
#define SHADING_TYPE_NORMAL 5

struct PerFrameConstants {
    float4x4 ModelViewMatrix;
    float4x4 ProjectionMatrix;
    float4x4 ModelViewProjectionMatrix;
    float4x4 ModelViewInverseMatrix;
    float TessLevel;
};

struct OutputVertex {
    float4 positionOut [[position]];
    float3 position;
    float3 normal;
#if SHADING_TYPE == SHADING_TYPE_PATCH_TYPE || SHADING_TYPE == SHADING_TYPE_PATCH_DEPTH || SHADING_TYPE == SHADING_TYPE_PATCH_COORD || SHADING_TYPE_FACE_VARYING_COLOR
    float3 patchColor;
#endif
};

struct SolidColorVertex {
    float4 positionOut [[position]];

    half4 getColor() const {
        return unpack_unorm4x8_to_half(_color);
    }

    void setColor(half4 color) {
        _color = pack_half_to_unorm4x8(color);
    }

private:
    uint _color [[flat, user(color)]];
};

struct PackedInputVertex {
    packed_float3 position;
};

struct Light {
    float3 Position;
    float3 ambient;
    float3 diffuse;
    float3 specular;
};

float3 lighting(float3 diffuseColor, const constant Light* lightData, float3 eyePos, float3 eyeN)
{

    float3 color(0);
    for(int i = 0; i < 2; i++)
    {
        const auto l = lightData[i].Position;
        const auto h = normalize(l + float3(0,0,1));
        const auto d = max(0.0, dot(eyeN, l));
        const auto s = powr(max(0.0, dot(eyeN, h)), 500.0f);

        color += lightData[i].ambient
        + d * lightData[i].diffuse * diffuseColor
        + s * lightData[i].specular;
    }

    return color;
}


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
getAdaptivePatchColor(int3 patchParam
#if OSD_PATCH_ENABLE_SINGLE_CREASE
                      , float2 vSegments
#else
#endif
                      )
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
    // check this after boundary/corner since single crease patch also has edgeCount.
    if (vSegments.y > 0) {
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

float4
getAdaptiveDepthColor(int3 patchParam)
{
    //  Represent depth with repeating cycle of four colors:
    const float4 depthColors[4] = {
        float4(0.0f,  0.5f,  0.5f,  1.0f),
        float4(1.0f,  1.0f,  1.0f,  1.0f),
        float4(0.0f,  1.0f,  1.0f,  1.0f),
        float4(0.5f,  1.0f,  0.5f,  1.0f)
    };
    return depthColors[OsdGetPatchRefinementLevel(patchParam) & 3];
}

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


//----------------------------------------------------------
// OSD Kernel
//----------------------------------------------------------
//The user of OSD should define this kernel which serves as the landing point for all patch computation
//This compute function should just be copied and pasted, modifying the section under "User Vertex Transform"
//Or the entire function may be moddified as needed (for example to add a patch index buffer)
kernel void compute_main(
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    unsigned thread_position_in_grid [[thread_position_in_grid]],
    unsigned thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    unsigned threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    OsdPatchParamBufferSet osdBuffers, //This struct contains all of the buffers needed by OSD
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

#if SHADING_TYPE == SHADING_TYPE_FACE_VARYING_COLOR
float3
interpolateFaceVaryingColor(
        int                       patch_id,
        float2                    uv,
        const device float*       fvarData,
        const device int*         fvarIndices,
        const device packed_int3* fvarPatchParams,
        const constant int*       fvarPatchArrays)
{
    OsdPatchArray fvarPatchArray = OsdPatchArrayInit(
        fvarPatchArrays[0],
        fvarPatchArrays[1],
        fvarPatchArrays[2],
        fvarPatchArrays[3],
        fvarPatchArrays[4],
        fvarPatchArrays[5]);
    OsdPatchParam fvarParam = OsdPatchParamInit(
        fvarPatchParams[patch_id][0],
        fvarPatchParams[patch_id][1],
        fvarPatchParams[patch_id][2]);

    int fvarPatchType = OsdPatchParamIsRegular(fvarParam)
                        ? fvarPatchArray.regDesc
                        : fvarPatchArray.desc;

    float wP[20], wDu[20], wDv[20], wDuu[20], wDuv[20], wDvv[20];
    int numPoints = OsdEvaluatePatchBasisNormalized(fvarPatchType, fvarParam,
                uv.x, uv.y, wP, wDu, wDv, wDuu, wDuv, wDvv);

    int primOffset = patch_id * fvarPatchArray.stride;

    float2 interpUV = float2(0);
    for (int i = 0; i < numPoints; ++i) {
        int index = fvarIndices[primOffset + i] * 2 /* OSD_FVAR_WIDTH */ + 0 /* fvarOffset */;
        float2 cv = float2(fvarData[index + 0], fvarData[index + 1]);
        interpUV += wP[i] * cv;
    }

    return float3(interpUV, 0);
}
#endif

#if OSD_PATCH_REGULAR || OSD_PATCH_GREGORY_BASIS || OSD_PATCH_GREGORY || OSD_PATCH_GREGORY_BOUNDARY
[[patch(quad, VERTEX_CONTROL_POINTS_PER_PATCH)]]
#elif OSD_PATCH_BOX_SPLINE_TRIANGLE || OSD_PATCH_GREGORY_TRIANGLE
[[patch(triangle, VERTEX_CONTROL_POINTS_PER_PATCH)]]
#endif
vertex OutputVertex vertex_main(
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
#if USE_STAGE_IN
    const PatchInput patchInput [[stage_in]],
#else
    const OsdVertexBufferSet patchInput,
#endif
    const device float*       osdFaceVaryingData        [[buffer(OSD_FVAR_DATA_BUFFER_INDEX)]],
    const device int*         osdFaceVaryingIndices     [[buffer(OSD_FVAR_INDICES_BUFFER_INDEX)]],
    const device packed_int3* osdFaceVaryingPatchParams [[buffer(OSD_FVAR_PATCHPARAM_BUFFER_INDEX)]],
    const constant int*       osdFaceVaryingPatchArrays [[buffer(OSD_FVAR_PATCH_ARRAYS_BUFFER_INDEX)]],
#if OSD_PATCH_REGULAR || OSD_PATCH_GREGORY_BASIS || OSD_PATCH_GREGORY || OSD_PATCH_GREGORY_BOUNDARY
    float2 position_in_patch [[position_in_patch]],
#elif OSD_PATCH_BOX_SPLINE_TRIANGLE || OSD_PATCH_GREGORY_TRIANGLE
    float3 position_in_patch [[position_in_patch]],
#endif
    uint patch_id [[patch_id]]
    )
{
    OutputVertex out;

#if USE_STAGE_IN
    int3 patchParam = patchInput.patchParam;
#else
    int3 patchParam = patchInput.patchParamBuffer[patch_id];
#endif

    int refinementLevel = OsdGetPatchRefinementLevel(patchParam);
    float tessLevel = min(frameConsts.TessLevel, (float)OSD_MAX_TESS_LEVEL) /
        exp2((float)refinementLevel - 1);

    auto patchVertex = OsdComputePatch(tessLevel, position_in_patch, patch_id, patchInput);

    out.position = (frameConsts.ModelViewMatrix * float4(patchVertex.position, 1.0f)).xyz;
    out.positionOut = frameConsts.ModelViewProjectionMatrix * float4(patchVertex.position, 1.0f);

    out.normal = mul(frameConsts.ModelViewMatrix, patchVertex.normal);
#if SHADING_TYPE == SHADING_TYPE_PATCH_TYPE
#if OSD_PATCH_ENABLE_SINGLE_CREASE
    out.patchColor = getAdaptivePatchColor(patchParam, patchVertex.vSegments).xyz;
#else
    out.patchColor = getAdaptivePatchColor(patchParam).xyz;
#endif
#elif SHADING_TYPE == SHADING_TYPE_PATCH_DEPTH
    out.patchColor = getAdaptiveDepthColor(patchParam).xyz;
#elif SHADING_TYPE == SHADING_TYPE_NORMAL
#elif SHADING_TYPE == SHADING_TYPE_PATCH_COORD
    out.patchColor = patchVertex.patchCoord.xyz;
#elif SHADING_TYPE == SHADING_TYPE_FACE_VARYING_COLOR
    out.patchColor = interpolateFaceVaryingColor(
        patch_id,
        patchVertex.tessCoord.xy,
        osdFaceVaryingData,
        osdFaceVaryingIndices,
        osdFaceVaryingPatchParams,
        osdFaceVaryingPatchArrays);
#endif

    return out;
}
#endif

#if OSD_PATCH_REGULAR
const constant unsigned BSplineControlLineIndices[] = {
    0, 1, //Outer lines
    1, 2,
    2, 3,
    3, 7,
    7, 11,
    11, 15,
    15, 14,
    14, 13,
    13, 12,
    12, 8,
    8, 4,
    4, 0,

    //Inner lines
    5, 6,
    6, 10,
    10, 9,
    9, 5,

    //TL edge lines
    1, 5,
    4, 5,

    //TR edge lines
    2, 6,
    6, 7,

    //BL edge lines
    8, 9,
    9, 13,

    //BR edge lines
    10, 14,
    10, 11
};

vertex SolidColorVertex vertex_lines(
    const device unsigned* indicesBuffer [[buffer(INDICES_BUFFER_INDEX)]],
    const device OsdPerPatchVertexBezier* osdPerPatchVertexBezier [[buffer(OSD_PERPATCHVERTEX_BUFFER_INDEX)]],
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    uint vertex_id [[vertex_id]]
    )
{
    const auto idx_size = sizeof(BSplineControlLineIndices) / sizeof(BSplineControlLineIndices[0]);
    const auto idx = vertex_id % idx_size;
    const auto patch_id = vertex_id / idx_size;

    const auto in = osdPerPatchVertexBezier[patch_id * VERTEX_CONTROL_POINTS_PER_PATCH + BSplineControlLineIndices[idx]];

    SolidColorVertex out;

    out.positionOut = frameConsts.ModelViewProjectionMatrix * float4(in.P, 1.0);
    out.positionOut.z -= 0.001;

    if(idx > 22) {
        out.setColor(half4(0,1,0,1));
    }
    else
    {
        out.setColor(half4(1,0,0,1));
    }

    return out;
}
#endif

#if OSD_PATCH_GREGORY_BASIS || OSD_PATCH_GREGORY || OSD_PATCH_GREGORY_BOUNDARY
const constant uint GregoryBasisControlLineIndices[] = {
    //Outer Edge
    0, 2,
    2, 16,
    16, 15,
    15, 17,
    17, 11,
    11, 10,
    10, 12,
    12, 6,
    6, 5,
    5, 7,
    7, 1,
    1, 0,

    //Outside-Inside Edges
    1, 3,
    2, 4,
    16, 18,
    17, 19,
    11, 13,
    12, 14,
    6, 8,
    7, 9,

    //Inner Edge
    3, 4,
    4, 18,
    18, 19,
    19, 13,
    13, 14,
    14, 8,
    8, 9,
    9, 3,
};


vertex SolidColorVertex vertex_lines(
#ifdef OSD_PATCH_GREGORY_BASIS
    const device unsigned* indicesBuffer [[buffer(INDICES_BUFFER_INDEX)]],
    const device PackedInputVertex* vertexBuffer [[buffer(VERTEX_BUFFER_INDEX)]],
#else
    const device PackedInputVertex* vertexBuffer [[buffer(OSD_PERPATCHVERTEX_BUFFER_INDEX)]],
#endif
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    uint vertex_id [[vertex_id]]
    )
{
    const auto idx_size = sizeof(GregoryBasisControlLineIndices) / sizeof(GregoryBasisControlLineIndices[0]);
    const auto idx = vertex_id % idx_size;
    const auto patch_id = vertex_id / idx_size;

#ifdef OSD_PATCH_GREGORY_BASIS
    const auto in = vertexBuffer[indicesBuffer[patch_id * VERTEX_CONTROL_POINTS_PER_PATCH + GregoryBasisControlLineIndices[idx]]];
#else
    const auto in = vertexBuffer[patch_id * 20 + GregoryBasisControlLineIndices[idx]];
#endif
    SolidColorVertex out;

    out.positionOut = frameConsts.ModelViewProjectionMatrix * float4(in.position, 1.0);
    out.positionOut.z -= 0.001;

    if(idx > 22) {
        out.setColor(half4(0,1,0,1));
    }
    else
    {
        out.setColor(half4(1,0,0,1));
    }

    return out;
}
#endif

#if OSD_PATCH_QUADS || OSD_PATCH_TRIANGLES

#if OSD_PATCH_QUADS
const constant uint triangleIdx[6] = {
    0, 2, 1, 3, 2, 0
};
#endif

vertex OutputVertex vertex_main(
    device unsigned* indicesBuffer [[buffer(INDICES_BUFFER_INDEX)]],
    device PackedInputVertex* vertexBuffer [[buffer(VERTEX_BUFFER_INDEX)]],
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    const device float2* osdFaceVaryingData[[buffer(OSD_FVAR_DATA_BUFFER_INDEX)]],
    const device int* osdFaceVaryingIndices[[buffer(OSD_FVAR_INDICES_BUFFER_INDEX)]],
    uint vertex_id [[vertex_id]]
    )
{
#if OSD_PATCH_QUADS
    const auto quadId = vertex_id / 6;
#else
    const auto primID = vertex_id / 3;
#endif

#if OSD_PATCH_QUADS
    float3 p0 = vertexBuffer[indicesBuffer[quadId * 4 + 0]].position;
    float3 p1 = vertexBuffer[indicesBuffer[quadId * 4 + 1]].position;
    float3 p2 = vertexBuffer[indicesBuffer[quadId * 4 + 2]].position;
    float3 position = vertexBuffer[indicesBuffer[quadId * 4 + triangleIdx[vertex_id % 6]]].position;
    float2 uv = osdFaceVaryingData[osdFaceVaryingIndices[quadId * 4 + triangleIdx[vertex_id % 6]]].xy;
#else
    float3 p0 = vertexBuffer[indicesBuffer[primID * 3 + 0]].position;
    float3 p1 = vertexBuffer[indicesBuffer[primID * 3 + 1]].position;
    float3 p2 = vertexBuffer[indicesBuffer[primID * 3 + 2]].position;
    float3 position = vertexBuffer[indicesBuffer[vertex_id]].position;
    float2 uv = osdFaceVaryingData[osdFaceVaryingIndices[vertex_id]].xy;
#endif

    float3 normal = normalize(cross(p2 - p1, p0 - p1));


    OutputVertex out;
    out.position = (frameConsts.ModelViewMatrix * float4(position, 1.0)).xyz;
    out.positionOut = frameConsts.ModelViewProjectionMatrix * float4(position, 1.0);
    out.normal = (frameConsts.ModelViewMatrix * float4(normal, 0.0)).xyz;

#if SHADING_TYPE == SHADING_TYPE_PATCH_TYPE || SHADING_TYPE == SHADING_TYPE_PATCH_DEPTH || SHADING_TYPE == SHADING_TYPE_PATCH_COORD
    out.patchColor = out.normal;
#elif SHADING_TYPE == SHADING_TYPE_FACE_VARYING_COLOR
    out.patchColor.rg = uv;
#endif

    return out;
}

vertex SolidColorVertex vertex_lines(
    device unsigned* indicesBuffer [[buffer(INDICES_BUFFER_INDEX)]],
    device PackedInputVertex* vertexBuffer [[buffer(VERTEX_BUFFER_INDEX)]],
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    uint vertex_id [[vertex_id]]
    )
{
#if OSD_PATCH_QUADS
    const auto quadId = vertex_id / 6;
#else
    const auto primID = vertex_id / 3;
#endif

#if OSD_PATCH_QUADS
    float3 position = vertexBuffer[indicesBuffer[quadId * 4 + triangleIdx[vertex_id % 6]]].position;
#else
    float3 position = vertexBuffer[indicesBuffer[vertex_id]].position;
#endif

    SolidColorVertex out;
    out.positionOut = frameConsts.ModelViewProjectionMatrix * float4(position, 1.0);

    return out;
}
#endif

fragment half4 fragment_solidcolor(SolidColorVertex in [[stage_in]])
{
    return in.getColor();
}


fragment float4 fragment_main(OutputVertex in [[stage_in]],
                              const constant Light* lightData [[buffer(0)]],
                              const constant PerFrameConstants& frameConsts [[buffer(1)]],
                              const constant float4& shade [[buffer(2)]])
{
    float4 color;

#if SHADING_TYPE == SHADING_TYPE_MATERIAL
    const float3 diffuseColor = float3(0.4f, 0.4f, 0.8f);
#elif SHADING_TYPE == SHADING_TYPE_PATCH_TYPE || SHADING_TYPE == SHADING_TYPE_PATCH_DEPTH
    const float3 diffuseColor = in.patchColor;
#endif
#if SHADING_TYPE == SHADING_TYPE_NORMAL
    color.xyz = normalize(in.normal) * 0.5 + 0.5;
#elif SHADING_TYPE == SHADING_TYPE_PATCH_COORD || SHADING_TYPE == SHADING_TYPE_FACE_VARYING_COLOR
    color.xyz = lighting(1.0, lightData, in.position, normalize(in.normal));
    int checker = int(floor(20*in.patchColor.r)+floor(20*in.patchColor.g))&1;
    color.xyz *= float3(in.patchColor.rg*checker, 1-checker);
    color.xyz = pow(color.xyz, 1/2.2);
#else
    color.xyz = lighting(diffuseColor, lightData, in.position, normalize(in.normal));
#endif
    color.w = 1;
    return max(color,shade);
}
