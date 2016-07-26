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

#define SHADING_TYPE_MATERIAL 0
#define SHADING_TYPE_PATCH 1
#define SHADING_TYPE_NORMAL 2
#define SHADING_TYPE_PATCH_COORD 3


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
        
#if SHADING_TYPE == SHADING_TYPE_PATCH || SHADING_TYPE == SHADING_TYPE_PATCH_COORD
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
    float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 0
    float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 1
    float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 2
    float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 3
    float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 4
    
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
    if (edgeCount == 2) {
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
#endif
    
    int pattern = popcount(OsdGetPatchTransitionMask(patchParam));
    
    return patchColors[6*patchType + pattern];
}

#if OSD_IS_ADAPTIVE
#if USE_STAGE_IN
#if OSD_PATCH_REGULAR
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

struct PatchInput
{
    patch_control_point<ControlPoint> cv;
#if !USE_PTVS_FACTORS
    float4 tessOuterLo [[attribute(5)]];
    float4 tessOuterHi [[attribute(6)]];
#endif
    int3 patchParam [[attribute(10)]];
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

struct PatchInput
{
    patch_control_point<ControlPoint> cv;
    int3 patchParam [[attribute(10)]];
};
#elif OSD_PATCH_GREGORY_BASIS
struct ControlPoint
{
    float3 position [[attribute(0)]];
};

struct PatchInput
{
    patch_control_point<ControlPoint> cv;
    int3 patchParam [[attribute(10)]];
};
#endif
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
    device MTLQuadTessellationFactorsHalf* quadTessellationFactors [[buffer(QUAD_TESSFACTORS_INDEX)]]
#if OSD_USE_PATCH_INDEX_BUFFER
    ,device unsigned* patchIndex [[buffer(OSD_PATCH_INDEX_BUFFER_INDEX)]]
    ,device MTLDrawPatchIndirectArguments* drawIndirectCommands [[buffer(OSD_DRAWINDIRECT_BUFFER_INDEX)]]
#endif
)
{

    //----------------------------------------------------------
    // OSD Kernel Setup
    //----------------------------------------------------------

    //Contains the shared patchParam value used by all threads that act upon a single patch
    //the .z (sharpness) field is set to -1 (NAN) if that patch should be culled to signal other threads to return.
    threadgroup int3 patchParam[PATCHES_PER_THREADGROUP];

    threadgroup PatchVertexType patchVertices[PATCHES_PER_THREADGROUP * CONTROL_POINTS_PER_PATCH];

    const auto real_threadgroup = thread_position_in_grid / REAL_THREADGROUP_DIVISOR;
    const auto subthreadgroup_in_threadgroup = thread_position_in_threadgroup / REAL_THREADGROUP_DIVISOR;
    const auto real_thread_in_threadgroup = thread_position_in_threadgroup & (REAL_THREADGROUP_DIVISOR - 1);

#if NEEDS_BARRIER
    const auto validThread = thread_position_in_grid * CONTROL_POINTS_PER_THREAD < osdBuffers.kernelExecutionLimit;
#else
    const auto validThread = true;
    if(thread_position_in_grid * CONTROL_POINTS_PER_THREAD >= osdBuffers.kernelExecutionLimit)
        return;
#endif

    //----------------------------------------------------------
    // OSD Vertex Transform
    //----------------------------------------------------------
    if(validThread)
    {
        patchParam[subthreadgroup_in_threadgroup] = OsdGetPatchParam(real_threadgroup, osdBuffers.patchParamBuffer);
        
        for(unsigned threadOffset = 0; threadOffset < CONTROL_POINTS_PER_THREAD; threadOffset++)
        {
            const auto vertexId = osdBuffers.indexBuffer[(thread_position_in_grid * CONTROL_POINTS_PER_THREAD + threadOffset) * IndexLookupStride];
            const auto v = osdBuffers.vertexBuffer[vertexId];

            threadgroup auto& patchVertex = patchVertices[thread_position_in_threadgroup * CONTROL_POINTS_PER_THREAD + threadOffset];

            //----------------------------------------------------------
            // User Vertex Transform
            //----------------------------------------------------------

            OsdComputePerVertex(float4(v.position,1), patchVertex, vertexId, frameConsts.ModelViewProjectionMatrix, osdBuffers);
        }
    }

#if NEEDS_BARRIER
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif


    //----------------------------------------------------------
    // OSD Patch Cull
    //----------------------------------------------------------
    if(validThread)
    {
#if PATCHES_PER_THREADGROUP > 1
        auto patch = patchVertices + subthreadgroup_in_threadgroup * CONTROL_POINTS_PER_THREAD * CONTROL_POINTS_PER_PATCH;
#else
        //Small optimization for the '1 patch per threadgroup' case
        auto patch = patchVertices;
#endif

        if(!OsdCullPerPatchVertex(patch, frameConsts.ModelViewMatrix))
        {
#if !OSD_USE_PATCH_INDEX_BUFFER
            quadTessellationFactors[real_threadgroup].edgeTessellationFactor[0] = 0.0h;
            quadTessellationFactors[real_threadgroup].edgeTessellationFactor[1] = 0.0h;
            quadTessellationFactors[real_threadgroup].edgeTessellationFactor[2] = 0.0h;
            quadTessellationFactors[real_threadgroup].edgeTessellationFactor[3] = 0.0h;
            quadTessellationFactors[real_threadgroup].insideTessellationFactor[0] = 0.0h;
            quadTessellationFactors[real_threadgroup].insideTessellationFactor[1] = 0.0h;
#endif

            patchParam[subthreadgroup_in_threadgroup].z = -1;
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
    if(validThread && patchParam[subthreadgroup_in_threadgroup].z != -1)
    {
        for(unsigned threadOffset = 0; threadOffset < CONTROL_POINTS_PER_THREAD; threadOffset++)
        {
            OsdComputePerPatchVertex(
                patchParam[subthreadgroup_in_threadgroup],
                real_thread_in_threadgroup * CONTROL_POINTS_PER_THREAD + threadOffset,
                real_threadgroup,
                thread_position_in_grid * CONTROL_POINTS_PER_THREAD + threadOffset,
                patchVertices + subthreadgroup_in_threadgroup * CONTROL_POINTS_PER_PATCH,
                osdBuffers
                );
        }
    }

#if NEEDS_BARRIER
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
#endif

    //----------------------------------------------------------
    // OSD Tessellation Factors 
    //----------------------------------------------------------
    if(validThread && real_thread_in_threadgroup == 0)
    {

#if OSD_USE_PATCH_INDEX_BUFFER
        const auto patchId = atomic_fetch_add_explicit((device atomic_uint*)&drawIndirectCommands->patchCount, 1, memory_order_relaxed);
        patchIndex[patchId] = real_threadgroup;
#else
        const auto patchId = real_threadgroup;
#endif

        OsdComputePerPatchFactors(
            patchParam[subthreadgroup_in_threadgroup],
            frameConsts.TessLevel,
            real_threadgroup,
            frameConsts.ProjectionMatrix,
            frameConsts.ModelViewMatrix,
            osdBuffers,
            patchVertices + subthreadgroup_in_threadgroup * CONTROL_POINTS_PER_PATCH,
            quadTessellationFactors[patchId]
        );
    }
}

[[patch(quad, VERTEX_CONTROL_POINTS_PER_PATCH)]]
vertex OutputVertex vertex_main(
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
#if USE_STAGE_IN
    const PatchInput patchInput [[stage_in]],
#else
    const OsdVertexBufferSet patchInput,
#endif
    float2 position_in_patch [[position_in_patch]],
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
#if SHADING_TYPE == SHADING_TYPE_PATCH
#if OSD_PATCH_ENABLE_SINGLE_CREASE
    out.patchColor = getAdaptivePatchColor(patchParam, patchVertex.vSegments).xyz;
#else
    out.patchColor = getAdaptivePatchColor(patchParam).xyz;
#endif
#elif SHADING_TYPE == SHADING_TYPE_NORMAL
#elif SHADING_TYPE == SHADING_TYPE_PATCH_COORD
    out.patchColor = patchVertex.patchCoord.xyz;
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
    const device OsdPerPatchVertexBezier* osdPerPatchVertexBezier [[buffer(OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX)]],
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

#if OSD_PATCH_GREGORY_BASIS || OSD_PATCH_GREGORY_BOUNDARY || OSD_PATCH_GREGORY
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
#if OSD_PATCH_GREGORY_BASIS
    const device unsigned* indicesBuffer [[buffer(INDICES_BUFFER_INDEX)]],
    const device PackedInputVertex* vertexBuffer [[buffer(VERTEX_BUFFER_INDEX)]],
#else
    const device PackedInputVertex* vertexBuffer [[buffer(OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX)]],
#endif
    const constant PerFrameConstants& frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    uint vertex_id [[vertex_id]]
    )
{
    const auto idx_size = sizeof(GregoryBasisControlLineIndices) / sizeof(GregoryBasisControlLineIndices[0]);
    const auto idx = vertex_id % idx_size;
    const auto patch_id = vertex_id / idx_size;   

#if OSD_PATCH_GREGORY_BASIS
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
#else
    float3 p0 = vertexBuffer[indicesBuffer[primID * 3 + 0]].position;
    float3 p1 = vertexBuffer[indicesBuffer[primID * 3 + 1]].position;
    float3 p2 = vertexBuffer[indicesBuffer[primID * 3 + 2]].position;
    float3 position = vertexBuffer[indicesBuffer[vertex_id]].position;
#endif

    float3 normal = normalize(cross(p2 - p1, p0 - p1));

    
    OutputVertex out;
    out.position = (frameConsts.ModelViewMatrix * float4(position, 1.0)).xyz;
    out.positionOut = frameConsts.ModelViewProjectionMatrix * float4(position, 1.0);
    out.normal = (frameConsts.ModelViewMatrix * float4(normal,0.0)).xyz;
    
#if SHADING_TYPE == SHADING_TYPE_PATCH || SHADING_TYPE == SHADING_TYPE_PATCH_COORD
    out.patchColor = out.normal;
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
#elif SHADING_TYPE == SHADING_TYPE_PATCH
    const float3 diffuseColor = in.patchColor;
#else
#endif
#if SHADING_TYPE == SHADING_TYPE_NORMAL
    color.xyz = normalize(in.normal) * 0.5 + 0.5;
#elif SHADING_TYPE == SHADING_TYPE_PATCH_COORD
    color.xy = in.patchColor.xy;
    color.z = 0;
#else
    color.xyz = lighting(diffuseColor, lightData, in.position, normalize(in.normal));
#endif
    //    color.xyz = pow(color.xyz, 2.2);
    color.w = 1;
    return max(color,shade);
}
