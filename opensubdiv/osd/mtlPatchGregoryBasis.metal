#line 0 "osd/mtlPatchGregoryBasis.metal"

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
// Patches.GregoryBasis.Hull
//----------------------------------------------------------

void OsdComputePerVertex(
	float4 position,
    threadgroup HullVertex& hullVertex,
    int vertexId,
    float4x4 modelViewProjectionMatrix,
    OsdPatchParamBufferSet osdBuffers
    )
{
	hullVertex.position = position;
#if OSD_ENABLE_PATCH_CULL
    float4 clipPos = mul(modelViewProjectionMatrix, position);    
    short3 clip0 = short3(clipPos.x < clipPos.w,                    
    clipPos.y < clipPos.w,                    
    clipPos.z < clipPos.w);                   
    short3 clip1 = short3(clipPos.x > -clipPos.w,                   
    clipPos.y > -clipPos.w,                   
    clipPos.z > -clipPos.w);                  
    hullVertex.clipFlag = short3(clip0) + 2*short3(clip1);              
#endif
}

//----------------------------------------------------------
// Patches.GregoryBasis.Factors
//----------------------------------------------------------

void OsdComputePerPatchFactors(
	int3 patchParam,
	float tessLevel,
	unsigned patchID,
	float4x4 projectionMatrix,
	float4x4 modelViewMatrix,
	OsdPatchParamBufferSet osdBuffer,
	threadgroup PatchVertexType* patchVertices,
	device MTLQuadTessellationFactorsHalf& quadFactors
	)
{
    float4 tessLevelOuter = float4(0,0,0,0);
    float2 tessLevelInner = float2(0,0);

	OsdGetTessLevels(
 		tessLevel, 
 		projectionMatrix, 
 		modelViewMatrix,
		patchVertices[0].position.xyz, 
		patchVertices[3].position.xyz, 
		patchVertices[2].position.xyz, 
		patchVertices[1].position.xyz,
		patchParam, 
		tessLevelOuter, 
		tessLevelInner
		);

    quadFactors.edgeTessellationFactor[0] = tessLevelOuter[0];
    quadFactors.edgeTessellationFactor[1] = tessLevelOuter[1];
    quadFactors.edgeTessellationFactor[2] = tessLevelOuter[2];
    quadFactors.edgeTessellationFactor[3] = tessLevelOuter[3];
    quadFactors.insideTessellationFactor[0] = tessLevelInner[0];
    quadFactors.insideTessellationFactor[1] = tessLevelInner[1];
}

//----------------------------------------------------------
// Patches.GregoryBasis.Vertex
//----------------------------------------------------------

void OsdComputePerPatchVertex(
	int3 patchParam, 
	unsigned ID, 
	unsigned PrimitiveID, 
	unsigned ControlID,
	threadgroup PatchVertexType* patchVertices,
	OsdPatchParamBufferSet osdBuffers
	)
{
	//Does nothing, all transforms are in the PTVS

}

//----------------------------------------------------------
// Patches.GregoryBasis.Domain
//----------------------------------------------------------

#define USE_128BIT_GREGORY_BASIS_INDICES_READ 1


#if USE_STAGE_IN
template<typename PerPatchVertexGregoryBasis>
#endif
OsdPatchVertex ds_gregory_basis_patches(

#if USE_STAGE_IN
                     PerPatchVertexGregoryBasis patch,
#else
                     const device OsdInputVertexType* patch,
                     const device unsigned* patchIndices,
#endif
                     int3 patchParam,
                     float2 UV
                     )
{
    OsdPatchVertex output;
    float3 P = float3(0,0,0), dPu = float3(0,0,0), dPv = float3(0,0,0);
    float3 N = float3(0,0,0), dNu = float3(0,0,0), dNv = float3(0,0,0);
 
#if USE_STAGE_IN
    float3 cv[20];
    for(int i = 0; i < 20; i++)
        cv[i] = patch[i].position;
#else   
#if USE_128BIT_GREGORY_BASIS_INDICES_READ
    float3 cv[20];
    for(int i = 0; i < 5; i++) {
        int4 indices = ((device int4*)patchIndices)[i];
        
        int n = i * 4;
        cv[n + 0] = (patch + indices[0])->position;
        cv[n + 1] = (patch + indices[1])->position;
        cv[n + 2] = (patch + indices[2])->position;
        cv[n + 3] = (patch + indices[3])->position;
    }
#else
    float3 cv[20];
    for (int i = 0; i < 20; ++i) {
        cv[i] = patch[patchIndices[i]].position;
    }
#endif
#endif
    
    OsdEvalPatchGregory(patchParam, UV, cv, P, dPu, dPv, N, dNu, dNv);
    
    output.position = P;
    output.normal = N;
    output.tangent = dPu;
    output.bitangent = dPv;
#if OSD_COMPUTE_NORMAL_DERIVATIVES
    output.Nu = dNu;
    output.Nv = dNv;
#endif
    
    output.patchCoord = OsdInterpolatePatchCoord(UV, patchParam);
    
    return output;
}

#if USE_STAGE_IN
template<typename PerPatchVertexGregoryBasis>
#endif
OsdPatchVertex OsdComputePatch(
	float tessLevel,
	float2 domainCoord,
	unsigned patchID,
#if USE_STAGE_IN
	PerPatchVertexGregoryBasis osdPatch
#else
	OsdVertexBufferSet osdBuffers
#endif
	)
{
	return ds_gregory_basis_patches(
#if USE_STAGE_IN
		osdPatch.cv,
		osdPatch.patchParam,
#else
		osdBuffers.vertexBuffer,
		osdBuffers.indexBuffer + patchID * VERTEX_CONTROL_POINTS_PER_PATCH,
		osdBuffers.patchParamBuffer[patchID],
#endif
		domainCoord
		);
}
