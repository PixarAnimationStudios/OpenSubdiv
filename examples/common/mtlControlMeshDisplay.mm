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

#include "mtlControlMeshDisplay.h"
#include <cassert>
#include "mtlUtils.h"

static const char* s_Shader = R"(
#include <metal_stdlib>
using namespace metal;

float4 sharpnessToColor(float s) {                                        
  //  0.0       2.0       4.0                                             
  // green --- yellow --- red                                             
  return float4(min(1.0, s * 0.5),
                min(1.0, 2.0 - s * 0.5),
                0.0, 1.0);
}

struct DrawData
{
	float4x4 ModelViewProjectionMatrix;
};

struct VertexData
{
	float3 position [[attribute(0)]];
	float sharpness [[attribute(1)]];
};

struct FragmentData
{
	float4 position [[position]];
	float4 color;
};

vertex FragmentData vs_main(VertexData in [[stage_in]],
			   const constant DrawData& drawData [[buffer(2)]]
			   )
{
	FragmentData out;
	out.position = drawData.ModelViewProjectionMatrix * float4(in.position, 1.0);
	out.color = sharpnessToColor(in.sharpness);
	return out;
}

fragment float4 fs_main(FragmentData in [[stage_in]])
{
    return in.color;
}
)";

MTLControlMeshDisplay::MTLControlMeshDisplay(id<MTLDevice> device, MTLRenderPipelineDescriptor* pipelineDescriptor) 
	: _device(device), _displayEdges(false), _displayVertices(false), _numEdges(0), _numPoints(0) {
    const auto result = createProgram(pipelineDescriptor);
	assert(result && "Failed to create program for MTLControlMeshDisplay");
}

 void MTLControlMeshDisplay::SetTopology(OpenSubdiv::Far::TopologyLevel const &level) {
    using namespace OpenSubdiv;
     
	_numEdges = level.GetNumEdges();
	_numPoints = level.GetNumVertices();

	std::vector<int> edgeIndices;
	std::vector<float> edgeSharpness;
	std::vector<float> vertSharpness;

	edgeIndices.reserve(_numEdges * 2);
	edgeSharpness.reserve(_numEdges);
	vertSharpness.reserve(_numPoints);

	for(int i = 0; i < _numEdges; i++) {
	    const auto verts = level.GetEdgeVertices(i);
	    edgeIndices.emplace_back(verts[0]);
	    edgeIndices.emplace_back(verts[1]);
	    edgeSharpness.emplace_back(level.GetEdgeSharpness(i));
	}

	for(int i = 0; i < _numPoints; i++) {
	    vertSharpness.emplace_back(level.GetVertexSharpness(i));
	}

    _edgeIndicesBuffer = Osd::MTLNewBufferFromVector(_device, edgeIndices);
    _edgeSharpnessBuffer = Osd::MTLNewBufferFromVector(_device, edgeSharpness);
    _vertexSharpnessBuffer = Osd::MTLNewBufferFromVector(_device, vertSharpness);
}

bool MTLControlMeshDisplay::createProgram(MTLRenderPipelineDescriptor* _pipelineDescriptor) {
	const auto options = [MTLCompileOptions new];
	NSError* error = nil;

	const auto library = [_device newLibraryWithSource:@(s_Shader) options:options error:&error];
	if(!library) {
        printf("Failed to create library for MTLControlMeshDisplay\n%s\n", error ? [[error localizedDescription] UTF8String] : "");
		return false;
	}

	const auto vertexFunction = [library newFunctionWithName:@"vs_main"];
	const auto fragmentFunction = [library newFunctionWithName:@"fs_main"];

	MTLRenderPipelineDescriptor* pipelineDescriptor = [_pipelineDescriptor copy];
	pipelineDescriptor.vertexFunction = vertexFunction;
	pipelineDescriptor.fragmentFunction = fragmentFunction;
	const auto vertexDescriptor = pipelineDescriptor.vertexDescriptor;
	vertexDescriptor.layouts[1].stride = sizeof(float) * 6;
	vertexDescriptor.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;
	vertexDescriptor.layouts[1].stepRate = 1;
	vertexDescriptor.attributes[1].bufferIndex = 1;
	vertexDescriptor.attributes[1].offset = 0;
	vertexDescriptor.attributes[1].format = MTLVertexFormatFloat3;

	_renderPipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];

	if(!_renderPipelineState) {
        printf("Failed to create render pipeline state for MTLControlMeshDisplay\n%s\n", error ? [[error localizedDescription] UTF8String] : "");
	}
	return true;
}

void MTLControlMeshDisplay::Draw(id<MTLRenderCommandEncoder> encoder, 
    id<MTLBuffer> vertexBuffer,
    const float *modelViewProjectionMatrix) {
	[encoder setRenderPipelineState: _renderPipelineState];
	[encoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
	[encoder setVertexBytes:modelViewProjectionMatrix length: sizeof(float) * 16 atIndex:2];

	if(_displayEdges) {
		[encoder setVertexBuffer:_edgeSharpnessBuffer offset:0 atIndex:1];
		[encoder drawIndexedPrimitives:MTLPrimitiveTypeLine
                            indexCount:_numEdges * 2
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:_edgeIndicesBuffer
                     indexBufferOffset:0
                         instanceCount:1
                            baseVertex:0
                          baseInstance:0];
	}

	if(_displayVertices) {
		[encoder setVertexBuffer:_vertexSharpnessBuffer offset:0 atIndex:1];
		[encoder drawPrimitives:MTLPrimitiveTypePoint
                    vertexStart:0
                    vertexCount:_numPoints];
	}
}
