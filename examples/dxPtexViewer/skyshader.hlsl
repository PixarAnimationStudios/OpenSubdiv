//
//   Copyright 2013 Nvidia
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

struct VS_InputVertex {
    float3 position : POSITION0;
    float2 texCoord : TEXCOORD0;
};

struct VS_OutputVertex {
    float4 position : SV_POSITION0;
    float2 texCoord : TEXCOORD0;
};

cbuffer Transform : register( b0 ) {
    float4x4 ModelViewMatrix;
};

//--------------------------------------------------------------
// sky vertex shader
//--------------------------------------------------------------


void vs_main(in VS_InputVertex input,
             out VS_OutputVertex output) {

    output.position = mul(ModelViewMatrix, float4(input.position,1));
    output.texCoord = input.texCoord;
}

//--------------------------------------------------------------
// sky pixel shader
//--------------------------------------------------------------

struct PS_InputVertex {
    float4 position : SV_POSITION0;
    float2 texCoord : TEXCOORD0;
};

Texture2D tx : register(t0);

SamplerState sm : register(s0);

float4
gamma(float4 value, float g) {
    return float4(pow(value.xyz, float3(g,g,g)), 1);
}


float4
ps_main(in PS_InputVertex input) : SV_Target {

    float4 tex = tx.Sample(sm, input.texCoord.xy);

    //float4 outColor = gamma(tex,0.4545);
    float4 outColor = tex;

    return outColor;
}

