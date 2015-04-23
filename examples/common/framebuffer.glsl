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

//--------------------------------------------------------------
// image vertex shader
//--------------------------------------------------------------

#ifdef IMAGE_VERTEX_SHADER

layout (location=0) in vec2 position;
out vec2 outUV;

void
main()
{
    outUV = vec2(position.xy*0.5) + vec2(0.5);
    gl_Position = vec4(position.x, position.y, 0, 1);
}

#endif

//--------------------------------------------------------------
#ifdef IMAGE_FRAGMENT_SHADER

uniform sampler2D colorMap;
uniform sampler2D normalMap;
uniform sampler2D depthMap;

in vec2 outUV;
out vec4 outColor;

void main()
{

    vec4 colorSample = texture(colorMap, outUV);
    
    //background color as a vertical grey ramp
    vec4 bgColor = vec4(mix(0.1, 0.5, sin(outUV.y*3.14159)));

    outColor = mix(bgColor, colorSample, colorSample.a);
}

#endif
