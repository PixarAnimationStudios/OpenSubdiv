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
// sky vertex shader
//--------------------------------------------------------------
#ifdef SKY_VERTEX_SHADER

layout (location=0) in vec3 position;
layout (location=1) in vec2 texCoord;

uniform mat4 ModelViewProjectionMatrix;

out vec2 outTexCoord;

void
main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(position, 1);
    outTexCoord = texCoord.xy;
}

#endif

//--------------------------------------------------------------
// sky fragment shader
//--------------------------------------------------------------
#ifdef SKY_FRAGMENT_SHADER

uniform sampler2D environmentMap;

in vec2 outTexCoord;
out vec4 outColor;

vec4 getEnvironmentHDR(sampler2D sampler, vec2 uv)
{
    vec4 tex = texture(sampler, uv);
    tex = vec4(pow(tex.xyz, vec3(0.4545)), 1);
    return tex;
}

void
main()
{
    outColor = getEnvironmentHDR(environmentMap, outTexCoord);
}

#endif
