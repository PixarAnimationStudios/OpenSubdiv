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
// image fragment shader
//--------------------------------------------------------------
#ifdef IMAGE_FRAGMENT_SHADER

uniform sampler2D colorMap;
uniform sampler2D depthMap;
in vec2 outUV;
out vec4 outColor;

#ifdef BLUR
#define NUM_BLUR_SAMPLES 7
uniform vec2 Offsets[NUM_BLUR_SAMPLES];
uniform float Weights[NUM_BLUR_SAMPLES];

void main()
{
    outColor = vec4(0);
    for (int i = 0; i < NUM_BLUR_SAMPLES; ++i) {
        float w = Weights[i];
        vec2 o = Offsets[i];
        outColor += w * texture(colorMap, outUV + o);
    }
}
#endif

#ifdef HIPASS
uniform float Threshold = 0.95;
const vec3 Black = vec3(0, 0, 0);

void main()
{
    vec3 c = texture(colorMap, outUV).rgb;
    float gray = dot(c, c);
    outColor = vec4(gray > Threshold ? c : Black, 1);
}
#endif

#ifdef COMPOSITE
uniform float alpha = 1.0;
void main()
{
    outColor = alpha * texture(colorMap, outUV);
}
#endif

#endif
