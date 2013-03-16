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
