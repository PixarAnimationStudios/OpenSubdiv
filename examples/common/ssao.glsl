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

const vec2 oflut[49] = vec2[](
    vec2(-0.864,-0.930),
    vec2(-0.952,-0.608),
    vec2(-0.820,-0.318),
    vec2(-0.960,-0.076),
    vec2(-0.939,0.244),
    vec2(-0.827,0.421),
    vec2(-0.871,0.561),
    vec2(-0.668,-0.810),
    vec2(-0.749,-0.557),
    vec2(-0.734,-0.422),
    vec2(-0.681,-0.043),
    vec2(-0.598,0.132),
    vec2(-0.720,0.323),
    vec2(-0.513,0.723),
    vec2(-0.371,-0.829),
    vec2(-0.366,-0.735),
    vec2(-0.259,-0.318),
    vec2(-0.344,-0.079),
    vec2(-0.386,0.196),
    vec2(-0.405,0.425),
    vec2(-0.310,0.518),
    vec2(-0.154,-0.931),
    vec2(-0.171,-0.572),
    vec2(-0.241,-0.420),
    vec2(-0.130,-0.225),
    vec2(-0.092,0.189),
    vec2(-0.018,0.310),
    vec2(-0.179,0.512),
    vec2(0.133,-0.942),
    vec2(0.235,-0.738),
    vec2(0.229,-0.380),
    vec2(0.027,-0.030),
    vec2(0.052,0.183),
    vec2(0.141,0.415),
    vec2(0.129,0.736),
    vec2(0.340,-0.931),
    vec2(0.254,-0.564),
    vec2(0.388,-0.417),
    vec2(0.363,-0.104),
    vec2(0.413,0.234),
    vec2(0.421,0.321),
    vec2(0.423,0.653),
    vec2(0.631,-0.756),
    vec2(0.665,-0.736),
    vec2(0.551,-0.350),
    vec2(0.526,-0.220),
    vec2(0.519,0.053),
    vec2(0.750,0.321),
    vec2(0.736,0.641) );

float ZBuffer2Linear( float z ,float znear, float zfar)
{
    float z_n = 2.0 * z - 1.0;
    float z_e = 2.0 * znear * zfar / (zfar+znear-z_n*(zfar-znear));
    return z_e / 10.0;
}

uniform float radius;
uniform float scale;
uniform float gamma;
uniform float contrast;

void main()
{

    vec4 colorSample = texture(colorMap, outUV);

    float zbuffer = texture(depthMap, outUV).r;

    float rz = ZBuffer2Linear( zbuffer, 1.0, 500.0 );

    float an = fract(fract(outUV.x*0.36257507)*0.38746515+outUV.y*0.32126721);

    float k1=cos(6.2831*an);
    float k2=sin(6.2831*an);

    // calculate occlusion factor

    float ao = 0.0;
    for( int i=0; i<49; i++ )
    {
        vec2 of = vec2( oflut[i].x*k1 - oflut[i].y*k2,
                        oflut[i].x*k2 + oflut[i].y*k1 );

        //sampling point
        vec2 sa =  outUV.xy + radius * of;

        //difference in zbuffer
        float zd = rz - ZBuffer2Linear( texture(depthMap, sa).r, 1.0, 500.0);

        //adjust based on scale.
        zd = zd * scale;

        //accumulate zdiffs transfer func
        ao += clamp( zd, 0.0, 1.0 ) - clamp((zd-5.0)*.02,0.0,1.0);
    }

    //add in occlusion to color.
    ao = 1.0 - ao/49.0;

    //scale occlusion toward white.
    ao = clamp(ao, 0.0, 1.0);

    //add in contrast/gamma to colorsample.
    outColor = contrast*pow(colorSample*ao,vec4(gamma));
    
    //background color as a vertical grey ramp
    vec4 bgColor = vec4(mix(0.1, 0.5, sin(outUV.y*3.14159)));

    outColor = mix(bgColor, outColor, colorSample.a);
}

#endif
