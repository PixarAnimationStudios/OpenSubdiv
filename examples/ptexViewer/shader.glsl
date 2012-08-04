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

#version 400

//--------------------------------------------------------------
// Common
//--------------------------------------------------------------
uniform int ptexLevel;
uniform isamplerBuffer ptexIndices;

vec4 PTexLookup(vec2 faceUV,
                sampler2DArray data,
                samplerBuffer packings,
                isamplerBuffer pages)
{
    ivec2 ptexIndex = texelFetch(ptexIndices, gl_PrimitiveID).xy;
    int faceID = abs(ptexIndex.x);
    int u = ptexIndex.y >> 16;
    int v = (ptexIndex.y & 0xffff);
    int lv = ptexLevel;
    if (ptexIndex.x < 0) lv >>= 1; // non-quad root face

    int page = texelFetch(pages, faceID).x;
    vec4 packing = texelFetch(packings, faceID);

    vec2 uv = (faceUV * vec2(1.0)/lv) + vec2(u, v)/lv;

    vec3 coords = vec3( packing.x + uv.x * packing.z,
                        packing.y + uv.y * packing.w,
                        page);

    return texture(data, coords);
}

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;

out vec3 vPosition;
out vec3 vNormal;
out vec4 vColor;

void main()
{
    vPosition = position;
    vNormal = normal;
    vColor = vec4(1, 1, 1, 1);
}

#endif

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

#if USE_PTEX_DISPLACEMENT
uniform sampler2DArray textureDisplace_Data;
uniform samplerBuffer textureDisplace_Packing;
uniform isamplerBuffer textureDisplace_Pages;
#endif

uniform mat4 objectToClipMatrix;
uniform mat4 objectToEyeMatrix;

in vec3 vPosition[4];
in vec3 vNormal[4];

flat out vec3 gFacetNormal;
out vec3 Peye;
out vec3 Neye;
out vec4 Cout;
out vec2 gFaceUV;

void main()
{
  gl_PrimitiveID = gl_PrimitiveIDIn;

  vec3 pos[4];
  vec2 teTextureCoord[4];
  teTextureCoord[0] = vec2(0, 0);
  teTextureCoord[1] = vec2(1, 0);
  teTextureCoord[2] = vec2(1, 1);
  teTextureCoord[3] = vec2(0, 1);
  pos[0] = vPosition[0];
  pos[1] = vPosition[1];
  pos[2] = vPosition[2];
  pos[3] = vPosition[3];


#if USE_PTEX_DISPLACEMENT
  for(int i=0; i< 4; i++){
     vec4 displace = PTexLookup(teTextureCoord[i],
                               textureDisplace_Data,
                               textureDisplace_Packing,
    textureDisplace_Pages);
    
     pos[i] += displace.x * vNormal[i];
}
#endif

    vec3 A = pos[0] - pos[1];
    vec3 B = pos[3] - pos[1];
    vec3 C = pos[2] - pos[1];
    gFacetNormal = (objectToEyeMatrix*vec4(normalize(cross(B, A)), 0)).xyz;

    Peye = pos[0];
    gl_Position = objectToClipMatrix * vec4(pos[0], 1);
    Neye = (objectToEyeMatrix * vec4(vNormal[0], 0)).xyz;
    gFaceUV = teTextureCoord[0];
    EmitVertex();

    Peye = pos[1];
    gl_Position = objectToClipMatrix * vec4(pos[1], 1);
    Neye = (objectToEyeMatrix * vec4(vNormal[1], 0)).xyz;
    gFaceUV = teTextureCoord[1];
    EmitVertex();

    Peye = pos[3];
    gl_Position = objectToClipMatrix * vec4(pos[3], 1);
    Neye = (objectToEyeMatrix * vec4(vNormal[3], 0)).xyz;
    gFaceUV = teTextureCoord[3];
    EmitVertex();

    gFacetNormal = (objectToEyeMatrix*vec4(normalize(cross(C, B)), 0)).xyz;

    Peye = pos[2];
    gl_Position = objectToClipMatrix * vec4(pos[2], 1);
    Neye = (objectToEyeMatrix * vec4(vNormal[2], 0)).xyz;
    gFaceUV = teTextureCoord[2];
    EmitVertex();

    EndPrimitive();
}

#endif

//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

uniform int ptexFaceOffset;

#if USE_PTEX_COLOR
uniform sampler2DArray textureImage_Data;
uniform samplerBuffer textureImage_Packing;
uniform isamplerBuffer textureImage_Pages;
#endif

#if USE_PTEX_OCCLUSION
uniform sampler2DArray textureOcclusion_Data;
uniform samplerBuffer textureOcclusion_Packing;
uniform isamplerBuffer textureOcclusion_Pages;
#endif

flat in vec3 gFacetNormal;
in vec3 Neye;
in vec3 Peye;
in vec4 Cout;
in vec2 gFaceUV;

#define NUM_LIGHTS 1

struct LightSource {
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

uniform LightSource lightSource[NUM_LIGHTS];
uniform bool useLighting = true;

vec4
lighting(vec3 Peye, vec3 Neye)
{
    vec4 color = vec4(0);

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        vec4 Plight = lightSource[i].position;

        vec3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        vec3 n = normalize(Neye);
        vec3 h = normalize(l + vec3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 8.0f);

#if USE_PTEX_OCCLUSION
        float occ = PTexLookup(gFaceUV,
                               textureOcclusion_Data,
                               textureOcclusion_Packing,
                               textureOcclusion_Pages).x;
#else
        float occ = 0.0;
#endif
        color += (1.0-occ) * ((lightSource[i].ambient +
                               d * lightSource[i].diffuse +
                               s * lightSource[i].specular));
    }

    color.a = 1;

    return color;
}

void
main()
{
#if USE_PTEX_COLOR
    vec4 texColor = PTexLookup(gFaceUV,
                               textureImage_Data,
                               textureImage_Packing,
                               textureImage_Pages);
#else
    vec4 texColor = vec4(1);
#endif
    texColor = pow(texColor, vec4(0.4545));


    if (useLighting) {
#if USE_PTEX_DISPLACEMENT
        vec3 N = (gl_FrontFacing ? gFacetNormal : -gFacetNormal);
#else
        vec3 N = (gl_FrontFacing ? Neye : -Neye);
#endif

        //gl_FragColor = lighting(Peye, N) * texColor * Cout;
        gl_FragColor = lighting(Peye, N) * texColor;
    } else {
        gl_FragColor = texColor*Cout;
    }
}

#endif

//--------------------------------------------------------------
// Ptex debug vertex shader
//--------------------------------------------------------------
#ifdef PTEX_DEBUG_VERTEX_SHADER

in vec3 position;
out vec2 texCoord;

void
main()
{
    gl_Position = vec4(position, 1);
    texCoord = position.xy;
}

#endif

//--------------------------------------------------------------
// Ptex debug fragment shader
//--------------------------------------------------------------
#ifdef PTEX_DEBUG_FRAGMENT_SHADER

uniform int ptexDebugPage;
uniform sampler2DArray ptexDebugData;

in vec2 texCoord;

void
main()
{
    gl_FragColor = texture(ptexDebugData, vec3(texCoord.x, texCoord.y, ptexDebugPage));
}

#endif
