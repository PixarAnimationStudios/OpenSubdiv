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
#line 57

//--------------------------------------------------------------
// Common
//--------------------------------------------------------------

vec4 GeneratePatchCoord(vec2 localUV)  // for non-adpative
{
    ivec2 ptexIndex = texelFetch(OsdPatchParamBuffer, gl_PrimitiveID).xy;
    int faceID = ptexIndex.x;
    int lv = 1 << ((ptexIndex.y & 0xf) - ((ptexIndex.y >> 4) & 1));
    int u = (ptexIndex.y >> 17) & 0x3ff;
    int v = (ptexIndex.y >> 7) & 0x3ff;
    vec2 uv = localUV;
    uv = (uv * vec2(1.0)/lv) + vec2(u, v)/lv;

}

vec4 PTexLookup(vec4 patchCoord,
                sampler2DArray data,
                samplerBuffer packings,
                isamplerBuffer pages)
{
    vec2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    int page = texelFetch(pages, faceID).x;
    vec4 packing = texelFetch(packings, faceID);
    vec3 coords = vec3( packing.x + uv.x * packing.z,
                        packing.y + uv.y * packing.w,
                        page);

    return texture(data, coords);
}

#ifdef USE_PTEX_DISPLACEMENT

#define OSD_DISPLACEMENT_CALLBACK              \
    outpt.v.position =                         \
        displacement(outpt.v.position,         \
                     outpt.v.normal,           \
                     outpt.v.patchCoord);

uniform sampler2DArray textureDisplace_Data;
uniform samplerBuffer textureDisplace_Packing;
uniform isamplerBuffer textureDisplace_Pages;

vec4 displacement(vec4 position, vec3 normal, vec4 patchCoord)
{
    float disp = PTexLookup(patchCoord,
                            textureDisplace_Data,
                            textureDisplace_Packing,
                            textureDisplace_Pages).x;
    return position + vec4(disp * normal, 0);
}
#endif

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;

out block {
    OutputVertex v;
} outpt;

void main()
{
    outpt.v.position = vec4(position, 1);
    outpt.v.normal = normal;
}

#endif // VERTEX_SHADER

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

#ifdef PRIM_QUAD

    layout(lines_adjacency) in;

    #ifdef GEOMETRY_OUT_FILL
    layout(triangle_strip, max_vertices = 4) out;
    #endif

    #ifdef GEOMETRY_OUT_LINE
    layout(line_strip, max_vertices = 5) out;
    #endif

    in block {
        OutputVertex v;
    } inpt[4];

#endif // PRIM_QUAD

#ifdef  PRIM_TRI

    layout(triangles) in;

    #ifdef GEOMETRY_OUT_FILL
    layout(triangle_strip, max_vertices = 3) out;
    #endif

    #ifdef GEOMETRY_OUT_LINE
    layout(line_strip, max_vertices = 4) out;
    #endif

    in block {
        OutputVertex v;
    } inpt[3];

#endif // PRIM_TRI

out block {
    OutputVertex v;
} outpt;

void emit(int index, vec4 position, vec3 normal, vec4 patchCoord)
{
    outpt.v.position = position;
    outpt.v.normal = normal;
    outpt.v.patchCoord = patchCoord;

    outpt.v.tangent = inpt[index].v.tangent;

    gl_Position = ProjectionMatrix * outpt.v.position;
    EmitVertex();
}

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

    
#ifdef PRIM_QUAD
#ifdef GEOMETRY_OUT_FILL

    vec4 patchCoord[4];
    vec4 position[4];
    vec3 normal[4];

    // need to generate patch coord for non-patch quads
    patchCoord[0] = GeneratePatchCoord(vec2(0, 0));
    patchCoord[1] = GeneratePatchCoord(vec2(1, 0));
    patchCoord[2] = GeneratePatchCoord(vec2(1, 1));
    patchCoord[3] = GeneratePatchCoord(vec2(0, 1));

#ifdef USE_PTEX_DISPLACEMENT
    position[0] = displacement(inpt[0].v.position, inpt[0].v.normal, patchCoord[0]);
    position[1] = displacement(inpt[1].v.position, inpt[1].v.normal, patchCoord[1]);
    position[2] = displacement(inpt[2].v.position, inpt[2].v.normal, patchCoord[2]);
    position[3] = displacement(inpt[3].v.position, inpt[3].v.normal, patchCoord[3]);
#else
    position[0] = inpt[0].v.position;
    position[1] = inpt[1].v.position;
    position[2] = inpt[2].v.position;
    position[3] = inpt[3].v.position;
#endif

#ifdef FLAT_NORMALS
    // XXX: need to use vec C to get triangle normal.
    vec3 A = (position[0] - position[1]).xyz;
    vec3 B = (position[3] - position[1]).xyz;
    vec3 C = (position[2] - position[1]).xyz;
    normal[0] = normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
    normal[3] = normal[0];
#else
    normal[0] = inpt[0].v.normal;
    normal[1] = inpt[1].v.normal;
    normal[2] = inpt[2].v.normal;
    normal[3] = inpt[3].v.normal;
#endif

    emit(0, position[0], normal[0], patchCoord[0]);
    emit(1, position[1], normal[1], patchCoord[1]);
    emit(3, position[3], normal[3], patchCoord[3]);
    emit(2, position[2], normal[2], patchCoord[2]);
#else  // GEOMETRY_OUT_LINE
    emit(0, position[0], vec3(0), patchCoord[0]);
    emit(1, position[1], vec3(0), patchCoord[1]);
    emit(2, position[2], vec3(0), patchCoord[2]);
    emit(3, position[3], vec3(0), patchCoord[3]);
    emit(0, position[0], vec3(0), patchCoord[0]);
#endif
#endif // PRIM_QUAD

#ifdef PRIM_TRI

    vec4 position[3];
    vec4 patchCoord[3];
    vec3 normal[3];

    // patch coords are computed in tessellation shader
    patchCoord[0] = inpt[0].v.patchCoord;
    patchCoord[1] = inpt[1].v.patchCoord;
    patchCoord[2] = inpt[2].v.patchCoord;

#ifdef USE_PTEX_DISPLACEMENT
    position[0] = displacement(inpt[0].v.position, inpt[0].v.normal, patchCoord[0]);
    position[1] = displacement(inpt[1].v.position, inpt[1].v.normal, patchCoord[1]);
    position[2] = displacement(inpt[2].v.position, inpt[2].v.normal, patchCoord[2]);
#else
    position[0] = inpt[0].v.position;
    position[1] = inpt[1].v.position;
    position[2] = inpt[2].v.position;
#endif

#ifdef FLAT_NORMALS  // emit flat normals for displaced surface
    vec3 A = (position[0] - position[1]).xyz;
    vec3 B = (position[2] - position[1]).xyz;
    normal[0] = normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
#else
    normal[0] = inpt[0].v.normal;
    normal[1] = inpt[1].v.normal;
    normal[2] = inpt[2].v.normal;
#endif

    emit(0, position[0], normal[0], patchCoord[0]);
    emit(1, position[1], normal[1], patchCoord[1]);
    emit(2, position[2], normal[2], patchCoord[2]);
#ifdef GEOMETRY_OUT_LINE
    emit(0, position[0], normal[0], patchCoord[0]);
#endif //GEOMETRY_OUT_LINE
#endif // PRIM_TRI

    EndPrimitive();
}

#endif // GEOMETRY_SHADER

//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
} inpt;

uniform int ptexFaceOffset;

#ifdef USE_PTEX_COLOR
uniform sampler2DArray textureImage_Data;
uniform samplerBuffer textureImage_Packing;
uniform isamplerBuffer textureImage_Pages;
#endif

#ifdef USE_PTEX_OCCLUSION
uniform sampler2DArray textureOcclusion_Data;
uniform samplerBuffer textureOcclusion_Packing;
uniform isamplerBuffer textureOcclusion_Pages;
#endif

#if USE_PTEX_NORMAL
uniform sampler2DArray textureDisplace_Data;
uniform samplerBuffer textureDisplace_Packing;
uniform isamplerBuffer textureDisplace_Pages;

vec3
perturbNormalFromDisplacement(vec3 position, vec3 normal, vec4 patchCoord)
{
    // by Morten S. Mikkelsen
    // http://jbit.net/~sparky/sfgrad_bump/mm_sfgrad_bump.pdf
    // slightly modified for ptex guttering

    vec3 vSigmaS = dFdx(position);
    vec3 vSigmaT = dFdy(position);
    vec3 vN = normal;
    vec3 vR1 = cross(vSigmaT, vN);
    vec3 vR2 = cross(vN, vSigmaS);
    float fDet = dot(vSigmaS, vR1);
#if 0
    // not work well with ptex
    float dBs = dFdx(disp);
    float dBt = dFdy(disp);
#else
    vec2 texDx = dFdx(patchCoord.xy);
    vec2 texDy = dFdy(patchCoord.xy);
    
    // limit forward differencing to the width of ptex gutter
    const float resolution = 128.0;
    float d = min(1, (0.5/resolution)/max(length(texDx), length(texDy)));
    
    vec4 STll = patchCoord;
    vec4 STlr = patchCoord + d * vec4(texDx.x, texDx.y, 0, 0);
    vec4 STul = patchCoord + d * vec4(texDy.x, texDy.y, 0, 0);
    float Hll = PTexLookup(STll, textureDisplace_Data, textureDisplace_Packing, textureDisplace_Pages).x;
    float Hlr = PTexLookup(STlr, textureDisplace_Data, textureDisplace_Packing, textureDisplace_Pages).x;
    float Hul = PTexLookup(STul, textureDisplace_Data, textureDisplace_Packing, textureDisplace_Pages).x;
    float dBs = (Hlr - Hll)/d;
    float dBt = (Hul - Hll)/d;
#endif
    
    vec3 vSurfGrad = sign(fDet) * (dBs * vR1 + dBt * vR2);
    return normalize(abs(fDet) * vN - vSurfGrad);
}
#endif // USE_PTEX_NORMAL

uniform sampler2D diffuseEnvironmentMap;
uniform sampler2D specularEnvironmentMap;

#define NUM_LIGHTS 1

struct LightSource {
    vec4 position;
    vec4 diffuse;
    vec4 ambient;
    vec4 specular;
};

layout(std140) uniform Lighting {
    LightSource lightSource[NUM_LIGHTS];
};

uniform vec4 diffuseColor;
uniform vec4 ambientColor;
uniform vec4 specularColor; 
uniform vec3 eyePositionInWorld;

uniform float fresnelBias;
uniform float fresnelScale;
uniform float fresnelPower;

vec4 getEnvironment(sampler2D sampler, vec3 dir)
{
    return texture(sampler, vec2((atan(dir.x,dir.z)/3.1415926+1)*0.5, (1-dir.y)*0.5));
}

void
main()
{
#ifdef USE_PTEX_COLOR
    vec4 texColor = PTexLookup(inpt.v.patchCoord,
                               textureImage_Data,
                               textureImage_Packing,
                               textureImage_Pages);
#else
    vec4 texColor = vec4(1);
#endif

#if USE_PTEX_NORMAL
    vec3 objN = perturbNormalFromDisplacement(inpt.v.position.xyz,
                                              inpt.v.normal,
                                              inpt.v.patchCoord);
#else
    vec3 objN = inpt.v.normal;
#endif


#ifdef USE_PTEX_OCCLUSION
    float occ = PTexLookup(inpt.v.patchCoord,
                           textureOcclusion_Data,
                           textureOcclusion_Packing,
                           textureOcclusion_Pages).x;
#else
    float occ = 0.0;
#endif

    vec4 a = ambientColor;

#ifdef USE_DIFFUSE_ENV_MAP
    vec4 d = getEnvironment(diffuseEnvironmentMap, objN) * 1.4;
#else
    vec4 d = vec4(1);
#endif

    vec3 eye = normalize(inpt.v.position.xyz - eyePositionInWorld);

#ifdef USE_SPECULAR_ENV_MAP
    vec3 reflect = reflect(eye, objN);
    vec4 s = getEnvironment(specularEnvironmentMap, reflect);
#else
    vec4 s = vec4(1);
#endif

    float fresnel = fresnelBias + fresnelScale * pow(1.0+dot(objN,eye), fresnelPower);

    a *= (1.0-occ);
    d *= (1.0-occ)*diffuseColor;
    s *= (1.0-pow(occ, 0.2)) * specularColor * fresnel;
    
    gl_FragColor = (a + d) * texColor + s;
    gl_FragColor = pow(gl_FragColor, vec4(0.4545));
}

#endif // FRAGMENT_SHADER
