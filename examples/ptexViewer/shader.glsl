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
#line 57

//--------------------------------------------------------------
// Common
//--------------------------------------------------------------
uniform int nonAdaptiveLevel;
uniform float displacementScale = 1.0;
uniform float bumpScale = 1.0;

vec4 GeneratePatchCoord(vec2 localUV)  // for non-adpative
{
    ivec2 ptexIndex = texelFetchBuffer(g_ptexIndicesBuffer, gl_PrimitiveID).xy;
    int faceID = ptexIndex.x;
    int lv = 1 << (ptexIndex.y & 0xf);
    int u = (ptexIndex.y >> 17) & 0x3ff;
    int v = (ptexIndex.y >> 7) & 0x3ff;
    vec2 uv = localUV;
    uv = (uv * vec2(1.0)/lv) + vec2(u, v)/lv;
    
    return vec4(uv.x, uv.y, lv+0.5, faceID+0.5);
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

#define OSD_DISPLACEMENT_CALLBACK                   \
    output.v.position =                         \
        displacement(output.v.position,         \
                     output.v.normal,           \
                     output.v.patchCoord);

uniform sampler2DArray textureDisplace_Data;
uniform samplerBuffer textureDisplace_Packing;
uniform isamplerBuffer textureDisplace_Pages;

vec4 displacement(vec4 position, vec3 normal, vec4 patchCoord)
{
    float disp = PTexLookup(patchCoord,
                            textureDisplace_Data,
                            textureDisplace_Packing,
                            textureDisplace_Pages).x;
    return position + vec4(disp * normal, 0) * displacementScale;
}
#endif

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;
layout (location=1) in vec3 normal;

out block {
    OutputVertex v;
} output;

void main()
{
    output.v.position = ModelViewMatrix * position;
    output.v.normal = (ModelViewMatrix * vec4(normal, 0)).xyz;
}

#endif

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

//uniform int nonAdaptiveLevel;

#ifdef PRIM_QUAD

    layout(lines_adjacency) in;

    layout(triangle_strip, max_vertices = 4) out;

    #define EDGE_VERTS 4

    in block {
        OutputVertex v;
    } input[4];

#endif // PRIM_QUAD

#ifdef  PRIM_TRI

    layout(triangles) in;

    layout(triangle_strip, max_vertices = 3) out;

    #define EDGE_VERTS 3

    in block {
        OutputVertex v;
    } input[3];

#endif // PRIM_TRI

out block {
    OutputVertex v;
} output;

// --------------------------------------

void emit(vec4 position, vec3 normal, vec4 patchCoord)
{
    output.v.position = position;
    output.v.patchCoord = patchCoord;
    output.v.normal = normal;

    gl_Position = ProjectionMatrix * output.v.position;
    EmitVertex();
}

const float VIEWPORT_SCALE = 1024.0; // XXXdyu

float edgeDistance(vec4 p, vec4 p0, vec4 p1)
{
    return VIEWPORT_SCALE *
        abs((p.x - p0.x) * (p1.y - p0.y) -
            (p.y - p0.y) * (p1.x - p0.x)) / length(p1.xy - p0.xy);
}

void emit(int index, vec4 position, vec3 normal, vec4 patchCoord, vec4 edgeVerts[EDGE_VERTS])
{
    output.v.edgeDistance[0] =
        edgeDistance(edgeVerts[index], edgeVerts[0], edgeVerts[1]);
    output.v.edgeDistance[1] =
        edgeDistance(edgeVerts[index], edgeVerts[1], edgeVerts[2]);
#ifdef PRIM_TRI
    output.v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[0]);
#endif
#ifdef PRIM_QUAD
    output.v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[3]);
    output.v.edgeDistance[3] =
        edgeDistance(edgeVerts[index], edgeVerts[3], edgeVerts[0]);
#endif

    emit(position, normal, patchCoord);
}

// --------------------------------------

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

#ifdef PRIM_QUAD
    vec4 patchCoord[4];
    vec4 position[4];
    vec3 normal[4];

    // need to generate patch coord for non-patch quads
    patchCoord[0] = GeneratePatchCoord(vec2(0, 0));
    patchCoord[1] = GeneratePatchCoord(vec2(1, 0));
    patchCoord[2] = GeneratePatchCoord(vec2(1, 1));
    patchCoord[3] = GeneratePatchCoord(vec2(0, 1));

#ifdef USE_PTEX_DISPLACEMENT
    position[0] = displacement(input[0].v.position, input[0].v.normal, patchCoord[0]);
    position[1] = displacement(input[1].v.position, input[1].v.normal, patchCoord[1]);
    position[2] = displacement(input[2].v.position, input[2].v.normal, patchCoord[2]);
    position[3] = displacement(input[3].v.position, input[3].v.normal, patchCoord[3]);
#else
    position[0] = input[0].v.position;
    position[1] = input[1].v.position;
    position[2] = input[2].v.position;
    position[3] = input[3].v.position;
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
    normal[0] = input[0].v.normal;
    normal[1] = input[1].v.normal;
    normal[2] = input[2].v.normal;
    normal[3] = input[3].v.normal;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * input[0].v.position;
    edgeVerts[1] = ProjectionMatrix * input[1].v.position;
    edgeVerts[2] = ProjectionMatrix * input[2].v.position;
    edgeVerts[3] = ProjectionMatrix * input[3].v.position;

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;
    edgeVerts[3].xy /= edgeVerts[3].w;

    emit(0, position[0], normal[0], patchCoord[0], edgeVerts);
    emit(1, position[1], normal[1], patchCoord[1], edgeVerts);
    emit(3, position[3], normal[3], patchCoord[3], edgeVerts);
    emit(2, position[2], normal[2], patchCoord[2], edgeVerts);
#else
    emit(position[0], normal[0], patchCoord[0]);
    emit(position[1], normal[1], patchCoord[1]);
    emit(position[3], normal[3], patchCoord[3]);
    emit(position[2], normal[2], patchCoord[2]);
#endif
#endif // PRIM_QUAD

#ifdef PRIM_TRI
    vec4 position[3];
    vec4 patchCoord[3];
    vec3 normal[3];

    // patch coords are computed in tessellation shader
    patchCoord[0] = input[0].v.patchCoord;
    patchCoord[1] = input[1].v.patchCoord;
    patchCoord[2] = input[2].v.patchCoord;

#ifdef USE_PTEX_DISPLACEMENT
    position[0] = displacement(input[0].v.position, input[0].v.normal, patchCoord[0]);
    position[1] = displacement(input[1].v.position, input[1].v.normal, patchCoord[1]);
    position[2] = displacement(input[2].v.position, input[2].v.normal, patchCoord[2]);
#else
    position[0] = input[0].v.position;
    position[1] = input[1].v.position;
    position[2] = input[2].v.position;
#endif

#ifdef FLAT_NORMALS  // emit flat normals for displaced surface
    vec3 A = (position[0] - position[1]).xyz;
    vec3 B = (position[2] - position[1]).xyz;
    normal[0] = normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
#else
    normal[0] = input[0].v.normal;
    normal[1] = input[1].v.normal;
    normal[2] = input[2].v.normal;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * input[0].v.position;
    edgeVerts[1] = ProjectionMatrix * input[1].v.position;
    edgeVerts[2] = ProjectionMatrix * input[2].v.position;

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;

    emit(0, position[0], normal[0], patchCoord[0], edgeVerts);
    emit(1, position[1], normal[1], patchCoord[1], edgeVerts);
    emit(2, position[2], normal[2], patchCoord[2], edgeVerts);
#else
    emit(position[0], normal[0], patchCoord[0]);
    emit(position[1], normal[1], patchCoord[1]);
    emit(position[2], normal[2], patchCoord[2]);
#endif
#endif // PRIM_TRI

    EndPrimitive();
}

#endif
//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
} input;

out vec4 outColor;

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

#ifdef USE_PTEX_SPECULAR
uniform sampler2DArray textureSpecular_Data;
uniform samplerBuffer textureSpecular_Packing;
uniform isamplerBuffer textureSpecular_Pages;
#endif

#define NUM_LIGHTS 2

struct LightSource {
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

layout(std140) uniform Lighting {
    LightSource lightSource[NUM_LIGHTS];
};

uniform bool overrideColorEnable = false;
uniform vec4 overrideColor;

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
    float Hll = PTexLookup(STll, textureDisplace_Data, textureDisplace_Packing, textureDisplace_Pages).x * bumpScale;
    float Hlr = PTexLookup(STlr, textureDisplace_Data, textureDisplace_Packing, textureDisplace_Pages).x * bumpScale;
    float Hul = PTexLookup(STul, textureDisplace_Data, textureDisplace_Packing, textureDisplace_Pages).x * bumpScale;
    float dBs = (Hlr - Hll)/d;
    float dBt = (Hul - Hll)/d;
#endif
    
    vec3 vSurfGrad = sign(fDet) * (dBs * vR1 + dBt * vR2);
    return normalize(abs(fDet) * vN - vSurfGrad);
}
#endif // USE_PTEX_NORMAL

uniform sampler2D diffuseEnvironmentMap;
uniform sampler2D specularEnvironmentMap;

vec4 getEnvironmentHDR(sampler2D sampler, vec3 dir)
{
    dir = (ModelViewInverseMatrix * vec4(dir, 0)).xyz;
    vec2 uv = vec2((atan(dir.x,dir.z)/3.1415926535897+1)*0.5, (1-dir.y)*0.5);
    vec4 tex = texture(sampler, uv);
    tex = vec4(pow(tex.xyz, vec3(0.4545)), 1);
    return tex;
}

vec4
lighting(vec4 texColor, vec3 Peye, vec3 Neye)
{
    vec4 color = vec4(0);

#ifdef USE_PTEX_OCCLUSION
    float occ = PTexLookup(input.v.patchCoord,
                           textureOcclusion_Data,
                           textureOcclusion_Packing,
                           textureOcclusion_Pages).x;
#else
    float occ = 0.0;
#endif
    vec3 n = Neye;



    for (int i = 0; i < NUM_LIGHTS; ++i) {

        vec4 Plight = lightSource[i].position;
        vec3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        vec3 h = normalize(l + vec3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 64.0f);

        color += (1.0-occ) * ((lightSource[i].ambient +
                               d * lightSource[i].diffuse) * texColor + 
                              s * lightSource[i].specular);
    }

    color.a = 1;

    return color;
}

vec4
edgeColor(vec4 Cfill, vec4 edgeDistance)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float d =
        min(input.v.edgeDistance[0], min(input.v.edgeDistance[1], input.v.edgeDistance[2]));
#endif
#ifdef PRIM_QUAD
    float d =
        min(min(input.v.edgeDistance[0], input.v.edgeDistance[1]),
            min(input.v.edgeDistance[2], input.v.edgeDistance[3]));
#endif
    vec4 Cedge = vec4(1.0, 1.0, 0.0, 1.0);
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    Cfill.rgb = mix(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

void
main()
{
#if USE_PTEX_COLOR
    vec4 texColor = PTexLookup(input.v.patchCoord,
                               textureImage_Data,
                               textureImage_Packing,
                               textureImage_Pages);
//    texColor = vec4(pow(texColor.xyz, vec3(0.4545)), 1);
#else
    vec4 texColor = vec4(1);
#endif

#if USE_PTEX_NORMAL
    vec3 normal = perturbNormalFromDisplacement(input.v.position.xyz,
                                                input.v.normal,
                                                input.v.patchCoord);
#else
    vec3 normal = input.v.normal;
#endif

    if (overrideColorEnable) {
        texColor = overrideColor;
        vec4 Cf = lighting(texColor, input.v.position.xyz, normal);
        outColor = edgeColor(Cf, input.v.edgeDistance);
        return;
    }

#if USE_IBL
#ifdef USE_PTEX_OCCLUSION
    float occ = PTexLookup(input.v.patchCoord,
                           textureOcclusion_Data,
                           textureOcclusion_Packing,
                           textureOcclusion_Pages).x;
#else
    float occ = 0.0;
#endif

#ifdef USE_PTEX_SPECULAR
    float specular = PTexLookup(input.v.patchCoord,
                                textureSpecular_Data,
                                textureSpecular_Packing,
                                textureSpecular_Pages).x;
#else
    float specular = 1.0;
#endif

    vec4 a = vec4(0, 0, 0, 1); //ambientColor;
    vec4 d = getEnvironmentHDR(diffuseEnvironmentMap, normal) * 1.4;
    vec3 eye = normalize(input.v.position.xyz - vec3(0,0,0));
    vec3 reflect = reflect(eye, normal);
    vec4 s = getEnvironmentHDR(specularEnvironmentMap, reflect);
    const float fresnelBias = 0;
    const float fresnelScale = 1.0;
    const float fresnelPower = 2.0;
    float fresnel = fresnelBias + fresnelScale * pow(1.0+dot(normal,eye), fresnelPower);

    a *= (1.0-occ);
    d *= (1.0-occ);
    s *= min(specular, (1.0-occ)) * fresnel;

    vec4 Cf = (a + d) * texColor + s * 0.5;
#else
    vec4 Cf = lighting(texColor, input.v.position.xyz, normal);
#endif

    outColor = edgeColor(Cf, input.v.edgeDistance);
}

#endif
