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

#define NUM_LIGHTS 2

struct LightSource {
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

layout(std140) uniform Constant {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
    mat4 ModelViewInverseMatrix;
    LightSource lightSource[NUM_LIGHTS];
    float TessLevel;
    float displacementScale;
    float mipmapBias;
};

uniform int PrimitiveIdBase;

int OsdPrimitiveIdBase()
{
    return PrimitiveIdBase;
}

//--------------------------------------------------------------
// Common
//--------------------------------------------------------------

#if    defined(DISPLACEMENT_HW_BILINEAR)        \
    || defined(DISPLACEMENT_BILINEAR)           \
    || defined(DISPLACEMENT_BIQUADRATIC)        \
    || defined(NORMAL_HW_SCREENSPACE)           \
    || defined(NORMAL_SCREENSPACE)              \
    || defined(NORMAL_BIQUADRATIC)              \
    || defined(NORMAL_BIQUADRATIC_WG)
uniform sampler2DArray textureDisplace_Data;
uniform isamplerBuffer textureDisplace_Packing;
#endif

vec4 displacement(vec4 position, vec3 normal, vec4 patchCoord)
{
#if defined(DISPLACEMENT_HW_BILINEAR)
    float disp = PtexLookupFast(patchCoord,
                                textureDisplace_Data,
                                textureDisplace_Packing).x;
#elif defined(DISPLACEMENT_BILINEAR)
    float disp = PtexMipmapLookup(patchCoord, 
                                  mipmapBias,
                                  textureDisplace_Data,
                                  textureDisplace_Packing).x;
#elif defined(DISPLACEMENT_BIQUADRATIC)
    float disp = PtexMipmapLookupQuadratic(patchCoord, 
                                           mipmapBias,
                                           textureDisplace_Data,
                                           textureDisplace_Packing).x;
#else
    float disp = 0.0;
#endif
    return position + vec4(disp * normal, 0) * displacementScale;
}

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;
layout (location=1) in vec3 normal;

out vec4 vPosition;
out vec3 vNormal;

void main()
{
    vPosition = ModelViewMatrix * position;
    vNormal = (ModelViewMatrix * vec4(normal, 0)).xyz;
}

#endif

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

#ifdef PRIM_QUAD

    layout(lines_adjacency) in;

    layout(triangle_strip, max_vertices = 4) out;

    #define EDGE_VERTS 4

    in vec4 vPosition[4];
    in vec3 vNormal[4];

#endif // PRIM_QUAD

#ifdef  PRIM_TRI

    layout(triangles) in;

    layout(triangle_strip, max_vertices = 3) out;

    #define EDGE_VERTS 3

    in vec4 vPosition[3];
    in vec3 vNormal[3];

#endif // PRIM_TRI

out vec4 gPosition;
out vec4 gPatchCoord;
out vec3 gNormal;
noperspective out vec4 gEdgeDistance;

// --------------------------------------

void emit(int index, vec4 position, vec3 normal, vec4 patchCoord)
{
    gPosition = position;
    gPatchCoord = patchCoord;
    gNormal = normal;

    gl_Position = ProjectionMatrix * gPosition;
    EmitVertex();
}

const float VIEWPORT_SCALE = 1024.0; // XXXdyu

float edgeDistance(vec4 p, vec4 p0, vec4 p1)
{
    return VIEWPORT_SCALE *
        abs((p.x - p0.x) * (p1.y - p0.y) -
            (p.y - p0.y) * (p1.x - p0.x)) / length(p1.xy - p0.xy);
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
void emit(int index, vec4 position, vec3 normal, vec4 patchCoord, vec4 edgeVerts[EDGE_VERTS])
{
    gEdgeDistance[0] =
        edgeDistance(edgeVerts[index], edgeVerts[0], edgeVerts[1]);
    gEdgeDistance[1] =
        edgeDistance(edgeVerts[index], edgeVerts[1], edgeVerts[2]);
#ifdef PRIM_TRI
    gEdgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[0]);
#endif
#ifdef PRIM_QUAD
    gEdgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[3]);
    gEdgeDistance[3] =
        edgeDistance(edgeVerts[index], edgeVerts[3], edgeVerts[0]);
#endif

    emit(index, position, normal, patchCoord);
}
#endif

// --------------------------------------

vec4 GeneratePatchCoord(vec2 uv, int primitiveID) // for non-adaptive
{
    ivec3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(primitiveID));
    return OsdInterpolatePatchCoord(uv, patchParam);
}

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

#ifdef PRIM_QUAD
    vec4 patchCoord[4];
    vec4 position[4];
    vec3 normal[4];

    // need to generate patch coord for non-patch quads
    patchCoord[0] = GeneratePatchCoord(vec2(0, 0), gl_PrimitiveID);
    patchCoord[1] = GeneratePatchCoord(vec2(1, 0), gl_PrimitiveID);
    patchCoord[2] = GeneratePatchCoord(vec2(1, 1), gl_PrimitiveID);
    patchCoord[3] = GeneratePatchCoord(vec2(0, 1), gl_PrimitiveID);

#if defined(DISPLACEMENT_HW_BILINEAR) || defined(DISPLACEMENT_BILINEAR) || defined(DISPLACEMENT_BIQUADRATIC)
    position[0] = displacement(vPosition[0], vNormal[0], patchCoord[0]);
    position[1] = displacement(vPosition[1], vNormal[1], patchCoord[1]);
    position[2] = displacement(vPosition[2], vNormal[2], patchCoord[2]);
    position[3] = displacement(vPosition[3], vNormal[3], patchCoord[3]);
#else
    position[0] = vPosition[0];
    position[1] = vPosition[1];
    position[2] = vPosition[2];
    position[3] = vPosition[3];
#endif

#ifdef NORMAL_FACET
    // emit flat normals for displaced surface
    vec3 A = (position[0] - position[1]).xyz;
    vec3 B = (position[2] - position[1]).xyz;
    normal[0] = normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
    normal[3] = normal[0];
#else
    normal[0] = vNormal[0];
    normal[1] = vNormal[1];
    normal[2] = vNormal[2];
    normal[3] = vNormal[3];
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * vPosition[0];
    edgeVerts[1] = ProjectionMatrix * vPosition[1];
    edgeVerts[2] = ProjectionMatrix * vPosition[2];
    edgeVerts[3] = ProjectionMatrix * vPosition[3];

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;
    edgeVerts[3].xy /= edgeVerts[3].w;

    emit(0, position[0], normal[0], patchCoord[0], edgeVerts);
    emit(1, position[1], normal[1], patchCoord[1], edgeVerts);
    emit(3, position[3], normal[3], patchCoord[3], edgeVerts);
    emit(2, position[2], normal[2], patchCoord[2], edgeVerts);
#else
    gEdgeDistance = vec4(0);
    emit(0, position[0], normal[0], patchCoord[0]);
    emit(1, position[1], normal[1], patchCoord[1]);
    emit(3, position[3], normal[3], patchCoord[3]);
    emit(2, position[2], normal[2], patchCoord[2]);
#endif
#endif // PRIM_QUAD

#ifdef PRIM_TRI
    vec4 position[3];
    vec4 patchCoord[3];
    vec3 normal[3];

    // patch coords are computed in tessellation shader
    patchCoord[0] = vPatchCoord[0];
    patchCoord[1] = vPatchCoord[1];
    patchCoord[2] = vPatchCoord[2];

#if defined(DISPLACEMENT_HW_BILINEAR) || defined(DISPLACEMENT_BILINEAR) || defined(DISPLACEMENT_BIQUADRATIC)
    position[0] = displacement(vPosition[0], vNormal[0], patchCoord[0]);
    position[1] = displacement(vPosition[1], vNormal[1], patchCoord[1]);
    position[2] = displacement(vPosition[2], vNormal[2], patchCoord[2]);
#else
    position[0] = vPosition[0];
    position[1] = vPosition[1];
    position[2] = vPosition[2];
#endif

#ifdef NORMAL_FACET
    // emit flat normals for displaced surface
    vec3 A = (position[0] - position[1]).xyz;
    vec3 B = (position[2] - position[1]).xyz;
    normal[0] = normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
#else
    normal[0] = vNormal[0];
    normal[1] = vNormal[1];
    normal[2] = vNormal[2];
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * vPosition[0];
    edgeVerts[1] = ProjectionMatrix * vPosition[1];
    edgeVerts[2] = ProjectionMatrix * vPosition[2];

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;

    emit(0, position[0], normal[0], patchCoord[0], edgeVerts);
    emit(1, position[1], normal[1], patchCoord[1], edgeVerts);
    emit(2, position[2], normal[2], patchCoord[2], edgeVerts);
#else
    gEdgeDistance = vec4(0);
    emit(0, position[0], normal[0], patchCoord[0]);
    emit(1, position[1], normal[1], patchCoord[1]);
    emit(2, position[2], normal[2], patchCoord[2]);
#endif
#endif // PRIM_TRI

    EndPrimitive();
}

#endif
//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

in vec4 gPosition;
in vec3 gNormal;
in vec4 gPatchCoord;
noperspective in vec4 gEdgeDistance;
out vec4 outColor;

#if defined(COLOR_PTEX_NEAREST) || defined(COLOR_PTEX_HW_BILINEAR) || \
    defined(COLOR_PTEX_BILINEAR) || defined(COLOR_PTEX_BIQUADRATIC)
uniform sampler2DArray textureImage_Data;
uniform isamplerBuffer textureImage_Packing;
#endif

#ifdef USE_PTEX_OCCLUSION
uniform sampler2DArray textureOcclusion_Data;
uniform samplerBuffer textureOcclusion_Packing;
#endif

#ifdef USE_PTEX_SPECULAR
uniform sampler2DArray textureSpecular_Data;
uniform samplerBuffer textureSpecular_Packing;
#endif

uniform bool overrideColorEnable = false;
uniform vec4 overrideColor;

#if defined(NORMAL_HW_SCREENSPACE) || defined(NORMAL_SCREENSPACE)

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
#if defined NORMAL_HW_SCREENSPACE
    float Hll = PtexLookupFast(STll, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hlr = PtexLookupFast(STlr, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hul = PtexLookupFast(STul, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
#elif defined NORMAL_SCREENSPACE
    float Hll = PtexMipmapLookup(STll, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hlr = PtexMipmapLookup(STlr, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hul = PtexMipmapLookup(STul, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
#endif
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
lighting(vec4 texColor, vec3 Peye, vec3 Neye, float spec, float occ)
{
    vec4 color = vec4(0);
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
                               spec * s * lightSource[i].specular);
    }

    color.a = 1;

    return color;
}

vec4
edgeColor(vec4 Cfill)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float d =
        min(gEdgeDistance[0], min(gEdgeDistance[1], gEdgeDistance[2]));
#endif
#ifdef PRIM_QUAD
    float d =
        min(min(gEdgeDistance[0], gEdgeDistance[1]),
            min(gEdgeDistance[2], gEdgeDistance[3]));
#endif
#ifdef PRIM_LINE
    float d = 0;
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
    // ------------ normal ---------------

#if defined(NORMAL_HW_SCREENSPACE) || defined(NORMAL_SCREENSPACE)
    vec3 normal = perturbNormalFromDisplacement(gPosition.xyz,
                                                gNormal,
                                                gPatchCoord);
#else
    vec3 normal = gNormal;
#endif

    // ------------ color ---------------

#if defined COLOR_PTEX_NEAREST
    vec4 texColor = PtexLookupNearest(gPatchCoord,
                                      textureImage_Data,
                                      textureImage_Packing);
#elif defined COLOR_PTEX_HW_BILINEAR
    vec4 texColor = PtexLookupFast(gPatchCoord,
                                   textureImage_Data,
                                   textureImage_Packing);
#elif defined COLOR_PTEX_BILINEAR
    vec4 texColor = PtexMipmapLookup(gPatchCoord,
                                     mipmapBias,
                                     textureImage_Data,
                                     textureImage_Packing);
#elif defined COLOR_PTEX_BIQUADRATIC
    vec4 texColor = PtexMipmapLookupQuadratic(gPatchCoord,
                                              mipmapBias,
                                              textureImage_Data,
                                              textureImage_Packing);
#elif defined COLOR_PATCHTYPE
    vec4 texColor = edgeColor(lighting(GetOverrideColor(OsdGetPatchParam(OsdGetPatchIndex(gl_PrimitiveID))),
                                       gPosition.xyz, normal, 1, 0));
    outColor = texColor;
    return;
#elif defined COLOR_PATCHCOORD
    vec4 texColor = edgeColor(lighting(gPatchCoord, gPosition.xyz, normal, 1, 0));
    outColor = texColor;
    return;
#elif defined COLOR_NORMAL
    vec4 texColor = edgeColor(vec4(normal, 1));
    outColor = texColor;
    return;
#else // COLOR_NONE
    vec4 texColor = vec4(0.5);
#endif

    // ------------ occlusion ---------------

#ifdef USE_PTEX_OCCLUSION
    float occ = PtexMipmapLookup(gPatchCoord,
                                 mipmapBias,
                                 textureOcclusion_Data,
                                 textureOcclusion_Packing).x;
#else
    float occ = 0.0;
#endif

    // ------------ specular ---------------

#ifdef USE_PTEX_SPECULAR
    float specular = PtexMipmapLookup(gPatchCoord,
                                      mipmapBias,
                                      textureSpecular_Data,
                                      textureSpecular_Packing).x;
#else
    float specular = 1.0;
#endif

    vec4 Cf = lighting(texColor, gPosition.xyz, normal, specular, occ);

    // ------------ wireframe ---------------

    outColor = edgeColor(Cf);
}

#endif
