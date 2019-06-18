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

#if defined(SHADING_VARYING_COLOR) || defined(SHADING_FACEVARYING_COLOR)
#undef OSD_USER_VARYING_DECLARE
#define OSD_USER_VARYING_DECLARE \
    vec3 color;

#undef OSD_USER_VARYING_ATTRIBUTE_DECLARE
#define OSD_USER_VARYING_ATTRIBUTE_DECLARE \
    layout(location = 1) in vec3 color;

#undef OSD_USER_VARYING_PER_VERTEX
#define OSD_USER_VARYING_PER_VERTEX() \
    outpt.color = color

#undef OSD_USER_VARYING_PER_CONTROL_POINT
#define OSD_USER_VARYING_PER_CONTROL_POINT(ID_OUT, ID_IN) \
    outpt[ID_OUT].color = inpt[ID_IN].color

#undef OSD_USER_VARYING_PER_EVAL_POINT
#define OSD_USER_VARYING_PER_EVAL_POINT(UV, a, b, c, d) \
    outpt.color = \
        mix(mix(inpt[a].color, inpt[b].color, UV.x), \
            mix(inpt[c].color, inpt[d].color, UV.x), UV.y)

#undef OSD_USER_VARYING_PER_EVAL_POINT_TRIANGLE
#define OSD_USER_VARYING_PER_EVAL_POINT_TRIANGLE(UV, a, b, c) \
    outpt.color = \
        inpt[a].color * (1.0f - UV.x - UV.y) + \
        inpt[b].color * UV.x + \
        inpt[c].color * UV.y;
#else
#define OSD_USER_VARYING_DECLARE
#define OSD_USER_VARYING_ATTRIBUTE_DECLARE
#define OSD_USER_VARYING_PER_VERTEX()
#define OSD_USER_VARYING_PER_CONTROL_POINT(ID_OUT, ID_IN)
#define OSD_USER_VARYING_PER_EVAL_POINT(UV, a, b, c, d)
#define OSD_USER_VARYING_PER_EVAL_POINT_TRIANGLE(UV, a, b, c)
#endif

//--------------------------------------------------------------
// Uniforms / Uniform Blocks
//--------------------------------------------------------------

layout(std140) uniform Transform {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
    mat4 ModelViewInverseMatrix;
};

layout(std140) uniform Tessellation {
    float TessLevel;
};

uniform int GregoryQuadOffsetBase;
uniform int PrimitiveIdBase;

//--------------------------------------------------------------
// Osd external functions
//--------------------------------------------------------------

mat4 OsdModelViewMatrix()
{
    return ModelViewMatrix;
}
mat4 OsdProjectionMatrix()
{
    return ProjectionMatrix;
}
mat4 OsdModelViewProjectionMatrix()
{
    return ModelViewProjectionMatrix;
}
float OsdTessLevel()
{
    return TessLevel;
}
int OsdGregoryQuadOffsetBase()
{
    return GregoryQuadOffsetBase;
}
int OsdPrimitiveIdBase()
{
    return PrimitiveIdBase;
}
int OsdBaseVertex()
{
    return 0;
}

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;
OSD_USER_VARYING_ATTRIBUTE_DECLARE

out block {
    OutputVertex v;
#ifdef OSD_PATCH_ENABLE_SINGLE_CREASE
    vec2 vSegments;
#endif
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    outpt.v.position = ModelViewMatrix * position;
    outpt.v.patchCoord = vec4(0);
#ifdef OSD_PATCH_ENABLE_SINGLE_CREASE
    outpt.vSegments = vec2(0);
#endif
    OSD_USER_VARYING_PER_VERTEX();
}

#endif

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

#ifdef PRIM_QUAD

    layout(lines_adjacency) in;

    #define EDGE_VERTS 4

#endif // PRIM_QUAD

#ifdef  PRIM_TRI

    layout(triangles) in;

    #define EDGE_VERTS 3

#endif // PRIM_TRI


layout(triangle_strip, max_vertices = EDGE_VERTS) out;
in block {
    OutputVertex v;
#ifdef OSD_PATCH_ENABLE_SINGLE_CREASE
    vec2 vSegments;
#endif
    OSD_USER_VARYING_DECLARE
} inpt[EDGE_VERTS];

out block {
    OutputVertex v;
    noperspective out vec4 edgeDistance;
#ifdef OSD_PATCH_ENABLE_SINGLE_CREASE
    vec2 vSegments;
#endif
    OSD_USER_VARYING_DECLARE
} outpt;

uniform isamplerBuffer OsdFVarParamBuffer;
layout(std140) uniform OsdFVarArrayData {
    OsdPatchArray fvarPatchArray[2];
};

vec2
interpolateFaceVarying(vec2 uv, int fvarOffset)
{
    int patchIndex = OsdGetPatchIndex(gl_PrimitiveID);

    OsdPatchArray array = fvarPatchArray[0];

    ivec3 fvarPatchParam = texelFetch(OsdFVarParamBuffer, patchIndex).xyz;
    OsdPatchParam param = OsdPatchParamInit(fvarPatchParam.x,
                                            fvarPatchParam.y,
                                            fvarPatchParam.z);

    int patchType = OsdPatchParamIsRegular(param) ? array.regDesc : array.desc;

    float wP[20], wDu[20], wDv[20], wDuu[20], wDuv[20], wDvv[20];
    int numPoints = OsdEvaluatePatchBasisNormalized(patchType, param,
                uv.s, uv.t, wP, wDu, wDv, wDuu, wDuv, wDvv);

    int patchArrayStride = numPoints;

    int primOffset = patchIndex * patchArrayStride;

    vec2 result = vec2(0);
    for (int i=0; i<numPoints; ++i) {
        int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;
        vec2 cv = vec2(texelFetch(OsdFVarDataBuffer, index).s,
                       texelFetch(OsdFVarDataBuffer, index + 1).s);
        result += wP[i] * cv;
    }

    return result;
}

void emit(int index, vec3 normal)
{
    outpt.v.position = inpt[index].v.position;
    outpt.v.patchCoord = inpt[index].v.patchCoord;
#ifdef SMOOTH_NORMALS
    outpt.v.normal = inpt[index].v.normal;
#else
    outpt.v.normal = normal;
#endif

#ifdef OSD_PATCH_ENABLE_SINGLE_CREASE
    outpt.vSegments = inpt[index].vSegments;
#endif

#ifdef SHADING_VARYING_COLOR
    outpt.color = inpt[index].color;
#endif

#ifdef SHADING_FACEVARYING_COLOR
#ifdef SHADING_FACEVARYING_UNIFORM_SUBDIVISION
    // interpolate fvar data at refined tri or quad vertex locations
#ifdef PRIM_TRI
    vec2 trist[3] = vec2[](vec2(0,0), vec2(1,0), vec2(0,1));
    vec2 st = trist[index];
#endif
#ifdef PRIM_QUAD
    vec2 quadst[4] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,1));
    vec2 st = quadst[index];
#endif
#else
    // interpolate fvar data at tessellated vertex locations
    vec2 st = inpt[index].v.tessCoord;
#endif

    vec2 uv = interpolateFaceVarying(st, /*fvarOffset*/0);
    outpt.color = vec3(uv.s, uv.t, 0);
#endif

    gl_Position = ProjectionMatrix * inpt[index].v.position;
    EmitVertex();
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
const float VIEWPORT_SCALE = 1024.0; // XXXdyu

float edgeDistance(vec4 p, vec4 p0, vec4 p1)
{
    return VIEWPORT_SCALE *
        abs((p.x - p0.x) * (p1.y - p0.y) -
            (p.y - p0.y) * (p1.x - p0.x)) / length(p1.xy - p0.xy);
}

void emit(int index, vec3 normal, vec4 edgeVerts[EDGE_VERTS])
{
    outpt.edgeDistance[0] =
        edgeDistance(edgeVerts[index], edgeVerts[0], edgeVerts[1]);
    outpt.edgeDistance[1] =
        edgeDistance(edgeVerts[index], edgeVerts[1], edgeVerts[2]);
#ifdef PRIM_TRI
    outpt.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[0]);
#endif
#ifdef PRIM_QUAD
    outpt.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[3]);
    outpt.edgeDistance[3] =
        edgeDistance(edgeVerts[index], edgeVerts[3], edgeVerts[0]);
#endif

    emit(index, normal);
}
#endif

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

#ifdef PRIM_QUAD
    vec3 A = (inpt[0].v.position - inpt[1].v.position).xyz;
    vec3 B = (inpt[3].v.position - inpt[1].v.position).xyz;
    vec3 C = (inpt[2].v.position - inpt[1].v.position).xyz;
    vec3 n0 = normalize(cross(B, A));

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * inpt[0].v.position;
    edgeVerts[1] = ProjectionMatrix * inpt[1].v.position;
    edgeVerts[2] = ProjectionMatrix * inpt[2].v.position;
    edgeVerts[3] = ProjectionMatrix * inpt[3].v.position;

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;
    edgeVerts[3].xy /= edgeVerts[3].w;

    emit(0, n0, edgeVerts);
    emit(1, n0, edgeVerts);
    emit(3, n0, edgeVerts);
    emit(2, n0, edgeVerts);
#else
    emit(0, n0);
    emit(1, n0);
    emit(3, n0);
    emit(2, n0);
#endif
#endif // PRIM_QUAD

#ifdef PRIM_TRI
    vec3 A = (inpt[0].v.position - inpt[1].v.position).xyz;
    vec3 B = (inpt[2].v.position - inpt[1].v.position).xyz;
    vec3 n0 = normalize(cross(B, A));

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * inpt[0].v.position;
    edgeVerts[1] = ProjectionMatrix * inpt[1].v.position;
    edgeVerts[2] = ProjectionMatrix * inpt[2].v.position;

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;

    emit(0, n0, edgeVerts);
    emit(1, n0, edgeVerts);
    emit(2, n0, edgeVerts);
#else
    emit(0, n0);
    emit(1, n0);
    emit(2, n0);
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
    noperspective in vec4 edgeDistance;
#ifdef OSD_PATCH_ENABLE_SINGLE_CREASE
    vec2 vSegments;
#endif
    OSD_USER_VARYING_DECLARE
} inpt;

out vec4 outColor;

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

uniform vec4 diffuseColor = vec4(1);
uniform vec4 ambientColor = vec4(1);

vec4
lighting(vec4 diffuse, vec3 Peye, vec3 Neye)
{
    vec4 color = vec4(0);

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        vec4 Plight = lightSource[i].position;

        vec3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        vec3 n = normalize(Neye);
        vec3 h = normalize(l + vec3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 500.0f);

        color += lightSource[i].ambient * ambientColor
            + d * lightSource[i].diffuse * diffuse
            + s * lightSource[i].specular;
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
        min(inpt.edgeDistance[0], min(inpt.edgeDistance[1], inpt.edgeDistance[2]));
#endif
#ifdef PRIM_QUAD
    float d =
        min(min(inpt.edgeDistance[0], inpt.edgeDistance[1]),
            min(inpt.edgeDistance[2], inpt.edgeDistance[3]));
#endif
    float v = 0.8;
    vec4 Cedge = vec4(Cfill.r*v, Cfill.g*v, Cfill.b*v, 1);
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    Cfill.rgb = mix(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

vec4
getAdaptivePatchColor(ivec3 patchParam)
{
    const vec4 patchColors[7*6] = vec4[7*6](
        vec4(1.0f,  1.0f,  1.0f,  1.0f),   // regular
        vec4(0.0f,  1.0f,  1.0f,  1.0f),   // regular pattern 0
        vec4(0.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 1
        vec4(0.0f,  0.5f,  0.5f,  1.0f),   // regular pattern 2
        vec4(0.5f,  0.0f,  1.0f,  1.0f),   // regular pattern 3
        vec4(1.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 4

        vec4(1.0f,  0.5f,  0.5f,  1.0f),   // single crease
        vec4(1.0f,  0.70f,  0.6f,  1.0f),  // single crease pattern 0
        vec4(1.0f,  0.65f,  0.6f,  1.0f),  // single crease pattern 1
        vec4(1.0f,  0.60f,  0.6f,  1.0f),  // single crease pattern 2
        vec4(1.0f,  0.55f,  0.6f,  1.0f),  // single crease pattern 3
        vec4(1.0f,  0.50f,  0.6f,  1.0f),  // single crease pattern 4

        vec4(0.8f,  0.0f,  0.0f,  1.0f),   // boundary
        vec4(0.0f,  0.0f,  0.75f, 1.0f),   // boundary pattern 0
        vec4(0.0f,  0.2f,  0.75f, 1.0f),   // boundary pattern 1
        vec4(0.0f,  0.4f,  0.75f, 1.0f),   // boundary pattern 2
        vec4(0.0f,  0.6f,  0.75f, 1.0f),   // boundary pattern 3
        vec4(0.0f,  0.8f,  0.75f, 1.0f),   // boundary pattern 4

        vec4(0.0f,  1.0f,  0.0f,  1.0f),   // corner
        vec4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 0
        vec4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 1
        vec4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 2
        vec4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 3
        vec4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 4

        vec4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        vec4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        vec4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        vec4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        vec4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        vec4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory

        vec4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        vec4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        vec4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        vec4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        vec4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        vec4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary

        vec4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        vec4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        vec4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        vec4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        vec4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        vec4(1.0f,  0.7f,  0.3f,  1.0f)    // gregory basis
    );

    int patchType = 0;

    int edgeCount = bitCount(OsdGetPatchBoundaryMask(patchParam));
    if (edgeCount == 1) {
        patchType = 2; // BOUNDARY
    }
    if (edgeCount > 1) {
        patchType = 3; // CORNER (not correct for patches that are not isolated)
    }

#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    // check this after boundary/corner since single crease patch also has edgeCount.
    if (inpt.vSegments.y > 0) {
        patchType = 1;
    }
#elif defined OSD_PATCH_GREGORY
    patchType = 4;
#elif defined OSD_PATCH_GREGORY_BOUNDARY
    patchType = 5;
#elif defined OSD_PATCH_GREGORY_BASIS
    patchType = 6;
#elif defined OSD_PATCH_GREGORY_TRIANGLE
    patchType = 6;
#endif

    int pattern = bitCount(OsdGetPatchTransitionMask(patchParam));

    return patchColors[6*patchType + pattern];
}

vec4
getAdaptiveDepthColor(ivec3 patchParam)
{
    //  Represent depth with repeating cycle of four colors:
    const vec4 depthColors[4] = vec4[4](
        vec4(0.0f,  0.5f,  0.5f,  1.0f),
        vec4(1.0f,  1.0f,  1.0f,  1.0f),
        vec4(0.0f,  1.0f,  1.0f,  1.0f),
        vec4(0.5f,  1.0f,  0.5f,  1.0f)
    );
    return depthColors[OsdGetPatchRefinementLevel(patchParam) & 3];
}

#if defined(PRIM_QUAD) || defined(PRIM_TRI)
void
main()
{
    vec3 N = (gl_FrontFacing ? inpt.v.normal : -inpt.v.normal);

#if defined(SHADING_VARYING_COLOR)
    vec4 color = vec4(inpt.color, 1);
#elif defined(SHADING_FACEVARYING_COLOR)
    // generating a checkerboard pattern
    vec4 color = vec4(inpt.color.rg,
                      int(floor(20*inpt.color.r)+floor(20*inpt.color.g))&1, 1);
#elif defined(SHADING_PATCH_TYPE)
    vec4 color = getAdaptivePatchColor(OsdGetPatchParam(OsdGetPatchIndex(gl_PrimitiveID)));
#elif defined(SHADING_PATCH_DEPTH)
    vec4 color = getAdaptiveDepthColor(OsdGetPatchParam(OsdGetPatchIndex(gl_PrimitiveID)));
#elif defined(SHADING_PATCH_COORD)
    vec4 color = vec4(inpt.v.patchCoord.xy, 0, 1);
#elif defined(SHADING_MATERIAL)
    vec4 color = diffuseColor;
#else
    vec4 color = vec4(1, 1, 1, 1);
#endif

    vec4 Cf = lighting(color, inpt.v.position.xyz, N);

#if defined(SHADING_NORMAL)
    Cf.rgb = N;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    Cf = edgeColor(Cf, inpt.edgeDistance);
#endif

    outColor = Cf;
}
#endif

#endif
