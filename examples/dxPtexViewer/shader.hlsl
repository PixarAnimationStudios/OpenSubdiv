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

struct OutputPointVertex {
    float4 positionOut : SV_Position;
};

cbuffer Transform : register( b0 ) {
    float4x4 ModelViewMatrix;
    float4x4 ProjectionMatrix;
    float4x4 ModelViewProjectionMatrix;
    float4x4 ModelViewInverseMatrix;
};

cbuffer Tessellation : register( b1 ) {
    float TessLevel;
    int PrimitiveIdBase;
};

cbuffer Config : register( b3 ) {
    float displacementScale;
    float mipmapBias;
};

float4x4 OsdModelViewMatrix()
{
    return ModelViewMatrix;
}
float4x4 OsdProjectionMatrix()
{
    return ProjectionMatrix;
}
float4x4 OsdModelViewProjectionMatrix()
{
    return ModelViewProjectionMatrix;
}
float OsdTessLevel()
{
    return TessLevel;
}
int OsdGregoryQuadOffsetBase()
{
    return 0;
}
int OsdPrimitiveIdBase()
{
    return PrimitiveIdBase;
}

// ---------------------------------------------------------------------------

#if    defined(DISPLACEMENT_HW_BILINEAR)        \
    || defined(DISPLACEMENT_BILINEAR)           \
    || defined(DISPLACEMENT_BIQUADRATIC)        \
    || defined(NORMAL_HW_SCREENSPACE)           \
    || defined(NORMAL_SCREENSPACE)              \
    || defined(NORMAL_BIQUADRATIC)              \
    || defined(NORMAL_BIQUADRATIC_WG)

Texture2DArray textureDisplace_Data : register(t6);
Buffer<uint> textureDisplace_Packing : register(t7);
#endif

#if defined(DISPLACEMENT_HW_BILINEAR) \
    || defined(DISPLACEMENT_BILINEAR) \
    || defined(DISPLACEMENT_BIQUADRATIC)

#undef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK              \
    output.position =                          \
        displacement(output.position,          \
                     output.normal,            \
                     output.patchCoord);

float4 displacement(float4 position, float3 normal, float4 patchCoord)
{
#if defined(DISPLACEMENT_HW_BILINEAR)
    float disp = PtexLookupFast(patchCoord,
                                textureDisplace_Data,
                                textureDisplace_Packing).x;
#elif defined(DISPLACEMENT_BILINEAR)
    float disp = PtexMipmapLookup(patchCoord, mipmapBias,
                                  textureDisplace_Data,
                                  textureDisplace_Packing).x;
#elif defined(DISPLACEMENT_BIQUADRATIC)
    float disp = PtexMipmapLookupQuadratic(patchCoord, mipmapBias,
                                           textureDisplace_Data,
                                           textureDisplace_Packing).x;
#else
    float disp(0);
#endif
    return position + float4(disp*normal, 0) * displacementScale;
}
#endif

float4 GeneratePatchCoord(float2 uv, int primitiveID)  // for non-adaptive
{
    int3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(primitiveID));
    return OsdInterpolatePatchCoord(uv, patchParam);
}

// ---------------------------------------------------------------------------
//  Vertex Shader
// ---------------------------------------------------------------------------

void vs_main( in InputVertex input,
              out OutputVertex output )
{
    output.positionOut = mul(ModelViewProjectionMatrix, input.position);
    output.position = mul(ModelViewMatrix, input.position);
    output.normal = mul(ModelViewMatrix,float4(input.normal, 0)).xyz;

    output.patchCoord = float4(0,0,0,0);
    output.tangent = float3(0,0,0);
    output.bitangent = float3(0,0,0);
    output.edgeDistance = float4(0,0,0,0);
}

// ---------------------------------------------------------------------------
//  Geometry Shader
// ---------------------------------------------------------------------------

struct GS_OUT
{
    OutputVertex v;
    uint primitiveID : SV_PrimitiveID;
};

GS_OUT
outputVertex(OutputVertex input, float3 normal, uint primitiveID)
{
    GS_OUT gsout;
    gsout.v = input;
    gsout.v.normal = normal;
    gsout.primitiveID = primitiveID;
    return gsout;
}

GS_OUT
outputVertex(OutputVertex input, float3 normal, float4 patchCoord, uint primitiveID)
{
    GS_OUT gsout;
    gsout.v = input;
    gsout.v.normal = normal;
    gsout.v.patchCoord = patchCoord;
    gsout.primitiveID = primitiveID;
    return gsout;
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    #define EDGE_VERTS 3
#endif
#ifdef PRIM_QUAD
    #define EDGE_VERTS 4
#endif

static float VIEWPORT_SCALE = 1024.0; // XXXdyu

float edgeDistance(float2 p, float2 p0, float2 p1)
{
    return VIEWPORT_SCALE *
        abs((p.x - p0.x) * (p1.y - p0.y) -
            (p.y - p0.y) * (p1.x - p0.x)) / length(p1.xy - p0.xy);
}

GS_OUT
outputWireVertex(OutputVertex input, float3 normal,
                 int index, float2 edgeVerts[EDGE_VERTS], uint primitiveID)
{
    GS_OUT gsout;
    gsout.v = input;
    gsout.v.normal = normal;

    gsout.v.edgeDistance[0] =
        edgeDistance(edgeVerts[index], edgeVerts[0], edgeVerts[1]);
    gsout.v.edgeDistance[1] =
        edgeDistance(edgeVerts[index], edgeVerts[1], edgeVerts[2]);
#ifdef PRIM_TRI
    gsout.v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[0]);
#endif
#ifdef PRIM_QUAD
    gsout.v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[3]);
    gsout.v.edgeDistance[3] =
        edgeDistance(edgeVerts[index], edgeVerts[3], edgeVerts[0]);
#endif
    gsout.primitiveID = primitiveID;
    return gsout;
}
#endif

#ifdef PRIM_QUAD
[maxvertexcount(6)]
void gs_main( lineadj OutputVertex input[4],
              inout TriangleStream<GS_OUT> triStream,
              uint primitiveID : SV_PrimitiveID)
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[3].position - input[1].position).xyz;
    float3 C = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    float4 patchCoord[4];
    patchCoord[0] = GeneratePatchCoord(float2(0, 0), primitiveID);
    patchCoord[1] = GeneratePatchCoord(float2(1, 0), primitiveID);
    patchCoord[2] = GeneratePatchCoord(float2(1, 1), primitiveID);
    patchCoord[3] = GeneratePatchCoord(float2(0, 1), primitiveID);

    triStream.Append(outputVertex(input[0], n0, patchCoord[0], primitiveID));
    triStream.Append(outputVertex(input[1], n0, patchCoord[1], primitiveID));
    triStream.Append(outputVertex(input[3], n0, patchCoord[3], primitiveID));
    triStream.RestartStrip();
    triStream.Append(outputVertex(input[3], n0, patchCoord[3], primitiveID));
    triStream.Append(outputVertex(input[1], n0, patchCoord[1], primitiveID));
    triStream.Append(outputVertex(input[2], n0, patchCoord[2], primitiveID));
    triStream.RestartStrip();
}
#else // PRIM_TRI
[maxvertexcount(3)]
void gs_main( triangle OutputVertex input[3],
              inout TriangleStream<GS_OUT> triStream,
              uint primitiveID : SV_PrimitiveID)
{
    float4 position[3];
    float4 patchCoord[3];
    float3 normal[3];

    // patch coords are computed in tessellation shader
    patchCoord[0] = input[0].patchCoord;
    patchCoord[1] = input[1].patchCoord;
    patchCoord[2] = input[2].patchCoord;

    position[0] = input[0].position;
    position[1] = input[1].position;
    position[2] = input[2].position;

#ifdef NORMAL_FACET
    // emit flat normals for displaced surface
    float3 A = (position[0] - position[1]).xyz;
    float3 B = (position[2] - position[1]).xyz;
    normal[0]= normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
#else
    normal[0] = input[0].normal;
    normal[1] = input[1].normal;
    normal[2] = input[2].normal;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    float2 edgeVerts[3];
    edgeVerts[0] = input[0].positionOut.xy / input[0].positionOut.w;
    edgeVerts[1] = input[1].positionOut.xy / input[1].positionOut.w;
    edgeVerts[2] = input[2].positionOut.xy / input[2].positionOut.w;

    triStream.Append(outputWireVertex(input[0], normal[0], 0, edgeVerts, primitiveID));
    triStream.Append(outputWireVertex(input[1], normal[1], 1, edgeVerts, primitiveID));
    triStream.Append(outputWireVertex(input[2], normal[2], 2, edgeVerts, primitiveID));
#else
    triStream.Append(outputVertex(input[0], normal[0], primitiveID));
    triStream.Append(outputVertex(input[1], normal[1], primitiveID));
    triStream.Append(outputVertex(input[2], normal[2], primitiveID));
#endif
}

#endif


// ---------------------------------------------------------------------------
//  IBL lighting
// ---------------------------------------------------------------------------

Texture2D diffuseEnvironmentMap : register(t12);
Texture2D specularEnvironmentMap : register(t13);

SamplerState iblSampler : register(s0);

#define M_PI 3.14159265358

float4
gamma(float4 value, float g) {
    return float4(pow(value.xyz, float3(g,g,g)), 1);
}

float4
getEnvironmentHDR(Texture2D tx, SamplerState sm, float3 dir)
{
    dir = mul(ModelViewInverseMatrix, float4(dir, 0)).xyz;
    float2 uv = float2((atan2(dir.x,dir.z)/M_PI+1)*0.5, (1-dir.y)*0.5);
    return tx.Sample(sm, uv);
}


// ---------------------------------------------------------------------------
//  Lighting
// ---------------------------------------------------------------------------

#define NUM_LIGHTS 2

struct LightSource {
    float4 position;
    float4 ambient;
    float4 diffuse;
    float4 specular;
};

cbuffer Lighting : register( b2 ) {
    LightSource lightSource[NUM_LIGHTS];
};

float4
lighting(float4 texColor, float3 Peye, float3 Neye, float occ)
{
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    float3 n = Neye;

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        float4 Plight = lightSource[i].position;
        float3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        float3 h = normalize(l + float3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 64.0f);

        color += (1.0 - occ) * ((lightSource[i].ambient +
                                 d * lightSource[i].diffuse) * texColor +
                                s * lightSource[i].specular);
    }

    color.a = 1.0;
    return color;
}

// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------

float4
edgeColor(float4 Cfill, float4 edgeDistance)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float d =
        min(edgeDistance[0], min(edgeDistance[1], edgeDistance[2]));
#endif
#ifdef PRIM_QUAD
    float d =
        min(min(edgeDistance[0], edgeDistance[1]),
            min(edgeDistance[2], edgeDistance[3]));
#endif
    float4 Cedge = float4(1.0, 1.0, 0.0, 1.0);
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    Cfill.rgb = lerp(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------


#if defined(COLOR_PTEX_NEAREST) ||     \
    defined(COLOR_PTEX_HW_BILINEAR) || \
    defined(COLOR_PTEX_BILINEAR) ||    \
    defined(COLOR_PTEX_BIQUADRATIC)
Texture2DArray textureImage_Data : register(t4);
Buffer<uint> textureImage_Packing : register(t5);
#endif

#ifdef USE_PTEX_OCCLUSION
Texture2DArray textureOcclusion_Data : register(t8);
Buffer<uint> textureOcclusion_Packing : register(t9);
#endif

#ifdef USE_PTEX_SPECULAR
Texture2DArray textureSpecular_Data : register(t10);
Buffer<uint> textureSpecular_Packing : register(t11);
#endif

float4
getAdaptivePatchColor(int3 patchParam, float sharpness)
{
    const float4 patchColors[7*6] = {
        float4(1.0f,  1.0f,  1.0f,  1.0f),   // regular
        float4(0.0f,  1.0f,  1.0f,  1.0f),   // regular pattern 0
        float4(0.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 1
        float4(0.0f,  0.5f,  0.5f,  1.0f),   // regular pattern 2
        float4(0.5f,  0.0f,  1.0f,  1.0f),   // regular pattern 3
        float4(1.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 4

        float4(1.0f,  0.5f,  0.5f,  1.0f),   // single crease
        float4(1.0f,  0.70f,  0.6f,  1.0f),  // single crease pattern 0
        float4(1.0f,  0.65f,  0.6f,  1.0f),  // single crease pattern 1
        float4(1.0f,  0.60f,  0.6f,  1.0f),  // single crease pattern 2
        float4(1.0f,  0.55f,  0.6f,  1.0f),  // single crease pattern 3
        float4(1.0f,  0.50f,  0.6f,  1.0f),  // single crease pattern 4

        float4(0.8f,  0.0f,  0.0f,  1.0f),   // boundary
        float4(0.0f,  0.0f,  0.75f, 1.0f),   // boundary pattern 0
        float4(0.0f,  0.2f,  0.75f, 1.0f),   // boundary pattern 1
        float4(0.0f,  0.4f,  0.75f, 1.0f),   // boundary pattern 2
        float4(0.0f,  0.6f,  0.75f, 1.0f),   // boundary pattern 3
        float4(0.0f,  0.8f,  0.75f, 1.0f),   // boundary pattern 4

        float4(0.0f,  1.0f,  0.0f,  1.0f),   // corner
        float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 0
        float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 1
        float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 2
        float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 3
        float4(0.25f, 0.25f, 0.25f, 1.0f),   // corner pattern 4

        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory

        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary

        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f)    // gregory basis
    };

    int patchType = 0;

    int edgeCount = countbits(OsdGetPatchBoundaryMask(patchParam));
    if (edgeCount == 1) {
        patchType = 2; // BOUNDARY
    }
    if (edgeCount == 2) {
        patchType = 3; // CORNER
    }

#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    if (sharpness > 0) {
        patchType = 1;
    }
#elif defined OSD_PATCH_GREGORY
    patchType = 4;
#elif defined OSD_PATCH_GREGORY_BOUNDARY
    patchType = 5;
#elif defined OSD_PATCH_GREGORY_BASIS
    patchType = 6;
#endif

    int pattern = countbits(OsdGetPatchTransitionMask(patchParam));

    return patchColors[6*patchType + pattern];
}

void
ps_main(in OutputVertex input,
        uint primitiveID : SV_PrimitiveID,
        out float4 outColor : SV_Target )
{
    // ------------ normal ---------------
#if defined(NORMAL_HW_SCREENSPACE) || defined(NORMAL_SCREENSPACE)
    float3 normal = perturbNormalFromDisplacement(input.position.xyz,
                                                  input.normal,
                                                  input.patchCoord);
#elif defined(NORMAL_BIQUADRATIC) || defined(NORMAL_BIQUADRATIC_WG)
    float4 du, dv;
    float4 disp = PtexMipmapLookupQuadratic(du, dv, input.patchCoord,
                                            mipmapBias,
                                            textureDisplace_Data,
                                            textureDisplace_Packing);

    disp *= displacementScale;
    du *= displacementScale;
    dv *= displacementScale;

    float3 n = normalize(cross(input.tangent, input.bitangent));
    float3 tangent = input.tangent + n * du.x;
    float3 bitangent = input.bitangent + n * dv.x;

#if defined(NORMAL_BIQUADRATIC_WG)
    tangent += input.Nu * disp.x;
    bitangent += input.Nv * disp.x;
#endif

    float3 normal = normalize(cross(tangent, bitangent));
#else
    float3 normal = input.normal;
#endif

    // ------------ color ---------------
#if defined(COLOR_PTEX_NEAREST)
    float4 texColor = PtexLookupNearest(input.patchCoord,
                                        textureImage_Data,
                                        textureImage_Packing);
#elif defined(COLOR_PTEX_HW_BILINEAR)
    float4 texColor = PtexLookupFast(input.patchCoord,
                                   textureImage_Data,
                                   textureImage_Packing);
#elif defined(COLOR_PTEX_BILINEAR)
    float4 texColor = PtexMipmapLookup(input.patchCoord, mipmapBias,
                                     textureImage_Data,
                                     textureImage_Packing);
#elif defined(COLOR_PTEX_BIQUADRATIC)
    float4 texColor = PtexMipmapLookupQuadratic(input.patchCoord, mipmapBias,
                                              textureImage_Data,
                                              textureImage_Packing);
#elif defined(COLOR_PATCHTYPE)
    float4 patchColor = getAdaptivePatchColor(
        OsdGetPatchParam(OsdGetPatchIndex(primitiveID)), 0);
    float4 texColor = edgeColor(lighting(patchColor, input.position.xyz, normal, 0),
                                input.edgeDistance);
    outColor = texColor;
    return;
#elif defined(COLOR_PATCHCOORD)
    float4 texColor = edgeColor(lighting(input.patchCoord, input.position.xyz, normal, 0),
                                input.edgeDistance);
    outColor = texColor;
    return;
#elif defined(COLOR_NORMAL)
    float4 texColor = edgeColor(float4(normal.x, normal.y, normal.z, 1),
                                input.edgeDistance);
    outColor = texColor;
    return;
#else // COLOR_NONE
    float4 texColor = float4(0.5, 0.5, 0.5, 1);
#endif

    // ------------ occlusion ---------------

#ifdef USE_PTEX_OCCLUSION
    float occ = PtexMipmapLookup(input.patchCoord, mipmapBias,
                                 textureOcclusion_Data,
                                 textureOcclusion_Packing).x;
#else
    float occ = 0.0;
#endif

    // ------------ specular ---------------

#ifdef USE_PTEX_SPECULAR
    float specular = PtexMipmapLookup(input.patchCoord, mipmapBias,
                                      textureSpecular_Data,
                                      textureSpecular_Packing).x;
#else
    float specular = 1.0;
#endif

    // ------------ lighting ---------------
#ifdef USE_IBL
    // non-plausible BRDF
    float4 a = float4(0, 0, 0, 1); //ambientColor;
    float4 d = getEnvironmentHDR(diffuseEnvironmentMap, iblSampler, normal);

    float3 eye = normalize(input.position.xyz - float3(0,0,0));
    float3 r = reflect(eye, normal);
    float4 s = getEnvironmentHDR(specularEnvironmentMap, iblSampler, r);

    const float fresnelBias = 0.01;
    const float fresnelScale = 1.0;
    const float fresnelPower = 3.5;
    float F = fresnelBias + fresnelScale * pow(1.0+dot(normal,eye), fresnelPower);

    // Geometric attenuation term (
    float NoV = dot(normal, -eye);
    float alpha = 0.75 * 0.75; // roughness ^ 2
    float k = alpha * 0.5;
    float G = NoV/(NoV*(1-k)+k);

    a *= (1-occ);
    d *= (1-occ);
    s *= min(specular, (1-occ)) * (F*G);

    float4 Cf = (a+d)*texColor*(1-F)/M_PI + s;
    //Cf = gamma(Cf, 2.2);

#else
    float4 Cf = lighting(texColor, input.position.xyz, normal, occ);
#endif

    // ------------ wireframe ---------------
    outColor = edgeColor(Cf, input.edgeDistance);
}
