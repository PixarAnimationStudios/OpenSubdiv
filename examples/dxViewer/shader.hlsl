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

struct OutputPointVertex {
    float4 positionOut : SV_Position;
};

// ---------------------------------------------------------------------------
//  Vertex Shader
// ---------------------------------------------------------------------------

void vs_main( in InputVertex input,
              out OutputVertex output )
{
    output.positionOut = mul(ModelViewProjectionMatrix, input.position);
    output.position = mul(ModelViewMatrix, input.position);
    output.normal = mul(ModelViewMatrix,float4(input.normal, 0)).xyz;
}

// ---------------------------------------------------------------------------
//  Geometry Shader
// ---------------------------------------------------------------------------

OutputVertex
outputVertex(OutputVertex input, float3 normal)
{
    OutputVertex v = input;
    v.normal = normal;
    return v;
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

OutputVertex
outputWireVertex(OutputVertex input, float3 normal,
                 int index, float2 edgeVerts[EDGE_VERTS])
{
    OutputVertex v = input;
    v.normal = normal;

    v.edgeDistance[0] =
        edgeDistance(edgeVerts[index], edgeVerts[0], edgeVerts[1]);
    v.edgeDistance[1] =
        edgeDistance(edgeVerts[index], edgeVerts[1], edgeVerts[2]);
#ifdef PRIM_TRI
    v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[0]);
#endif
#ifdef PRIM_QUAD
    v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[3]);
    v.edgeDistance[3] =
        edgeDistance(edgeVerts[index], edgeVerts[3], edgeVerts[0]);
#endif

    return v;
}
#endif

[maxvertexcount(6)]
void gs_quad( lineadj OutputVertex input[4],
              inout TriangleStream<OutputVertex> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[3].position - input[1].position).xyz;
    float3 C = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    triStream.Append(outputVertex(input[0], n0));
    triStream.Append(outputVertex(input[1], n0));
    triStream.Append(outputVertex(input[3], n0));
    triStream.RestartStrip();
    triStream.Append(outputVertex(input[3], n0));
    triStream.Append(outputVertex(input[1], n0));
    triStream.Append(outputVertex(input[2], n0));
    triStream.RestartStrip();
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_QUAD
[maxvertexcount(6)]
void gs_quad_wire( lineadj OutputVertex input[4],
              inout TriangleStream<OutputVertex> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[3].position - input[1].position).xyz;
    float3 C = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    float2 edgeVerts[4];
    edgeVerts[0] = input[0].positionOut.xy / input[0].positionOut.w;
    edgeVerts[1] = input[1].positionOut.xy / input[1].positionOut.w;
    edgeVerts[2] = input[2].positionOut.xy / input[2].positionOut.w;
    edgeVerts[3] = input[3].positionOut.xy / input[3].positionOut.w;

    triStream.Append(outputWireVertex(input[0], n0, 0, edgeVerts));
    triStream.Append(outputWireVertex(input[1], n0, 1, edgeVerts));
    triStream.Append(outputWireVertex(input[3], n0, 3, edgeVerts));
    triStream.RestartStrip();
    triStream.Append(outputWireVertex(input[3], n0, 3, edgeVerts));
    triStream.Append(outputWireVertex(input[1], n0, 1, edgeVerts));
    triStream.Append(outputWireVertex(input[2], n0, 2, edgeVerts));
    triStream.RestartStrip();
}
#endif
#endif

[maxvertexcount(3)]
void gs_triangle( triangle OutputVertex input[3],
                  inout TriangleStream<OutputVertex> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    triStream.Append(outputVertex(input[0], n0));
    triStream.Append(outputVertex(input[1], n0));
    triStream.Append(outputVertex(input[2], n0));
}

[maxvertexcount(3)]
void gs_triangle_smooth( triangle OutputVertex input[3],
                         inout TriangleStream<OutputVertex> triStream )
{
    triStream.Append(outputVertex(input[0], input[0].normal));
    triStream.Append(outputVertex(input[1], input[1].normal));
    triStream.Append(outputVertex(input[2], input[2].normal));
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
[maxvertexcount(3)]
void gs_triangle_wire( triangle OutputVertex input[3],
                       inout TriangleStream<OutputVertex> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    float2 edgeVerts[3];
    edgeVerts[0] = input[0].positionOut.xy / input[0].positionOut.w;
    edgeVerts[1] = input[1].positionOut.xy / input[1].positionOut.w;
    edgeVerts[2] = input[2].positionOut.xy / input[2].positionOut.w;

    triStream.Append(outputWireVertex(input[0], n0, 0, edgeVerts));
    triStream.Append(outputWireVertex(input[1], n0, 1, edgeVerts));
    triStream.Append(outputWireVertex(input[2], n0, 2, edgeVerts));
}

[maxvertexcount(3)]
void gs_triangle_smooth_wire( triangle OutputVertex input[3],
                              inout TriangleStream<OutputVertex> triStream )
{
    float2 edgeVerts[3];
    edgeVerts[0] = input[0].positionOut.xy / input[0].positionOut.w;
    edgeVerts[1] = input[1].positionOut.xy / input[1].positionOut.w;
    edgeVerts[2] = input[2].positionOut.xy / input[2].positionOut.w;

    triStream.Append(outputWireVertex(input[0], input[0].normal, 0, edgeVerts));
    triStream.Append(outputWireVertex(input[1], input[1].normal, 1, edgeVerts));
    triStream.Append(outputWireVertex(input[2], input[2].normal, 2, edgeVerts));
}
#endif
#endif

[maxvertexcount(1)]
void gs_point( point OutputVertex input[1],
               inout PointStream<OutputPointVertex> pointStream )
{
    OutputPointVertex v0;
    v0.positionOut = input[0].positionOut;

    pointStream.Append(v0);
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
lighting(float3 Peye, float3 Neye)
{
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    //float4 material = float4(0.4, 0.4, 0.8, 1);
    float4 material = float4(0.13, 0.13, 0.61, 1); // sRGB (gamma 2.2)

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        float4 Plight = lightSource[i].position;

        float3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        float3 n = normalize(Neye);
        float3 h = normalize(l + float3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 500.0f);

        color += lightSource[i].ambient * material
            + d * lightSource[i].diffuse * material
            + s * lightSource[i].specular;
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

void
ps_main( in OutputVertex input,
         bool isFrontFacing : SV_IsFrontFace,
         out float4 colorOut : SV_Target )
{
    float3 N = (isFrontFacing ? input.normal : -input.normal);
    colorOut = edgeColor(lighting(input.position.xyz, N), input.edgeDistance);
}

void
ps_main_point( in OutputPointVertex input,
               out float4 colorOut : SV_Target )
{
    colorOut = float4(1, 1, 1, 1);
}
