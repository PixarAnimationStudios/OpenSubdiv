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

uniform sampler2DArray textureImage_Data;
uniform samplerBuffer textureImage_Packing;
uniform isamplerBuffer textureImage_Pages;

vec4 displacement(vec4 position, vec3 normal, vec4 patchCoord)
{
    float disp = PTexLookup(patchCoord,
                            textureImage_Data,
                            textureImage_Packing,
                            textureImage_Pages).x;
    return position + 0.01*vec4(disp * normal, 0);
}

//--------------------------------------------------------------
// Uniform Blocks
//--------------------------------------------------------------

layout(std140) uniform Transform {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
    mat4 ModelViewInverseMatrix;
    mat4 ProjectionWithoutPickMatrix;
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

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;

out block {
    OutputVertex v;
} outpt;

void main()
{
    outpt.v.position = ModelViewMatrix * position;
}

#endif

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

layout(triangles) in;

layout(triangle_strip, max_vertices = 3) out;

in block {
   OutputVertex v;
} inpt[3];

out block {
    OutputVertex v;
    vec4 depthPosition;
} outpt;

void emit(int index, vec4 position)
{
    vec2 uv = vec2(inpt[index].v.patchCoord.xy);
    outpt.v.position    = ProjectionMatrix * position;
    outpt.depthPosition = ProjectionWithoutPickMatrix * position;
    outpt.v.patchCoord  = inpt[index].v.patchCoord;
    gl_Position          = vec4(uv*2-vec2(1.0), 0, 1);
    EmitVertex();
}

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

    vec4 patchCoord[3];
    vec4 position[3];

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

    emit(0, position[0]);
    emit(1, position[1]);
    emit(2, position[2]);
    EndPrimitive();
}

#endif

//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
    vec4 depthPosition;
} inpt;

layout(size1x32) uniform image2DArray outTextureImage;
uniform sampler2D paintTexture;
uniform sampler2D depthTexture;
uniform int imageSize = 256;

void
main()
{
    vec4 p = inpt.v.position;
    p.xyz /= p.w;

    vec4 wp = inpt.depthPosition;
    wp.z -= 0.001;
    wp.xyz /= wp.w;

    vec4 c = texture(paintTexture, p.xy*0.5+0.5);
    float depth = texture(depthTexture, wp.xy*0.5+0.5).x;
    if (wp.z*0.5+0.5 >= depth) return;
    
    ivec3 pos = ivec3(inpt.v.patchCoord.x * imageSize,
                      inpt.v.patchCoord.y * imageSize,
                      int(inpt.v.patchCoord.w));

    vec4 d = imageLoad(outTextureImage, pos);
    c = c + d;
    imageStore(outTextureImage, pos, c);
    discard;
}

#endif
