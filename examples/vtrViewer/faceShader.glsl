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

#if __VERSION__ < 420
    #define centroid
#endif

layout(std140) uniform Transform {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
};

struct ControlVertex {
    vec4 position;
    centroid vec4 patchCoord; // u, v, level, faceID
    ivec4 ptexInfo;           // U offset, V offset, 2^ptexlevel', rotation
    ivec3 clipFlag;
};

struct OutputVertex {
    vec4 position;
    vec3 normal;
    centroid vec4 patchCoord; // u, v, level, faceID
    centroid vec2 tessCoord;  // tesscoord.st
    vec3 tangent;
    vec3 bitangent;
};

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

layout(lines_adjacency) in;

#define EDGE_VERTS 4

layout(triangle_strip, max_vertices = EDGE_VERTS) out;

in block {
    OutputVertex v;
} inpt[EDGE_VERTS];

out block {
    OutputVertex v;
} outpt;

void emit(int index, vec3 normal, vec2 uv)
{
    outpt.v.position = inpt[index].v.position;
    outpt.v.normal = normal;
    outpt.v.tessCoord = uv;

    gl_Position = ProjectionMatrix * inpt[index].v.position;
    EmitVertex();
}

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

    vec3 A = (inpt[0].v.position - inpt[1].v.position).xyz;
    vec3 B = (inpt[3].v.position - inpt[1].v.position).xyz;
    vec3 C = (inpt[2].v.position - inpt[1].v.position).xyz;
    vec3 n0 = normalize(cross(B, A));

    emit(0, n0, vec2(0.0,0.0));
    emit(1, n0, vec2(0.0,1.0));
    emit(3, n0, vec2(1.0,0.0));
    emit(2, n0, vec2(1.0,1.0));

    EndPrimitive();
}

#endif

//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
} inpt;

out vec4 outColor;
out vec3 outNormal;

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

uniform samplerBuffer faceColors;
uniform sampler2D faceTexture;

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

        color +=       lightSource[i].ambient * ambientColor
                 + d * lightSource[i].diffuse * diffuse
                 + s * lightSource[i].specular;
    }

    color.a = 1;
    return color;
}

void
main()
{
    vec3 N = (gl_FrontFacing ? inpt.v.normal : -inpt.v.normal);

    vec4 faceColor = texelFetch(faceColors, gl_PrimitiveID);
    
    vec4 tex = texture(faceTexture, inpt.v.tessCoord);

    vec4 Cf = lighting(diffuseColor * faceColor * tex, inpt.v.position.xyz, N);

    outColor = Cf;
    outNormal = N;
}

#endif
