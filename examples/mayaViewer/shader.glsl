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

#endif // VERTEX_SHADER

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

uniform samplerBuffer g_uvFVarBuffer;

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

#ifdef PRIM_POINT

    layout(points) in;
    layout(points, max_vertices = 1) out;

    in block {
        OutputVertex v;
    } inpt[1];

#endif // PRIM_POINT

out block {
    OutputVertex v;
} outpt;

void emitUniform(int index, vec3 normal)
{
    outpt.v.position = inpt[index].v.position;
#ifdef SMOOTH_NORMALS
    outpt.v.normal = inpt[index].v.normal;
#else
    outpt.v.normal = normal;
#endif

    // We fetch each uv component separately since the texture buffer
    // has a single component internal format, i.e. R32F instead of RG32F.
    // Start with an offset representing 4 verts per primitive and
    // multiply by 2 on each fetch to account for two floats per UV.
    // uvFVarBuffer is a flat array of floats, but is accessed as if it
    // has the structure of float[p][4][2] where p=primitiveID:
    //      [ [ uv uv uv uv ] [ uv uv uv uv ] [ ... ] ]
    //            prim 0           prim 1
    int uvOffset = gl_PrimitiveID * 4;

    outpt.v.patchCoord.st = 
        vec2( texelFetch( g_uvFVarBuffer, (uvOffset+index)*2   ).s,
              texelFetch( g_uvFVarBuffer, (uvOffset+index)*2+1 ).s );

    gl_Position = ProjectionMatrix * inpt[index].v.position;

    EmitVertex();
}

void emitAdaptive(int index, vec3 normal, vec2 uvs[4])
{
    outpt.v.position = inpt[index].v.position;
#ifdef SMOOTH_NORMALS
    outpt.v.normal = inpt[index].v.normal;
#else
    outpt.v.normal = normal;
#endif
    
    // Bi-linear interpolation within the patch
    outpt.v.patchCoord = inpt[index].v.patchCoord;
    vec2 st = inpt[index].v.tessCoord;
    outpt.v.patchCoord.st =
        vec2( mix( mix(uvs[0].x, uvs[1].x, st.s ), mix(uvs[3].x, uvs[2].x, st.s ), st.t),
              mix( mix(uvs[0].y, uvs[1].y, st.s ), mix(uvs[3].y, uvs[2].y, st.s ), st.t)  );
              
    gl_Position = ProjectionMatrix * inpt[index].v.position;

    EmitVertex();
}

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

#ifdef FVAR_ADAPTIVE

    // We fetch each uv component separately since the texture buffer
    // has a single component internal format, i.e. R32F instead of RG32F.
    // Start with an offset representing 4 verts per primitive and
    // multiply by 2 on each fetch to account for two floats per UV.
    // uvFVarBuffer is a flat array of floats, but is accessed as if it
    // has the structure of float[p][4][2] where p=primitiveID:
    //      [ [ uv uv uv uv ] [ uv uv uv uv ] [ ... ] ]
    //            prim 0           prim 1

    // Offset based on prim id and offset into patch-type fvar data table
    int uvOffset = (gl_PrimitiveID+PrimitiveIdBase) * 4;

    vec2 uvs[4];
    uvs[0] = vec2( texelFetch( g_uvFVarBuffer, (uvOffset+0)*2   ).s,
                   texelFetch( g_uvFVarBuffer, (uvOffset+0)*2+1 ).s );
     
    uvs[1] = vec2( texelFetch( g_uvFVarBuffer, (uvOffset+1)*2   ).s,
                   texelFetch( g_uvFVarBuffer, (uvOffset+1)*2+1 ).s );
     
    uvs[2] = vec2( texelFetch( g_uvFVarBuffer, (uvOffset+2)*2   ).s,
                   texelFetch( g_uvFVarBuffer, (uvOffset+2)*2+1 ).s );
     
    uvs[3] = vec2( texelFetch( g_uvFVarBuffer, (uvOffset+3)*2   ).s,
                   texelFetch( g_uvFVarBuffer, (uvOffset+3)*2+1 ).s );
#endif

    vec3 n0 = vec3(0);

#if defined( PRIM_POINT )

    emitUniform(0, n0);

#elif defined ( PRIM_TRI)

    vec3 A = (inpt[1].v.position - inpt[0].v.position).xyz;
    vec3 B = (inpt[2].v.position - inpt[0].v.position).xyz;
    n0 = normalize(cross(B, A));
    #ifdef FVAR_ADAPTIVE
        emitAdaptive(0, n0, uvs);
        emitAdaptive(1, n0, uvs);
        emitAdaptive(2, n0, uvs);
    #ifdef GEOMETRY_OUT_LINE
        emitAdaptive(0, n0, uvs);
    #endif //GEOMETRY_OUT_LINE

#else

    emitUniform(0, n0);
    emitUniform(1, n0);
    emitUniform(2, n0);
    #ifdef GEOMETRY_OUT_LINE
        emitUniform(0, n0);
    #endif //GEOMETRY_OUT_LINE

#endif //FVAR_ADAPTIVE

#elif defined ( PRIM_QUAD )

    vec3 A = (inpt[0].v.position - inpt[1].v.position).xyz;
    vec3 B = (inpt[3].v.position - inpt[1].v.position).xyz;
    //vec3 C = (inpt[2].v.position - inpt[1].v.position).xyz;
    n0 = normalize(cross(B, A));

#ifdef GEOMETRY_OUT_FILL
    emitUniform(0, n0);
    emitUniform(1, n0);
    emitUniform(3, n0);
    emitUniform(2, n0);
#else  // GEOMETRY_OUT_LINE
    emitUniform(0, n0);
    emitUniform(1, n0);
    emitUniform(2, n0);
    emitUniform(3, n0);
    emitUniform(0, n0);
#endif //GEOMETRY_OUT_LINE

#endif //PRIM_*

    EndPrimitive();
}

#endif // GEOMETRY_SHADER

//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

uniform sampler2D diffuseMap;

in block {
    OutputVertex v;
} inpt;

#define NUM_LIGHTS 2

struct LightSource {
    vec4 position;
    vec4 diffuse;
    vec4 ambient;
    vec4 specular;
};

layout(std140) uniform Lighting {
    LightSource lightSource[NUM_LIGHTS];
};

uniform vec4 diffuseColor = vec4(0.8);
uniform vec4 ambientColor = vec4(0.2);
uniform vec4 specularColor = vec4(0.8);
uniform float shininess = 64;

vec4
lighting(vec3 Peye, vec3 Neye, vec4 texColor)
{
    vec4 color = vec4(0);

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        vec4 Plight = lightSource[i].position;

        vec3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        vec3 n = normalize(Neye);
        vec3 h = normalize(l + vec3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), shininess);

        color += lightSource[i].ambient * ambientColor
            + d * lightSource[i].diffuse * diffuseColor * texColor
            + s * lightSource[i].specular * specularColor;
    }

    color.a = 1;
    return color;
}

#ifdef PRIM_POINT
uniform vec4 fragColor;
void
main()
{
    gl_FragColor = fragColor;
}
#endif

#ifdef GEOMETRY_OUT_LINE
uniform vec4 fragColor;
void
main()
{
    gl_FragColor = fragColor;
}

#endif

#ifdef GEOMETRY_OUT_FILL
void
main()
{
    vec3 N = (gl_FrontFacing ? inpt.v.normal : -inpt.v.normal);
#ifdef USE_DIFFUSE_MAP
    vec4 texColor = texture(diffuseMap, inpt.v.patchCoord.st);
    gl_FragColor = lighting(inpt.v.position.xyz, N, texColor);
#else
    gl_FragColor = lighting(inpt.v.position.xyz, N, vec4(1.0));
#endif
}
#endif // GEOMETRY_OUT_LINE

#endif // FRAGMENT_SHADER
