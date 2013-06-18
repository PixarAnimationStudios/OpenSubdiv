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
    int uvOffset = (gl_PrimitiveID+LevelBase) * 4;

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
