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

//#version 330

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;

out vec3 vPosition;
out vec3 vNormal;
out vec4 vColor;

void main()
{
    vPosition = position;
    vNormal = normal;
    vColor = vec4(1, 1, 1, 1);
}

#endif

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

in vec3 vPosition[4];
in vec3 vNormal[4];

#else // PRIM_TRI

layout(triangles) in;

#ifdef GEOMETRY_OUT_FILL
layout(triangle_strip, max_vertices = 3) out;
#endif

#ifdef GEOMETRY_OUT_LINE
layout(line_strip, max_vertices = 4) out;
#endif

in vec3 vPosition[3];
in vec3 vNormal[3];

#endif // PRIM_TRI/QUAD


uniform mat4 objectToClipMatrix;
uniform mat4 objectToEyeMatrix;

flat out vec3 gFacetNormal;
out vec3 Peye;
out vec3 Neye;
out vec4 Cout;

void emit(int index)
{
    Peye = vPosition[index];
    gl_Position = objectToClipMatrix * vec4(vPosition[index], 1);
    Neye = (objectToEyeMatrix * vec4(vNormal[index], 0)).xyz;
    EmitVertex();
}

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;
    
#ifdef PRIM_QUAD
#ifdef GEOMETRY_OUT_FILL
    vec3 A = vPosition[0] - vPosition[1];
    vec3 B = vPosition[3] - vPosition[1];
    vec3 C = vPosition[2] - vPosition[1];

    gFacetNormal = (objectToEyeMatrix*vec4(normalize(cross(B, A)), 0)).xyz;
    emit(0);
    emit(1);
    emit(3);
//    gFacetNormal = (objectToEyeMatrix*vec4(normalize(cross(C, B)), 0)).xyz;
    emit(2);
#else  // GEOMETRY_OUT_LINE
    emit(0);
    emit(1);
    emit(2);
    emit(3);
    emit(0);
#endif
#endif // PRIM_QUAD

#ifdef PRIM_TRI
    vec3 A = vPosition[1] - vPosition[0];
    vec3 B = vPosition[2] - vPosition[0];
    gFacetNormal = (objectToEyeMatrix*vec4(normalize(cross(B, A)), 0)).xyz;

    emit(0);
    emit(1);
    emit(2);
#ifdef GEOMETRY_OUT_LINE
    emit(0);
#endif //GEOMETRY_OUT_LINE
#endif // PRIM_TRI

    EndPrimitive();
}

#endif

//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

layout (location=0) out vec4 FragColor;
flat in vec3 gFacetNormal;
in vec3 Neye;
in vec3 Peye;
in vec4 Cout;

#define NUM_LIGHTS 2

struct LightSource {
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

uniform LightSource lightSource[NUM_LIGHTS];

vec4
lighting(vec3 Peye, vec3 Neye)
{
    vec4 color = vec4(0);
    vec4 material = vec4(0.4, 0.4, 0.8, 1);

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        vec4 Plight = lightSource[i].position;

        vec3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        vec3 n = normalize(Neye);
        vec3 h = normalize(l + vec3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 500.0f);

        color += lightSource[i].ambient * material
            + d * lightSource[i].diffuse * material
            + s * lightSource[i].specular;
    }

    color.a = 1;
    return color;
}

#ifdef GEOMETRY_OUT_LINE
uniform vec4 fragColor;
void
main()
{
    FragColor = fragColor;
}

#else

void
main()
{
    vec3 N = (gl_FrontFacing ? gFacetNormal : -gFacetNormal);
    FragColor = lighting(Peye, N);
}
#endif // GEOMETRY_OUT_LINE

#endif
