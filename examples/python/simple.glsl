#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
#

-- xforms

uniform mat4 Projection;
uniform mat4 Modelview;
uniform mat3 NormalMatrix;

-- vert

in vec4 Position;
in vec3 Normal;
out vec4 vPosition;
out vec3 vNormal;
void main()
{
    vPosition = Position;
    vNormal = Normal;
    gl_Position = Projection * Modelview * Position;
}

-- geom

in vec4 vPosition[4];
in vec3 vNormal[4];
out vec3 gNormal;
const bool facets = true;

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

void emit(int index)
{
    if (!facets) {
        gNormal = NormalMatrix * vNormal[index];
    }
    gl_Position = Projection * Modelview * vPosition[index];
    gl_Position.z = 1 - gl_Position.z;
    EmitVertex(); 
}

void main()
{
    if (facets) {
        vec3 A = vPosition[0].xyz;
        vec3 B = vPosition[1].xyz;
        vec3 C = vPosition[2].xyz;
        gNormal = NormalMatrix * normalize(cross(B - A, C - A));
    }

    emit(0);
    emit(1);
    emit(3);
    emit(2);
    EndPrimitive();
}

-- frag

in vec3 gNormal;
out vec4 FragColor;

uniform vec4 LightPosition = vec4(0.75, -0.25, 1, 1);
uniform vec3 AmbientMaterial = vec3(0.2, 0.1, 0.1);
uniform vec4 DiffuseMaterial = vec4(1.0, 209.0/255.0, 54.0/255.0, 1.0);
uniform vec3 SpecularMaterial = vec3(0.4, 0.4, 0.3);
uniform float Shininess = 200.0;
uniform float Fresnel = 0.1;

void main()
{
    vec3 N = normalize(gNormal);
    vec3 L = normalize((LightPosition).xyz);

    vec3 Eye = vec3(0, 0, 1);
    vec3 H = normalize(L + Eye);

    float df = max(0.0, dot(N, L));
    float sf = pow(max(0.0, dot(N, H)), Shininess);
    float rfTheta = Fresnel + (1-Fresnel) * pow(1-dot(N,Eye), 5);

    vec3 color = AmbientMaterial +
        df * DiffuseMaterial.rgb +
        sf * SpecularMaterial +
        rfTheta;

    FragColor = vec4(color, DiffuseMaterial.a);
}
