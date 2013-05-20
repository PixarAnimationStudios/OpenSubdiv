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

#extension GL_EXT_gpu_shader4 : require
//----------------------------------------------------------
// Patches.TessVertexGregory
//----------------------------------------------------------
#ifdef PATCH_VERTEX_GREGORY_SHADER

uniform samplerBuffer g_VertexBuffer;
uniform isamplerBuffer g_ValenceBuffer;

layout (location=0) in vec4 position;

out block {
    GregControlVertex v;
} output;

void main()
{
     int vID = gl_VertexID;

     output.v.hullPosition = (ModelViewMatrix * position).xyz;
     OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(position);

     uint valence = uint(texelFetchBuffer(g_ValenceBuffer,int(vID * (2 * OSD_MAX_VALENCE + 1))).x);
     output.v.valence = int(valence);

     vec3 f[OSD_MAX_VALENCE]; 
     vec3 pos = position.xyz;
     vec3 opos = vec3(0,0,0);

     for (uint i=0; i<valence; ++i) {
        uint im=(i+valence-1)%valence; 
        uint ip=(i+1)%valence; 

        uint idx_neighbor = uint(texelFetchBuffer(g_ValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*i + 0 + 1)).x);

        vec3 neighbor =
            vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+1)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+2)).x);

        uint idx_diagonal = uint(texelFetchBuffer(g_ValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*i + 1 + 1)).x);

        vec3 diagonal =
            vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+1)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+2)).x);

        uint idx_neighbor_p = uint(texelFetchBuffer(g_ValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*ip + 0 + 1)).x);

        vec3 neighbor_p =
            vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_p)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_p+1)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_p+2)).x);

        uint idx_neighbor_m = uint(texelFetchBuffer(g_ValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*im + 0 + 1)).x);

        vec3 neighbor_m =
            vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_m)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_m+1)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_m+2)).x);

        uint idx_diagonal_m = uint(texelFetchBuffer(g_ValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*im + 1 + 1)).x);

        vec3 diagonal_m =
            vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal_m)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal_m+1)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal_m+2)).x);

        f[i] = (pos * float(valence) + (neighbor_p + neighbor)*2.0 + diagonal) / (float(valence)+5.0);

        opos += f[i];
        output.v.r[i] = (neighbor_p-neighbor_m)/3.0 + (diagonal - diagonal_m)/6.0;
    }

    opos /= valence;
    output.v.position = vec4(opos, 1.0f).xyz;

#if OSD_NUM_VARYINGS > 0
    for (int i = 0; i < OSD_NUM_VARYINGS; ++i)
        output.v.varyings[i] = varyings[i];
#endif

    vec3 e;
    output.v.e0 = vec3(0,0,0);
    output.v.e1 = vec3(0,0,0);
    for(uint i=0; i<valence; ++i) {
        uint im = (i + valence -1) % valence;
        e = 0.5 * (f[i] + f[im]);
        output.v.e0 += csf(valence-3, 2*i) *e;
        output.v.e1 += csf(valence-3, 2*i + 1)*e;
    }
    output.v.e0 *= ef[valence - 3];
    output.v.e1 *= ef[valence - 3];
}
#endif

//----------------------------------------------------------
// Patches.TessControlGregory
//----------------------------------------------------------
#ifdef PATCH_TESS_CONTROL_GREGORY_SHADER

layout(vertices = 4) out;

uniform isamplerBuffer g_QuadOffsetBuffer;

in block {
    GregControlVertex v;
} input[];

out block {
    GregEvalVertex v;
} output[];

#define ID gl_InvocationID

void main()
{
    uint i = gl_InvocationID;
    uint ip = (i+1)%4;
    uint im = (i+3)%4;
    uint n = uint(input[i].v.valence);
    int base = GregoryQuadOffsetBase;

    output[ID].v.position = input[ID].v.position;

    uint start = texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + i)).x & 0x00ff;
    uint prev = uint(texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + i)).x) & 0xff00;
    prev=uint(prev/256);

    // Control Vertices based on : 
    // "Approximating Subdivision Surfaces with Gregory Patches for Hardware Tessellation" 
    // Loop, Schaefer, Ni, Castafio (ACM ToG Siggraph Asia 2009)
    //
    //  P3         e3-      e2+         E2
    //     O--------O--------O--------O
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |        O        O        |
    // e3+ O------O            O------O e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- O------O            O------O e1+
    //     |        O        O        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------O--------O--------O
    //  P0         e0+      e1-         E1
    //

    vec3 Ep = input[i].v.position + input[i].v.e0 * csf(n-3, 2*start) + input[i].v.e1*csf(n-3, 2*start +1);
    vec3 Em = input[i].v.position + input[i].v.e0 * csf(n-3, 2*prev ) + input[i].v.e1*csf(n-3, 2*prev + 1);

    uint np = input[ip].v.valence;
    uint nm = input[im].v.valence;

    uint prev_p = uint(texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + ip)).x)&0xff00;
    prev_p=uint(prev_p/256);
    vec3 Em_ip = input[ip].v.position + input[ip].v.e0*csf(np-3,2*prev_p) +input[ip].v.e1*csf(np-3, 2*prev_p+1);

    uint start_m = texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + im)).x&0x00ff;
    vec3 Ep_im = input[im].v.position + input[im].v.e0*csf(nm-3, 2*start_m) + input[im].v.e1*csf(nm-3, 2*start_m+1);

    float s1 = 3 - 2*csf(n-3,2)-csf(np-3,2);
    float s2 = 2*csf(n-3,2);

    vec3 Fp = (csf(np-3,2)*input[i].v.position + s1*Ep + s2*Em_ip + input[i].v.r[start])/3.0;

    s1 = 3.0 -2.0*cos(2.0*M_PI/float(n)) - cos(2*M_PI/float(nm));
    vec3 Fm = (csf(nm-3,2)*input[i].v.position + s1*Em +s2*Ep_im - input[i].v.r[prev])/3.0;

    output[ID].v.Ep = Ep;
    output[ID].v.Em = Em;
    output[ID].v.Fp = Fp;
    output[ID].v.Fm = Fm;

    int patchLevel = GetPatchLevel();
    output[ID].v.patchCoord = vec4(0, 0,
                                   patchLevel+0.5,
                                   gl_PrimitiveID+LevelBase+0.5);

    OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER;

    if (ID == 0) {
        OSD_PATCH_CULL(4);

#if OSD_ENABLE_SCREENSPACE_TESSELLATION
        gl_TessLevelOuter[0] =
            TessAdaptive(input[0].v.hullPosition.xyz, input[1].v.hullPosition.xyz, patchLevel);
        gl_TessLevelOuter[1] =
            TessAdaptive(input[0].v.hullPosition.xyz, input[3].v.hullPosition.xyz, patchLevel);
        gl_TessLevelOuter[2] =
            TessAdaptive(input[2].v.hullPosition.xyz, input[3].v.hullPosition.xyz, patchLevel);
        gl_TessLevelOuter[3] =
            TessAdaptive(input[1].v.hullPosition.xyz, input[2].v.hullPosition.xyz, patchLevel);
        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
#else
        gl_TessLevelInner[0] = GetTessLevel(patchLevel);
        gl_TessLevelInner[1] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[0] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[1] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[2] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[3] = GetTessLevel(patchLevel);
#endif
    }
}
#endif

//----------------------------------------------------------
// Patches.TessEvalGregory
//----------------------------------------------------------
#ifdef PATCH_TESS_EVAL_GREGORY_SHADER

layout(quads) in;
layout(cw) in;

in block {
    GregEvalVertex v;
} input[];

out block {
    OutputVertex v;
} output;

void main()
{
    float u = gl_TessCoord.x,
          v = gl_TessCoord.y;

    vec3 p[20];

    p[0] = input[0].v.position;
    p[1] = input[0].v.Ep;
    p[2] = input[0].v.Em;
    p[3] = input[0].v.Fp;
    p[4] = input[0].v.Fm;

    p[5] = input[1].v.position;
    p[6] = input[1].v.Ep;
    p[7] = input[1].v.Em;
    p[8] = input[1].v.Fp;
    p[9] = input[1].v.Fm;

    p[10] = input[2].v.position;
    p[11] = input[2].v.Ep;
    p[12] = input[2].v.Em;
    p[13] = input[2].v.Fp;
    p[14] = input[2].v.Fm;

    p[15] = input[3].v.position;
    p[16] = input[3].v.Ep;
    p[17] = input[3].v.Em;
    p[18] = input[3].v.Fp;
    p[19] = input[3].v.Fm;

    vec3 q[16];

    float U = 1-u, V=1-v;

    float d11 = u+v; if(u+v==0.0f) d11 = 1.0f;
    float d12 = U+v; if(U+v==0.0f) d12 = 1.0f;
    float d21 = u+V; if(u+V==0.0f) d21 = 1.0f;
    float d22 = U+V; if(U+V==0.0f) d22 = 1.0f;

    q[ 5] = (u*p[3] + v*p[4])/d11;
    q[ 6] = (U*p[9] + v*p[8])/d12;
    q[ 9] = (u*p[19] + V*p[18])/d21;
    q[10] = (U*p[13] + V*p[14])/d22;

    q[ 0] = p[0];
    q[ 1] = p[1];
    q[ 2] = p[7];
    q[ 3] = p[5];
    q[ 4] = p[2];
    q[ 7] = p[6];
    q[ 8] = p[16];
    q[11] = p[12];
    q[12] = p[15];
    q[13] = p[17];
    q[14] = p[11];
    q[15] = p[10];

    float B[4], D[4];

    Univar4x4(u, B, D);
    vec3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] =  vec3(0, 0, 0);
        DUCP[i] =  vec3(0, 0, 0);

        for (uint j=0; j<4; ++j) {
            // reverse face front
            vec3 A = q[i + 4*j];

            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    vec3 WorldPos  = vec3(0, 0, 0);
    vec3 Tangent   = vec3(0, 0, 0);
    vec3 BiTangent = vec3(0, 0, 0);

    Univar4x4(v, B, D);

    for (uint i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
    }

    BiTangent = (ModelViewMatrix * vec4(BiTangent, 0)).xyz;
    Tangent = (ModelViewMatrix * vec4(Tangent, 0)).xyz;

    vec3 normal = normalize(cross(BiTangent, Tangent));

    output.v.position = ModelViewMatrix * vec4(WorldPos, 1.0);
    output.v.normal = normal;
    output.v.tangent = BiTangent;

    output.v.patchCoord = input[0].v.patchCoord;
    output.v.patchCoord.xy = vec2(v, u);

    OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER;

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = ProjectionMatrix * output.v.position;
}

#endif

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec4 color;

out block {
    OutputVertex v;
} output;

void main() {
    gl_Position = ModelViewProjectionMatrix * position;
    output.v.color = color;
}

#endif

//----------------------------------------------------------
// Patches.FragmentColor
//----------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
} input;

void main() {
    gl_FragColor = input.v.color;
}
#endif
