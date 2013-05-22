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
// Patches.TessVertexBoundaryGregory
//----------------------------------------------------------
#ifdef PATCH_VERTEX_BOUNDARY_GREGORY_SHADER

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

     int valence = texelFetchBuffer(g_ValenceBuffer,int(vID * (2 * OSD_MAX_VALENCE + 1))).x;
     output.v.valence = int(valence);
     uint ivalence = uint(abs(valence));

     vec3 f[OSD_MAX_VALENCE]; 
     vec3 pos = position.xyz;
     output.v.org = position.xyz;
     vec3 opos = vec3(0,0,0);

     int boundaryEdgeNeighbors[2];
     uint currNeighbor = 0;
     uint ibefore = 0;
     uint zerothNeighbor = 0;

     for (uint i=0; i<ivalence; ++i) {
        uint im=(i+ivalence-1)%ivalence; 
        uint ip=(i+1)%ivalence; 

        bool isBoundaryNeighbor = false;
        uint idx_neighbor = uint(texelFetchBuffer(g_ValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*i + 0 + 1)).x);
        int valenceNeighbor = texelFetchBuffer(g_ValenceBuffer,int(idx_neighbor * (2 * OSD_MAX_VALENCE + 1))).x;

        if (valenceNeighbor < 0) {
            isBoundaryNeighbor = true;
            boundaryEdgeNeighbors[currNeighbor++] = int(idx_neighbor);
            if (currNeighbor == 1)    {
                ibefore = i;
                zerothNeighbor = i;
            } else {
                if (i-ibefore == 1) {
                    int tmp = boundaryEdgeNeighbors[0];
                    boundaryEdgeNeighbors[0] = boundaryEdgeNeighbors[1];
                    boundaryEdgeNeighbors[1] = tmp;
                    zerothNeighbor = i;
                } 
            }
        }

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

        f[i] = (pos * float(ivalence) + (neighbor_p + neighbor)*2.0f + diagonal) / (float(ivalence)+5.0f);

        opos += f[i];
        output.v.r[i] = (neighbor_p-neighbor_m)/3.0f + (diagonal - diagonal_m)/6.0f;
    }

    opos /= ivalence;
    output.v.position = vec4(opos, 1.0f).xyz;
    output.v.zerothNeighbor = zerothNeighbor;

#if OSD_NUM_VARYINGS > 0
    for (int i = 0; i < OSD_NUM_VARYINGS; ++i)
        output.v.varyings[i] = varyings[i];
#endif


    if (currNeighbor == 1) {
        boundaryEdgeNeighbors[1] = boundaryEdgeNeighbors[0];
    }

    vec3 e;
    output.v.e0 = vec3(0,0,0);
    output.v.e1 = vec3(0,0,0);

    for(uint i=0; i<ivalence; ++i) {
        uint im = (i + ivalence -1) % ivalence;
        e = 0.5f * (f[i] + f[im]);
        output.v.e0 += csf(ivalence-3, 2*i) *e;
        output.v.e1 += csf(ivalence-3, 2*i + 1)*e;
    }
    output.v.e0 *= ef[ivalence - 3];
    output.v.e1 *= ef[ivalence - 3];

    if (valence < 0) {
        if (ivalence > 2) {
            output.v.position = (
                vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0])).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+1)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+2)).x) +
                vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1])).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+1)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+2)).x) +
                4.0f * pos)/6.0f;        
        } else {
            output.v.position = pos;                    
        }

        output.v.e0 = ( 
            vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0])).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+1)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+2)).x) -
            vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1])).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+1)).x,
                 texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+2)).x) 
            )/6.0;

        float k = float(float(ivalence) - 1.0f);    //k is the number of faces
        float c = cos(M_PI/k);
        float s = sin(M_PI/k);
        float gamma = -(4.0f*s)/(3.0f*k+c);
        float alpha_0k = -((1.0f+2.0f*c)*sqrt(1.0f+c))/((3.0f*k+c)*sqrt(1.0f-c));
        float beta_0 = s/(3.0f*k + c); 


        int idx_diagonal = texelFetchBuffer(g_ValenceBuffer,int((vID) * (2*OSD_MAX_VALENCE+1) + 2*zerothNeighbor + 1 + 1)).x;
        idx_diagonal = abs(idx_diagonal);
        vec3 diagonal =
                vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+1)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+2)).x);

        output.v.e1 = gamma * pos + 
            alpha_0k * vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0])).x,
                            texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+1)).x,
                            texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+2)).x) +
            alpha_0k * vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1])).x,
                            texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+1)).x,
                            texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+2)).x) +
            beta_0 * diagonal;

        for (uint x=1; x<ivalence - 1; ++x) {
            uint curri = ((x + zerothNeighbor)%ivalence);
            float alpha = (4.0f*sin((M_PI * float(x))/k))/(3.0f*k+c);
            float beta = (sin((M_PI * float(x))/k) + sin((M_PI * float(x+1))/k))/(3.0f*k+c);

            int idx_neighbor = texelFetchBuffer(g_ValenceBuffer, int((vID) * (2*OSD_MAX_VALENCE+1) + 2*curri + 0 + 1)).x;
            idx_neighbor = abs(idx_neighbor);

            vec3 neighbor =
                vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+1)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+2)).x);

            idx_diagonal = texelFetchBuffer(g_ValenceBuffer, int((vID) * (2*OSD_MAX_VALENCE+1) + 2*curri + 1 + 1)).x;

            diagonal =
                vec3(texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+1)).x,
                     texelFetchBuffer(g_VertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+2)).x);

            output.v.e1 += alpha * neighbor + beta * diagonal;                         
        }

        output.v.e1 /= 3.0f;
    } 
}
#endif

//----------------------------------------------------------
// Patches.TessControlBoundaryGregory
//----------------------------------------------------------
#ifdef PATCH_TESS_CONTROL_BOUNDARY_GREGORY_SHADER

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
    uint n = uint(abs(input[i].v.valence));
    uint ivalence = abs(input[i].v.valence);
    int base = GregoryQuadOffsetBase;

    output[ID].v.position = input[ID].v.position;

    uint start = texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + i)).x & 0x00ff;
    uint prev = uint(texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + i)).x) & 0xff00;
    prev=uint(prev/256);
    uint np = abs(input[ip].v.valence);
    uint nm = abs(input[im].v.valence);

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

    vec3 Ep = vec3(0.0f,0.0f,0.0f);
    vec3 Em = vec3(0.0f,0.0f,0.0f);
    vec3 Fp = vec3(0.0f,0.0f,0.0f);
    vec3 Fm = vec3(0.0f,0.0f,0.0f);

    uint prev_p = uint(texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + ip)).x)&0xff00;
    prev_p=uint(prev_p/256);
    vec3 Em_ip;

    if (input[ip].v.valence < -2) {
        uint j = (np + prev_p - input[ip].v.zerothNeighbor) % np;
        Em_ip = input[ip].v.position + cos((M_PI*j)/float(np-1))*input[ip].v.e0 + sin((M_PI*j)/float(np-1))*input[ip].v.e1;
    } else {
        Em_ip = input[ip].v.position + input[ip].v.e0*csf(np-3,2*prev_p) + input[ip].v.e1*csf(np-3,2*prev_p+1);
    }

    uint start_m = uint(texelFetchBuffer(g_QuadOffsetBuffer, int(4*gl_PrimitiveID+base + im)).x)&0x00ff;
    vec3 Ep_im;

    if (input[im].v.valence < -2) {
        uint j = (nm + start_m - input[im].v.zerothNeighbor) % nm;
        Ep_im = input[im].v.position + cos((M_PI*j)/float(nm-1))*input[im].v.e0 + sin((M_PI*j)/float(nm-1))*input[im].v.e1;
    } else {
        Ep_im = input[im].v.position + input[im].v.e0*csf(nm-3,2*start_m) + input[im].v.e1*csf(nm-3,2*start_m+1);
    }

    if (input[i].v.valence < 0) {
        n = (n-1)*2;
    }
    if (input[im].v.valence < 0) {
        nm = (nm-1)*2;
    }  
    if (input[ip].v.valence < 0) {
        np = (np-1)*2;
    }

    if (input[i].v.valence > 2) {
        Ep = input[i].v.position + (input[i].v.e0*csf(n-3,2*start) + input[i].v.e1*csf(n-3,2*start+1));
        Em = input[i].v.position + (input[i].v.e0*csf(n-3,2*prev) +  input[i].v.e1*csf(n-3,2*prev+1)); 

        float s1=3-2*csf(n-3,2)-csf(np-3,2);
        float s2=2*csf(n-3,2);

        Fp = (csf(np-3,2)*input[i].v.position + s1*Ep + s2*Em_ip + input[i].v.r[start])/3.0f; 
        s1 = 3.0f-2.0f*cos(2.0f*M_PI/n)-cos(2*M_PI/nm);
        Fm = (csf(nm-3,2)*input[i].v.position + s1*Em + s2*Ep_im - input[i].v.r[prev])/3.0f;

    } else if (input[i].v.valence < -2) {
        uint j = (ivalence + start - input[i].v.zerothNeighbor) % ivalence;

        Ep = input[i].v.position + cos((M_PI*j)/float(ivalence-1))*input[i].v.e0 + sin((M_PI*j)/float(ivalence-1))*input[i].v.e1;
        j = (ivalence + prev - input[i].v.zerothNeighbor) % ivalence;
        Em = input[i].v.position + cos((M_PI*j)/float(ivalence-1))*input[i].v.e0 + sin((M_PI*j)/float(ivalence-1))*input[i].v.e1;

        vec3 Rp = ((-2.0f * input[i].v.org - 1.0f * input[im].v.org) + (2.0f * input[ip].v.org + 1.0f * input[(i+2)%4].v.org))/3.0f;
        vec3 Rm = ((-2.0f * input[i].v.org - 1.0f * input[ip].v.org) + (2.0f * input[im].v.org + 1.0f * input[(i+2)%4].v.org))/3.0f;

        float s1=3-2*csf(n-3,2)-csf(np-3,2);
        float s2=2*csf(n-3,2);

        Fp = (csf(np-3,2)*input[i].v.position + s1*Ep + s2*Em_ip + input[i].v.r[start])/3.0f; 
        s1 = 3.0f-2.0f*cos(2.0f*M_PI/n)-cos(2.0f*M_PI/nm);
        Fm = (csf(nm-3,2)*input[i].v.position + s1*Em + s2*Ep_im - input[i].v.r[prev])/3.0f;

        if (input[im].v.valence < 0) {
            s1=3-2*csf(n-3,2)-csf(np-3,2);
            Fp = Fm = (csf(np-3,2)*input[i].v.position + s1*Ep + s2*Em_ip + input[i].v.r[start])/3.0f;
        } else if (input[ip].v.valence < 0) {
            s1 = 3.0f-2.0f*cos(2.0f*M_PI/n)-cos(2.0f*M_PI/nm);
            Fm = Fp = (csf(nm-3,2)*input[i].v.position + s1*Em + s2*Ep_im - input[i].v.r[prev])/3.0f;
        }

    } else if (input[i].v.valence == -2) {
        Ep = (2.0f * input[i].v.org + input[ip].v.org)/3.0f;
        Em = (2.0f * input[i].v.org + input[im].v.org)/3.0f;
        Fp = Fm = (4.0f * input[i].v.org + input[(i+2)%n].v.org + 2.0f * input[ip].v.org + 2.0f * input[im].v.org)/9.0f;
    }

    output[ID].v.Ep = Ep;
    output[ID].v.Em = Em;
    output[ID].v.Fp = Fp;
    output[ID].v.Fm = Fm;

    int patchLevel = GetPatchLevel();
    output[ID].v.patchCoord = vec4(0, 0,
                                   patchLevel+0.5f,
                                   gl_PrimitiveID+LevelBase+0.5f);

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
// Patches.TessEvalBoundaryGregory
//----------------------------------------------------------
#ifdef PATCH_TESS_EVAL_BOUNDARY_GREGORY_SHADER

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

    Univar4x4(gl_TessCoord.x, B, D);
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

    Univar4x4(gl_TessCoord.y, B, D);

    for (uint i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
    }

    BiTangent = (ModelViewMatrix * vec4(BiTangent, 0)).xyz;
    Tangent = (ModelViewMatrix * vec4(Tangent, 0)).xyz;

    vec3 normal = normalize(cross(BiTangent, Tangent));

    output.v.position = ModelViewMatrix * vec4(WorldPos, 1.0f);
    output.v.normal = normal;
    output.v.patchCoord = input[0].v.patchCoord;
    output.v.patchCoord.xy = vec2(v, u);
    output.v.tangent = normalize(BiTangent);

    OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER;

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = (ModelViewProjectionMatrix * vec4(WorldPos, 1.0f));
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
