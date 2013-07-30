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

//----------------------------------------------------------
// Patches.Coefficients
//----------------------------------------------------------

#if OSD_MAX_VALENCE<=10
uniform float ef[7] = float[](
    0.813008, 0.500000, 0.363636, 0.287505,
    0.238692, 0.204549, 0.179211
);
#else
uniform float ef[27] = float[](
    0.812816, 0.500000, 0.363644, 0.287514,
    0.238688, 0.204544, 0.179229, 0.159657,
    0.144042, 0.131276, 0.120632, 0.111614,
    0.103872, 0.09715, 0.0912559, 0.0860444,
    0.0814022, 0.0772401, 0.0734867, 0.0700842,
    0.0669851, 0.0641504, 0.0615475, 0.0591488,
    0.0569311, 0.0548745, 0.0529621
);
#endif

float csf(uint n, uint j)
{
    if (j%2 == 0) {
        return cos((2.0f * M_PI * float(float(j-0)/2.0f))/(float(n)+3.0f));
    } else {
        return sin((2.0f * M_PI * float(float(j-1)/2.0f))/(float(n)+3.0f));
    }
}

//----------------------------------------------------------
// Patches.TessVertexGregory
//----------------------------------------------------------
#ifdef OSD_PATCH_VERTEX_GREGORY_SHADER

uniform samplerBuffer OsdVertexBuffer;
uniform isamplerBuffer OsdValenceBuffer;

layout (location=0) in vec4 position;
OSD_USER_VARYING_ATTRIBUTE_DECLARE

out block {
    GregControlVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    int vID = gl_VertexID;

    outpt.v.hullPosition = (ModelViewMatrix * position).xyz;
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(position);
    OSD_USER_VARYING_PER_VERTEX();

    int ivalence = texelFetch(OsdValenceBuffer,int(vID * (2 * OSD_MAX_VALENCE + 1))).x;
    outpt.v.valence = ivalence;
    uint valence = uint(abs(ivalence));

    vec3 f[OSD_MAX_VALENCE]; 
    vec3 pos = position.xyz;
    vec3 opos = vec3(0,0,0);

#ifdef OSD_PATCH_GREGORY_BOUNDARY
    outpt.v.org = position.xyz;
    int boundaryEdgeNeighbors[2];
    uint currNeighbor = 0;
    uint ibefore = 0;
    uint zerothNeighbor = 0;
#endif

    for (uint i=0; i<valence; ++i) {
        uint im=(i+valence-1)%valence; 
        uint ip=(i+1)%valence; 

        uint idx_neighbor = uint(texelFetch(OsdValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*i + 0 + 1)).x);

#ifdef OSD_PATCH_GREGORY_BOUNDARY
        bool isBoundaryNeighbor = false;
        int valenceNeighbor = texelFetch(OsdValenceBuffer,int(idx_neighbor * (2*OSD_MAX_VALENCE+1))).x;

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
#endif

        vec3 neighbor =
            vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+1)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+2)).x);

        uint idx_diagonal = uint(texelFetch(OsdValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*i + 1 + 1)).x);

        vec3 diagonal =
            vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+1)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+2)).x);

        uint idx_neighbor_p = uint(texelFetch(OsdValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*ip + 0 + 1)).x);

        vec3 neighbor_p =
            vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_p)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_p+1)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_p+2)).x);

        uint idx_neighbor_m = uint(texelFetch(OsdValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*im + 0 + 1)).x);

        vec3 neighbor_m =
            vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_m)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_m+1)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor_m+2)).x);

        uint idx_diagonal_m = uint(texelFetch(OsdValenceBuffer, int(vID * (2*OSD_MAX_VALENCE+1) + 2*im + 1 + 1)).x);

        vec3 diagonal_m =
            vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal_m)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal_m+1)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal_m+2)).x);

        f[i] = (pos * float(valence) + (neighbor_p + neighbor)*2.0f + diagonal) / (float(valence)+5.0f);

        opos += f[i];
        outpt.v.r[i] = (neighbor_p-neighbor_m)/3.0f + (diagonal - diagonal_m)/6.0f;
    }

    opos /= valence;
    outpt.v.position = vec4(opos, 1.0f).xyz;

    vec3 e;
    outpt.v.e0 = vec3(0,0,0);
    outpt.v.e1 = vec3(0,0,0);

    for(uint i=0; i<valence; ++i) {
        uint im = (i + valence -1) % valence;
        e = 0.5f * (f[i] + f[im]);
        outpt.v.e0 += csf(valence-3, 2*i) *e;
        outpt.v.e1 += csf(valence-3, 2*i + 1)*e;
    }
    outpt.v.e0 *= ef[valence - 3];
    outpt.v.e1 *= ef[valence - 3];

#ifdef OSD_PATCH_GREGORY_BOUNDARY
    outpt.v.zerothNeighbor = zerothNeighbor;
    if (currNeighbor == 1) {
        boundaryEdgeNeighbors[1] = boundaryEdgeNeighbors[0];
    }

    if (ivalence < 0) {
        if (valence > 2) {
            outpt.v.position = (
                vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0])).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+1)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+2)).x) +
                vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1])).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+1)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+2)).x) +
                4.0f * pos)/6.0f;        
        } else {
            outpt.v.position = pos;                    
        }

        outpt.v.e0 = ( 
            vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0])).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+1)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+2)).x) -
            vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1])).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+1)).x,
                 texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+2)).x) 
            )/6.0;

        float k = float(float(valence) - 1.0f);    //k is the number of faces
        float c = cos(M_PI/k);
        float s = sin(M_PI/k);
        float gamma = -(4.0f*s)/(3.0f*k+c);
        float alpha_0k = -((1.0f+2.0f*c)*sqrt(1.0f+c))/((3.0f*k+c)*sqrt(1.0f-c));
        float beta_0 = s/(3.0f*k + c); 


        int idx_diagonal = texelFetch(OsdValenceBuffer,int((vID) * (2*OSD_MAX_VALENCE+1) + 2*zerothNeighbor + 1 + 1)).x;
        idx_diagonal = abs(idx_diagonal);
        vec3 diagonal =
                vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+1)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+2)).x);

        outpt.v.e1 = gamma * pos + 
            alpha_0k * vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0])).x,
                            texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+1)).x,
                            texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[0]+2)).x) +
            alpha_0k * vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1])).x,
                            texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+1)).x,
                            texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*boundaryEdgeNeighbors[1]+2)).x) +
            beta_0 * diagonal;

        for (uint x=1; x<valence - 1; ++x) {
            uint curri = ((x + zerothNeighbor)%valence);
            float alpha = (4.0f*sin((M_PI * float(x))/k))/(3.0f*k+c);
            float beta = (sin((M_PI * float(x))/k) + sin((M_PI * float(x+1))/k))/(3.0f*k+c);

            int idx_neighbor = texelFetch(OsdValenceBuffer, int((vID) * (2*OSD_MAX_VALENCE+1) + 2*curri + 0 + 1)).x;
            idx_neighbor = abs(idx_neighbor);

            vec3 neighbor =
                vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+1)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_neighbor+2)).x);

            idx_diagonal = texelFetch(OsdValenceBuffer, int((vID) * (2*OSD_MAX_VALENCE+1) + 2*curri + 1 + 1)).x;

            diagonal =
                vec3(texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+1)).x,
                     texelFetch(OsdVertexBuffer, int(OSD_NUM_ELEMENTS*idx_diagonal+2)).x);

            outpt.v.e1 += alpha * neighbor + beta * diagonal;                         
        }

        outpt.v.e1 /= 3.0f;
    } 
#endif
}

#endif

//----------------------------------------------------------
// Patches.TessControlGregory
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_CONTROL_GREGORY_SHADER

layout(vertices = 4) out;

uniform isamplerBuffer OsdQuadOffsetBuffer;

in block {
    GregControlVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    GregEvalVertex v;
    OSD_USER_VARYING_DECLARE
} outpt[];

#define ID gl_InvocationID

void main()
{
    uint i = gl_InvocationID;
    uint ip = (i+1)%4;
    uint im = (i+3)%4;
    uint valence = abs(inpt[i].v.valence);
    uint n = valence;
    int base = OsdGregoryQuadOffsetBase;

    outpt[ID].v.position = inpt[ID].v.position;

    uint start = uint(texelFetch(OsdQuadOffsetBuffer, int(4*gl_PrimitiveID+base + i)).x) & 0x00ffu;
    uint prev = uint(texelFetch(OsdQuadOffsetBuffer, int(4*gl_PrimitiveID+base + i)).x) & 0xff00u;
    prev = uint(prev/256);

    uint start_m = uint(texelFetch(OsdQuadOffsetBuffer, int(4*gl_PrimitiveID+base + im)).x) & 0x00ffu;
    uint prev_p = uint(texelFetch(OsdQuadOffsetBuffer, int(4*gl_PrimitiveID+base + ip)).x) & 0xff00u;
    prev_p = uint(prev_p/256);

    uint np = abs(inpt[ip].v.valence);
    uint nm = abs(inpt[im].v.valence);

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

#ifdef OSD_PATCH_GREGORY_BOUNDARY
    vec3 Ep = vec3(0.0f,0.0f,0.0f);
    vec3 Em = vec3(0.0f,0.0f,0.0f);
    vec3 Fp = vec3(0.0f,0.0f,0.0f);
    vec3 Fm = vec3(0.0f,0.0f,0.0f);

    vec3 Em_ip;
    if (inpt[ip].v.valence < -2) {
        uint j = (np + prev_p - inpt[ip].v.zerothNeighbor) % np;
        Em_ip = inpt[ip].v.position + cos((M_PI*j)/float(np-1))*inpt[ip].v.e0 + sin((M_PI*j)/float(np-1))*inpt[ip].v.e1;
    } else {
        Em_ip = inpt[ip].v.position + inpt[ip].v.e0*csf(np-3, 2*prev_p ) + inpt[ip].v.e1*csf(np-3, 2*prev_p + 1);
    }

    vec3 Ep_im;
    if (inpt[im].v.valence < -2) {
        uint j = (nm + start_m - inpt[im].v.zerothNeighbor) % nm;
        Ep_im = inpt[im].v.position + cos((M_PI*j)/float(nm-1))*inpt[im].v.e0 + sin((M_PI*j)/float(nm-1))*inpt[im].v.e1;
    } else {
        Ep_im = inpt[im].v.position + inpt[im].v.e0*csf(nm-3, 2*start_m) + inpt[im].v.e1*csf(nm-3, 2*start_m + 1);
    }

    if (inpt[i].v.valence < 0) {
        n = (n-1)*2;
    }
    if (inpt[im].v.valence < 0) {
        nm = (nm-1)*2;
    }  
    if (inpt[ip].v.valence < 0) {
        np = (np-1)*2;
    }

    if (inpt[i].v.valence > 2) {
        Ep = inpt[i].v.position + inpt[i].v.e0*csf(n-3, 2*start) + inpt[i].v.e1*csf(n-3, 2*start + 1);
        Em = inpt[i].v.position + inpt[i].v.e0*csf(n-3, 2*prev ) + inpt[i].v.e1*csf(n-3, 2*prev + 1); 

        float s1=3-2*csf(n-3,2)-csf(np-3,2);
        float s2=2*csf(n-3,2);

        Fp = (csf(np-3,2)*inpt[i].v.position + s1*Ep + s2*Em_ip + inpt[i].v.r[start])/3.0f; 
        s1 = 3.0f-2.0f*cos(2.0f*M_PI/float(n))-cos(2.0f*M_PI/float(nm));
        Fm = (csf(nm-3,2)*inpt[i].v.position + s1*Em + s2*Ep_im - inpt[i].v.r[prev])/3.0f;

    } else if (inpt[i].v.valence < -2) {
        uint j = (valence + start - inpt[i].v.zerothNeighbor) % valence;

        Ep = inpt[i].v.position + cos((M_PI*j)/float(valence-1))*inpt[i].v.e0 + sin((M_PI*j)/float(valence-1))*inpt[i].v.e1;
        j = (valence + prev - inpt[i].v.zerothNeighbor) % valence;
        Em = inpt[i].v.position + cos((M_PI*j)/float(valence-1))*inpt[i].v.e0 + sin((M_PI*j)/float(valence-1))*inpt[i].v.e1;

        vec3 Rp = ((-2.0f * inpt[i].v.org - 1.0f * inpt[im].v.org) + (2.0f * inpt[ip].v.org + 1.0f * inpt[(i+2)%4].v.org))/3.0f;
        vec3 Rm = ((-2.0f * inpt[i].v.org - 1.0f * inpt[ip].v.org) + (2.0f * inpt[im].v.org + 1.0f * inpt[(i+2)%4].v.org))/3.0f;

        float s1 = 3-2*csf(n-3,2)-csf(np-3,2);
        float s2 = 2*csf(n-3,2);

        Fp = (csf(np-3,2)*inpt[i].v.position + s1*Ep + s2*Em_ip + inpt[i].v.r[start])/3.0f; 
        s1 = 3.0f-2.0f*cos(2.0f*M_PI/float(n))-cos(2.0f*M_PI/float(nm));
        Fm = (csf(nm-3,2)*inpt[i].v.position + s1*Em + s2*Ep_im - inpt[i].v.r[prev])/3.0f;

        if (inpt[im].v.valence < 0) {
            s1 = 3-2*csf(n-3,2)-csf(np-3,2);
            Fp = Fm = (csf(np-3,2)*inpt[i].v.position + s1*Ep + s2*Em_ip + inpt[i].v.r[start])/3.0f;
        } else if (inpt[ip].v.valence < 0) {
            s1 = 3.0f-2.0f*cos(2.0f*M_PI/n)-cos(2.0f*M_PI/nm);
            Fm = Fp = (csf(nm-3,2)*inpt[i].v.position + s1*Em + s2*Ep_im - inpt[i].v.r[prev])/3.0f;
        }

    } else if (inpt[i].v.valence == -2) {
        Ep = (2.0f * inpt[i].v.org + inpt[ip].v.org)/3.0f;
        Em = (2.0f * inpt[i].v.org + inpt[im].v.org)/3.0f;
        Fp = Fm = (4.0f * inpt[i].v.org + inpt[(i+2)%n].v.org + 2.0f * inpt[ip].v.org + 2.0f * inpt[im].v.org)/9.0f;
    }

#else // not OSD_PATCH_GREGORY_BOUNDARY

    vec3 Ep = inpt[i].v.position + inpt[i].v.e0 * csf(n-3, 2*start) + inpt[i].v.e1*csf(n-3, 2*start + 1);
    vec3 Em = inpt[i].v.position + inpt[i].v.e0 * csf(n-3, 2*prev ) + inpt[i].v.e1*csf(n-3, 2*prev + 1);

    vec3 Em_ip = inpt[ip].v.position + inpt[ip].v.e0 * csf(np-3, 2*prev_p ) + inpt[ip].v.e1*csf(np-3, 2*prev_p + 1);
    vec3 Ep_im = inpt[im].v.position + inpt[im].v.e0 * csf(nm-3, 2*start_m) + inpt[im].v.e1*csf(nm-3, 2*start_m + 1);

    float s1 = 3-2*csf(n-3,2)-csf(np-3,2);
    float s2 = 2*csf(n-3,2);

    vec3 Fp = (csf(np-3,2)*inpt[i].v.position + s1*Ep + s2*Em_ip + inpt[i].v.r[start])/3.0f;
    s1 = 3.0f-2.0f*cos(2.0f*M_PI/float(n))-cos(2.0f*M_PI/float(nm));
    vec3 Fm = (csf(nm-3,2)*inpt[i].v.position + s1*Em + s2*Ep_im - inpt[i].v.r[prev])/3.0f;

#endif

    outpt[ID].v.Ep = Ep;
    outpt[ID].v.Em = Em;
    outpt[ID].v.Fp = Fp;
    outpt[ID].v.Fm = Fm;

    OSD_USER_VARYING_PER_CONTROL_POINT(ID, ID);

    int patchLevel = GetPatchLevel();
    outpt[ID].v.patchCoord = vec4(0, 0,
                                  patchLevel+0.5f,
                                  gl_PrimitiveID+OsdPrimitiveIdBase+0.5f);

    OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER;

    if (ID == 0) {
        OSD_PATCH_CULL(4);

#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
        gl_TessLevelOuter[0] =
            TessAdaptive(inpt[0].v.hullPosition.xyz, inpt[1].v.hullPosition.xyz);
        gl_TessLevelOuter[1] =
            TessAdaptive(inpt[0].v.hullPosition.xyz, inpt[3].v.hullPosition.xyz);
        gl_TessLevelOuter[2] =
            TessAdaptive(inpt[2].v.hullPosition.xyz, inpt[3].v.hullPosition.xyz);
        gl_TessLevelOuter[3] =
            TessAdaptive(inpt[1].v.hullPosition.xyz, inpt[2].v.hullPosition.xyz);
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
#ifdef OSD_PATCH_TESS_EVAL_GREGORY_SHADER

layout(quads) in;
layout(cw) in;

#if defined OSD_FRACTIONAL_ODD_SPACING
    layout(fractional_odd_spacing) in;
#elif defined OSD_FRACTIONAL_EVEN_SPACING
    layout(fractional_even_spacing) in;
#endif

in block {
    GregEvalVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    OutputVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    float u = gl_TessCoord.x,
          v = gl_TessCoord.y;

    vec3 p[20];

    p[0] = inpt[0].v.position;
    p[1] = inpt[0].v.Ep;
    p[2] = inpt[0].v.Em;
    p[3] = inpt[0].v.Fp;
    p[4] = inpt[0].v.Fm;

    p[5] = inpt[1].v.position;
    p[6] = inpt[1].v.Ep;
    p[7] = inpt[1].v.Em;
    p[8] = inpt[1].v.Fp;
    p[9] = inpt[1].v.Fm;

    p[10] = inpt[2].v.position;
    p[11] = inpt[2].v.Ep;
    p[12] = inpt[2].v.Em;
    p[13] = inpt[2].v.Fp;
    p[14] = inpt[2].v.Fm;

    p[15] = inpt[3].v.position;
    p[16] = inpt[3].v.Ep;
    p[17] = inpt[3].v.Em;
    p[18] = inpt[3].v.Fp;
    p[19] = inpt[3].v.Fm;

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

    outpt.v.position = ModelViewMatrix * vec4(WorldPos, 1.0f);
    outpt.v.normal = normal;
    outpt.v.tangent = normalize(BiTangent);

    OSD_USER_VARYING_PER_EVAL_POINT(vec2(u,v), 0, 3, 1, 2);

    outpt.v.patchCoord = inpt[0].v.patchCoord;
    outpt.v.patchCoord.xy = vec2(v, u);

    OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER;

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = ModelViewProjectionMatrix * vec4(WorldPos, 1.0f);
}

#endif
