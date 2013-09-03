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

#include "../osd/cpuKernel.h"
#include "../osd/tbbKernel.h"
#include "../osd/vertexDescriptor.h"

#include <math.h>
#include <tbb/parallel_for.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#define grain_size  200

class TBBFaceKernel {
    OsdVertexDescriptor const *vdesc;
    float        *vertex;
    float        *varying;
    int const    *F_IT;
    int const    *F_ITa;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vdesc->numVertexElements == 4 && varying == NULL) {
            ComputeFaceKernel<4>
                (vertex, F_IT, F_ITa, vertexOffset, tableOffset, r.begin(), r.end());
        } else if(vdesc->numVertexElements == 8 && varying == NULL) {
            ComputeFaceKernel<8>
                (vertex, F_IT, F_ITa, vertexOffset, tableOffset, r.begin(), r.end());
        }
        else {
            for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
                int h = F_ITa[2*i];
                int n = F_ITa[2*i+1];

                float weight = 1.0f/n;

                // XXX: should use local vertex struct variable instead of
                // accumulating directly into global memory.
                int dstIndex = i + vertexOffset - tableOffset;
                vdesc->Clear(vertex, varying, dstIndex);

                for (int j = 0; j < n; ++j) {
                    int index = F_IT[h+j];
                    vdesc->AddWithWeight(vertex, dstIndex, index, weight);
                    vdesc->AddVaryingWithWeight(varying, dstIndex, index, weight);
                }
           }
        }
    }

    TBBFaceKernel(TBBFaceKernel const &other)
    {
        this->vdesc  = other.vdesc;
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->F_IT   = other.F_IT;
        this->F_ITa  = other.F_ITa;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBFaceKernel(OsdVertexDescriptor const *vdesc_in,
                  float                     *vertex_in,
                  float                     *varying_in,
                  int const                 *F_IT_in,
                  int const                 *F_ITa_in,
                  int                        vertexOffset_in,
                  int                        tableOffset_in) :
                  vdesc  (vdesc_in),
                  vertex (vertex_in),
                  varying(varying_in),
                  F_IT   (F_IT_in),
                  F_ITa  (F_ITa_in),
                  vertexOffset(vertexOffset_in),
                  tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeFace(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    int const *F_IT, int const *F_ITa, int vertexOffset, int tableOffset,
    int start, int end) {

    TBBFaceKernel kernel(&vdesc, vertex, varying, F_IT, F_ITa,
                         vertexOffset, tableOffset);
    tbb::blocked_range<int> range(start, end, grain_size);
    tbb::parallel_for(range, kernel);
}

class TBBEdgeKernel {
    OsdVertexDescriptor const *vdesc;
    float        *vertex;
    float        *varying;
    int const    *E_IT;
    float const  *E_W;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vdesc->numVertexElements == 4 && varying == NULL) {
            ComputeEdgeKernel<4>(vertex, E_IT, E_W, vertexOffset, tableOffset,
                                 r.begin(), r.end());
        }
        else if(vdesc->numVertexElements == 8 && varying == NULL) {
            ComputeEdgeKernel<8>(vertex, E_IT, E_W, vertexOffset, tableOffset,
                                 r.begin(), r.end());
        }
        else {
            for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
                int eidx0 = E_IT[4*i+0];
                int eidx1 = E_IT[4*i+1];
                int eidx2 = E_IT[4*i+2];
                int eidx3 = E_IT[4*i+3];

                float vertWeight = E_W[i*2+0];

                int dstIndex = i + vertexOffset - tableOffset;
                vdesc->Clear(vertex, varying, dstIndex);

                vdesc->AddWithWeight(vertex, dstIndex, eidx0, vertWeight);
                vdesc->AddWithWeight(vertex, dstIndex, eidx1, vertWeight);

                if (eidx2 != -1) {
                    float faceWeight = E_W[i*2+1];

                    vdesc->AddWithWeight(vertex, dstIndex, eidx2, faceWeight);
                    vdesc->AddWithWeight(vertex, dstIndex, eidx3, faceWeight);
                }

                vdesc->AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
                vdesc->AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
            }
        }
    }

    TBBEdgeKernel(TBBEdgeKernel const &other)
    {
        this->vdesc  = other.vdesc;
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->E_IT   = other.E_IT;
        this->E_W    = other.E_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBEdgeKernel(OsdVertexDescriptor const *vdesc_in,
                  float                     *vertex_in,
                  float                     *varying_in,
                  int const                 *E_IT_in,
                  float const               *E_W_in,
                  int                        vertexOffset_in,
                  int                        tableOffset_in) :
                  vdesc  (vdesc_in),
                  vertex (vertex_in),
                  varying(varying_in),
                  E_IT   (E_IT_in),
                  E_W    (E_W_in),
                  vertexOffset(vertexOffset_in),
                  tableOffset(tableOffset_in)
    {};
};


void OsdTbbComputeEdge(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    int const *E_IT, float const *E_W, int vertexOffset, int tableOffset,
    int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBEdgeKernel kernel(&vdesc, vertex, varying, E_IT, E_W,
                         vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBVertexKernelA {
    OsdVertexDescriptor const *vdesc;
    float        *vertex;
    float        *varying;
    int const    *V_ITa;
    float const  *V_W;
    int           vertexOffset;
    int           tableOffset;
    int           pass;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vdesc->numVertexElements == 4 && varying == NULL) {
            ComputeVertexAKernel<4>(vertex, V_ITa, V_W, vertexOffset, tableOffset,
                                 r.begin(), r.end(), pass);
        }
        else if (vdesc->numVertexElements == 8 && varying == NULL) {
            ComputeVertexAKernel<8>(vertex, V_ITa, V_W, vertexOffset, tableOffset,
                                 r.begin(), r.end(), pass);
        }
        else {
            for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
                int n     = V_ITa[5*i+1];
                int p     = V_ITa[5*i+2];
                int eidx0 = V_ITa[5*i+3];
                int eidx1 = V_ITa[5*i+4];

                float weight = (pass == 1) ? V_W[i] : 1.0f - V_W[i];

                // In the case of fractional weight, the weight must be inverted since
                // the value is shared with the k_Smooth kernel (statistically the
                // k_Smooth kernel runs much more often than this one)
                if (weight > 0.0f && weight < 1.0f && n > 0)
                    weight = 1.0f - weight;

                int dstIndex = i + vertexOffset - tableOffset;

                if (not pass)
                    vdesc->Clear(vertex, varying, dstIndex);

                if (eidx0 == -1 || (pass == 0 && (n == -1))) {
                    vdesc->AddWithWeight(vertex, dstIndex, p, weight);
                } else {
                    vdesc->AddWithWeight(vertex, dstIndex, p, weight * 0.75f);
                    vdesc->AddWithWeight(vertex, dstIndex, eidx0, weight * 0.125f);
                    vdesc->AddWithWeight(vertex, dstIndex, eidx1, weight * 0.125f);
                }

                if (not pass)
                    vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
            }
        }
    }

    TBBVertexKernelA(TBBVertexKernelA const &other)
    {
        this->vdesc  = other.vdesc;
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->V_ITa  = other.V_ITa;
        this->V_W    = other.V_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
        this->pass         = other.pass;
    }

    TBBVertexKernelA(OsdVertexDescriptor const *vdesc_in,
                     float                     *vertex_in,
                     float                     *varying_in,
                     int const                 *V_ITa_in,
                     float const               *V_W_in,
                     int                        vertexOffset_in,
                     int                        tableOffset_in,
                     int                        pass_in) :
                     vdesc  (vdesc_in),
                     vertex (vertex_in),
                     varying(varying_in),
                     V_ITa  (V_ITa_in),
                     V_W    (V_W_in),
                     vertexOffset(vertexOffset_in),
                     tableOffset(tableOffset_in),
                     pass(pass_in)
    {};
};

void OsdTbbComputeVertexA(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    int const *V_ITa, float const *V_W, int vertexOffset, int tableOffset,
    int start, int end, int pass) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBVertexKernelA kernel(&vdesc, vertex, varying, V_ITa, V_W,
                            vertexOffset, tableOffset, pass);
    tbb::parallel_for(range, kernel);
}

class TBBVertexKernelB {
    OsdVertexDescriptor const *vdesc;
    float        *vertex;
    float        *varying;
    int const    *V_ITa;
    int const    *V_IT;
    float const  *V_W;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vdesc->numVertexElements == 4 && varying == NULL) {
            ComputeVertexBKernel<4>(vertex, V_ITa, V_IT, V_W,
                vertexOffset, tableOffset, r.begin(), r.end());
        }
        else if(vdesc->numVertexElements == 8 && varying == NULL) {
            ComputeVertexBKernel<8>(vertex, V_ITa, V_IT, V_W,
                vertexOffset, tableOffset, r.begin(), r.end());
        }
        else {
            for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
                int h = V_ITa[5*i];
                int n = V_ITa[5*i+1];
                int p = V_ITa[5*i+2];

                float weight = V_W[i];
                float wp = 1.0f/static_cast<float>(n*n);
                float wv = (n-2.0f) * n * wp;

                int dstIndex = i + vertexOffset - tableOffset;
                vdesc->Clear(vertex, varying, dstIndex);

                vdesc->AddWithWeight(vertex, dstIndex, p, weight * wv);

                for (int j = 0; j < n; ++j) {
                    vdesc->AddWithWeight(vertex, dstIndex, V_IT[h+j*2], weight * wp);
                    vdesc->AddWithWeight(vertex, dstIndex, V_IT[h+j*2+1], weight * wp);
                }
                vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
            }
        }
    }

    TBBVertexKernelB(TBBVertexKernelB const &other)
    {
        this->vdesc  = other.vdesc;
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->V_ITa  = other.V_ITa;
        this->V_IT   = other.V_IT;
        this->V_W    = other.V_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBVertexKernelB(OsdVertexDescriptor const *vdesc_in,
                     float                     *vertex_in,
                     float                     *varying_in,
                     int const                 *V_ITa_in,
                     int const                 *V_IT_in,
                     float const               *V_W_in,
                     int                        vertexOffset_in,
                     int                        tableOffset_in) :
                     vdesc  (vdesc_in),
                     vertex (vertex_in),
                     varying(varying_in),
                     V_ITa  (V_ITa_in),
                     V_IT   (V_IT_in),
                     V_W    (V_W_in),
                     vertexOffset(vertexOffset_in),
                     tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeVertexB(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    int const *V_ITa, int const *V_IT, float const *V_W,
    int vertexOffset, int tableOffset, int start, int end) {

    tbb::blocked_range<int> range(start, end, grain_size);
    TBBVertexKernelB kernel(&vdesc, vertex, varying, V_ITa, V_IT, V_W,
                            vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBLoopVertexKernelB {
    OsdVertexDescriptor const *vdesc;
    float        *vertex;
    float        *varying;
    int const    *V_ITa;
    int const    *V_IT;
    float const  *V_W;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vdesc->numVertexElements == 4 && varying == NULL) {
            ComputeLoopVertexBKernel<4>(vertex, V_ITa, V_IT, V_W, vertexOffset,
                                  tableOffset, r.begin(), r.end());
        }
        else if(vdesc->numVertexElements == 8 && varying == NULL) {
            ComputeLoopVertexBKernel<8>(vertex, V_ITa, V_IT, V_W, vertexOffset,
                                  tableOffset, r.begin(), r.end());
        }
        else {
            for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
                int h = V_ITa[5*i];
                int n = V_ITa[5*i+1];
                int p = V_ITa[5*i+2];

                float weight = V_W[i];
                float wp = 1.0f/static_cast<float>(n);
                float beta = 0.25f * cosf(static_cast<float>(M_PI) * 2.0f * wp) + 0.375f;
                beta = beta * beta;
                beta = (0.625f - beta) * wp;

                int dstIndex = i + vertexOffset - tableOffset;
                vdesc->Clear(vertex, varying, dstIndex);

                vdesc->AddWithWeight(vertex, dstIndex, p, weight * (1.0f - (beta * n)));

                for (int j = 0; j < n; ++j)
                    vdesc->AddWithWeight(vertex, dstIndex, V_IT[h+j], weight * beta);

                vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
            }
        }
    }

    TBBLoopVertexKernelB(TBBLoopVertexKernelB const &other)
    {
        this->vdesc  = other.vdesc;
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->V_ITa  = other.V_ITa;
        this->V_IT   = other.V_IT;
        this->V_W    = other.V_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBLoopVertexKernelB(OsdVertexDescriptor const *vdesc_in,
                         float                     *vertex_in,
                         float                     *varying_in,
                         int const                 *V_ITa_in,
                         int const                 *V_IT_in,
                         float const               *V_W_in,
                         int                        vertexOffset_in,
                         int                        tableOffset_in) :
                         vdesc  (vdesc_in),
                         vertex (vertex_in),
                         varying(varying_in),
                         V_ITa  (V_ITa_in),
                         V_IT   (V_IT_in),
                         V_W    (V_W_in),
                         vertexOffset(vertexOffset_in),
                         tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeLoopVertexB(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    int const *V_ITa, int const *V_IT, float const *V_W,
    int vertexOffset, int tableOffset, int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBLoopVertexKernelB kernel(&vdesc, vertex, varying, V_ITa, V_IT, V_W,
                                vertexOffset, tableOffset);

    tbb::parallel_for(range, kernel);
}

class TBBBilinearEdgeKernel {
    OsdVertexDescriptor const *vdesc;
    float        *vertex;
    float        *varying;
    int const    *E_IT;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vdesc->numVertexElements == 4 && varying == NULL) {
            ComputeBilinearEdgeKernel<4>(vertex, E_IT, vertexOffset, tableOffset,
                                         r.begin(), r.end());
        }
        else if(vdesc->numVertexElements == 8 && varying == NULL) {
            ComputeBilinearEdgeKernel<8>(vertex, E_IT, vertexOffset, tableOffset,
                                         r.begin(), r.end());
        }
        else {
            for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
                int eidx0 = E_IT[2*i+0];
                int eidx1 = E_IT[2*i+1];

                int dstIndex = i + vertexOffset - tableOffset;
                vdesc->Clear(vertex, varying, dstIndex);

                vdesc->AddWithWeight(vertex, dstIndex, eidx0, 0.5f);
                vdesc->AddWithWeight(vertex, dstIndex, eidx1, 0.5f);

                vdesc->AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
                vdesc->AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
            }
        }
    }

    TBBBilinearEdgeKernel(TBBBilinearEdgeKernel const &other)
    {
        this->vdesc  = other.vdesc;
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->E_IT   = other.E_IT;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBBilinearEdgeKernel(OsdVertexDescriptor const *vdesc_in,
                          float                     *vertex_in,
                          float                     *varying_in,
                          int const                 *E_IT_in,
                          int                        vertexOffset_in,
                          int                        tableOffset_in) :
                          vdesc  (vdesc_in),
                          vertex (vertex_in),
                          varying(varying_in),
                          E_IT   (E_IT_in),
                          vertexOffset(vertexOffset_in),
                          tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeBilinearEdge(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    int const *E_IT, int vertexOffset, int tableOffset, int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBBilinearEdgeKernel kernel(&vdesc, vertex, varying, E_IT, vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBBilinearVertexKernel {
    OsdVertexDescriptor const *vdesc;
    float        *vertex;
    float        *varying;
    int const    *V_ITa;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        int numVertexElements  = vdesc->numVertexElements;
        int numVaryingElements = vdesc->numVaryingElements;
        float *src, *des;
        for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
            int p = V_ITa[i];

            int dstIndex = i + vertexOffset - tableOffset;
            src = vertex + p        * numVertexElements;
            des = vertex + dstIndex * numVertexElements;
            memcpy(des, src, sizeof(float)*numVertexElements);
            if(varying) {
                src = varying + p        * numVaryingElements;
                des = varying + dstIndex * numVaryingElements;
                memcpy(des, src, sizeof(float)*numVaryingElements);
            }
        }
    }

    TBBBilinearVertexKernel(TBBBilinearVertexKernel const &other)
    {
        this->vdesc  = other.vdesc;
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->V_ITa  = other.V_ITa;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBBilinearVertexKernel(OsdVertexDescriptor const *vdesc_in,
                            float                     *vertex_in,
                            float                     *varying_in,
                            int const                 *V_ITa_in,
                            int                        vertexOffset_in,
                            int                        tableOffset_in) :
                            vdesc  (vdesc_in),
                            vertex (vertex_in),
                            varying(varying_in),
                            V_ITa  (V_ITa_in),
                            vertexOffset(vertexOffset_in),
                            tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeBilinearVertex(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    int const *V_ITa, int vertexOffset, int tableOffset, int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBBilinearVertexKernel kernel(&vdesc, vertex, varying, V_ITa, vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

void OsdTbbEditVertexAdd(
    OsdVertexDescriptor const &vdesc, float *vertex,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    unsigned int const *editIndices, float const *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {
        vdesc.ApplyVertexEditAdd(vertex,
                                 primVarOffset,
                                 primVarWidth,
                                 editIndices[i] + vertexOffset,
                                 &editValues[i*primVarWidth]);
    }
}

void OsdTbbEditVertexSet(
    OsdVertexDescriptor const &vdesc, float *vertex,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    unsigned int const *editIndices, float const *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {
        vdesc.ApplyVertexEditSet(vertex,
                                 primVarOffset,
                                 primVarWidth,
                                 editIndices[i] + vertexOffset,
                                 &editValues[i*primVarWidth]);
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
