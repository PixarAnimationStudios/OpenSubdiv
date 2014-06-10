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

#include "../osd/cpuKernel.h"
#include "../osd/tbbKernel.h"
#include "../osd/vertexDescriptor.h"

#include <math.h>
#include <tbb/parallel_for.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#define grain_size  200

static inline void
clear(float *origin, int index, OsdVertexBufferDescriptor const &desc) {

    if (origin) {
        float *dst = origin + index * desc.stride;
        memset(dst, 0, desc.length * sizeof(float));
    }
}

static inline void
addWithWeight(float *origin, int dstIndex, int srcIndex,
              float weight, OsdVertexBufferDescriptor const &desc) {

    if (origin) {
        const float *src = origin + srcIndex * desc.stride;
        float *dst = origin + dstIndex * desc.stride;
        for (int k = 0; k < desc.length; ++k) {
            dst[k] += src[k] * weight;
        }
    }
}

class TBBFaceKernel {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *F_IT;
    int const    *F_ITa;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vertexDesc.length == 4 && varying == NULL) {
            ComputeFaceKernel<4>
                (vertex, F_IT, F_ITa, vertexOffset, tableOffset, r.begin(), r.end());
        } else if(vertexDesc.length == 8 && varying == NULL) {
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

                clear(vertex, dstIndex, vertexDesc);
                clear(varying, dstIndex, varyingDesc);

                for (int j = 0; j < n; ++j) {
                    int index = F_IT[h+j];
                    addWithWeight(vertex, dstIndex, index, weight, vertexDesc);
                    addWithWeight(varying, dstIndex, index, weight, varyingDesc);
                }
           }
        }
    }

    TBBFaceKernel(TBBFaceKernel const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->F_IT   = other.F_IT;
        this->F_ITa  = other.F_ITa;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBFaceKernel(float                     *vertex_in,
                  float                     *varying_in,
                  OsdVertexBufferDescriptor  const &vertexDesc_in,
                  OsdVertexBufferDescriptor  const &varyingDesc_in,
                  int const                 *F_IT_in,
                  int const                 *F_ITa_in,
                  int                        vertexOffset_in,
                  int                        tableOffset_in) :
                  vertex (vertex_in),
                  varying(varying_in),
                  vertexDesc(vertexDesc_in),
                  varyingDesc(varyingDesc_in),
                  F_IT   (F_IT_in),
                  F_ITa  (F_ITa_in),
                  vertexOffset(vertexOffset_in),
                  tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *F_IT, int const *F_ITa, int vertexOffset, int tableOffset,
    int start, int end) {

    TBBFaceKernel kernel(vertex, varying, vertexDesc, varyingDesc, F_IT, F_ITa,
                         vertexOffset, tableOffset);
    tbb::blocked_range<int> range(start, end, grain_size);
    tbb::parallel_for(range, kernel);
}

class TBBQuadFaceKernel {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *F_IT;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        for (int i = r.begin(); i < r.end(); i++) {
            int fidx0 = F_IT[tableOffset + 4 * i + 0];
            int fidx1 = F_IT[tableOffset + 4 * i + 1];
            int fidx2 = F_IT[tableOffset + 4 * i + 2];
            int fidx3 = F_IT[tableOffset + 4 * i + 3];

            // XXX: should use local vertex struct variable instead of
            // accumulating directly into global memory.
            int dstIndex = i + vertexOffset;

            clear(vertex, dstIndex, vertexDesc);
            clear(varying, dstIndex, varyingDesc);

            addWithWeight(vertex, dstIndex, fidx0, 0.25f, vertexDesc);
            addWithWeight(vertex, dstIndex, fidx1, 0.25f, vertexDesc);
            addWithWeight(vertex, dstIndex, fidx2, 0.25f, vertexDesc);
            addWithWeight(vertex, dstIndex, fidx3, 0.25f, vertexDesc);
            addWithWeight(varying, dstIndex, fidx0, 0.25f, varyingDesc);
            addWithWeight(varying, dstIndex, fidx1, 0.25f, varyingDesc);
            addWithWeight(varying, dstIndex, fidx2, 0.25f, varyingDesc);
            addWithWeight(varying, dstIndex, fidx3, 0.25f, varyingDesc);
        }
    }

    TBBQuadFaceKernel(TBBQuadFaceKernel const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->F_IT   = other.F_IT;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBQuadFaceKernel(float                     *vertex_in,
                      float                     *varying_in,
                      OsdVertexBufferDescriptor  const &vertexDesc_in,
                      OsdVertexBufferDescriptor  const &varyingDesc_in,
                      int const                 *F_IT_in,
                      int                        vertexOffset_in,
                      int                        tableOffset_in) :
                      vertex (vertex_in),
                      varying(varying_in),
                      vertexDesc(vertexDesc_in),
                      varyingDesc(varyingDesc_in),
                      F_IT   (F_IT_in),
                      vertexOffset(vertexOffset_in),
                      tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeQuadFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *F_IT, int vertexOffset, int tableOffset,
    int start, int end) {

    TBBQuadFaceKernel kernel(vertex, varying, vertexDesc, varyingDesc, F_IT,
                             vertexOffset, tableOffset);
    tbb::blocked_range<int> range(start, end, grain_size);
    tbb::parallel_for(range, kernel);
}

class TBBTriQuadFaceKernel {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *F_IT;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        for (int i = r.begin(); i < r.end(); i++) {
            int fidx0 = F_IT[tableOffset + 4 * i + 0];
            int fidx1 = F_IT[tableOffset + 4 * i + 1];
            int fidx2 = F_IT[tableOffset + 4 * i + 2];
            int fidx3 = F_IT[tableOffset + 4 * i + 3];
            bool triangle = (fidx2 == fidx3);
            float weight = (triangle ? 1.0f / 3.0f : 1.0f / 4.0f);

            // XXX: should use local vertex struct variable instead of
            // accumulating directly into global memory.
            int dstIndex = i + vertexOffset;

            clear(vertex, dstIndex, vertexDesc);
            clear(varying, dstIndex, varyingDesc);

            addWithWeight(vertex, dstIndex, fidx0, weight, vertexDesc);
            addWithWeight(vertex, dstIndex, fidx1, weight, vertexDesc);
            addWithWeight(vertex, dstIndex, fidx2, weight, vertexDesc);
            addWithWeight(varying, dstIndex, fidx0, weight, varyingDesc);
            addWithWeight(varying, dstIndex, fidx1, weight, varyingDesc);
            addWithWeight(varying, dstIndex, fidx2, weight, varyingDesc);
            if (!triangle) {
                addWithWeight(vertex, dstIndex, fidx3, weight, vertexDesc);
                addWithWeight(varying, dstIndex, fidx3, weight, varyingDesc);
            }
        }
    }

    TBBTriQuadFaceKernel(TBBTriQuadFaceKernel const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->F_IT   = other.F_IT;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBTriQuadFaceKernel(float                     *vertex_in,
                         float                     *varying_in,
                         OsdVertexBufferDescriptor  const &vertexDesc_in,
                         OsdVertexBufferDescriptor  const &varyingDesc_in,
                         int const                 *F_IT_in,
                         int                        vertexOffset_in,
                         int                        tableOffset_in) :
                         vertex (vertex_in),
                         varying(varying_in),
                         vertexDesc(vertexDesc_in),
                         varyingDesc(varyingDesc_in),
                         F_IT   (F_IT_in),
                         vertexOffset(vertexOffset_in),
                         tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeTriQuadFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *F_IT, int vertexOffset, int tableOffset,
    int start, int end) {

    TBBTriQuadFaceKernel kernel(vertex, varying, vertexDesc, varyingDesc, F_IT,
                                vertexOffset, tableOffset);
    tbb::blocked_range<int> range(start, end, grain_size);
    tbb::parallel_for(range, kernel);
}

class TBBEdgeKernel {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *E_IT;
    float const  *E_W;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vertexDesc.length == 4 && varying == NULL) {
            ComputeEdgeKernel<4>(vertex, E_IT, E_W, vertexOffset, tableOffset,
                                 r.begin(), r.end());
        }
        else if(vertexDesc.length == 8 && varying == NULL) {
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
                clear(vertex, dstIndex, vertexDesc);
                clear(varying, dstIndex, varyingDesc);

                addWithWeight(vertex, dstIndex, eidx0, vertWeight, vertexDesc);
                addWithWeight(vertex, dstIndex, eidx1, vertWeight, vertexDesc);

                if (eidx2 != -1) {
                    float faceWeight = E_W[i*2+1];

                    addWithWeight(vertex, dstIndex, eidx2, faceWeight, vertexDesc);
                    addWithWeight(vertex, dstIndex, eidx3, faceWeight, vertexDesc);
                }

                addWithWeight(varying, dstIndex, eidx0, 0.5f, varyingDesc);
                addWithWeight(varying, dstIndex, eidx1, 0.5f, varyingDesc);
            }
        }
    }

    TBBEdgeKernel(TBBEdgeKernel const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->E_IT   = other.E_IT;
        this->E_W    = other.E_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBEdgeKernel(float                     *vertex_in,
                  float                     *varying_in,
                  OsdVertexBufferDescriptor const &vertexDesc_in,
                  OsdVertexBufferDescriptor const &varyingDesc_in,
                  int const                 *E_IT_in,
                  float const               *E_W_in,
                  int                        vertexOffset_in,
                  int                        tableOffset_in) :
                  vertex (vertex_in),
                  varying(varying_in),
                  vertexDesc(vertexDesc_in),
                  varyingDesc(varyingDesc_in),
                  E_IT   (E_IT_in),
                  E_W    (E_W_in),
                  vertexOffset(vertexOffset_in),
                  tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeEdge(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *E_IT, float const *E_W, int vertexOffset, int tableOffset,
    int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBEdgeKernel kernel(vertex, varying, vertexDesc, varyingDesc, E_IT, E_W,
                         vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBRestrictedEdgeKernel {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *E_IT;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
            int eidx0 = E_IT[4*i+0];
            int eidx1 = E_IT[4*i+1];
            int eidx2 = E_IT[4*i+2];
            int eidx3 = E_IT[4*i+3];

            int dstIndex = i + vertexOffset - tableOffset;
            clear(vertex, dstIndex, vertexDesc);
            clear(varying, dstIndex, varyingDesc);

            addWithWeight(vertex, dstIndex, eidx0, 0.25f, vertexDesc);
            addWithWeight(vertex, dstIndex, eidx1, 0.25f, vertexDesc);
            addWithWeight(vertex, dstIndex, eidx2, 0.25f, vertexDesc);
            addWithWeight(vertex, dstIndex, eidx3, 0.25f, vertexDesc);
            addWithWeight(varying, dstIndex, eidx0, 0.5f, varyingDesc);
            addWithWeight(varying, dstIndex, eidx1, 0.5f, varyingDesc);
        }
    }

    TBBRestrictedEdgeKernel(TBBRestrictedEdgeKernel const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->E_IT   = other.E_IT;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBRestrictedEdgeKernel(float                     *vertex_in,
                            float                     *varying_in,
                            OsdVertexBufferDescriptor const &vertexDesc_in,
                            OsdVertexBufferDescriptor const &varyingDesc_in,
                            int const                 *E_IT_in,
                            int                        vertexOffset_in,
                            int                        tableOffset_in) :
                            vertex (vertex_in),
                            varying(varying_in),
                            vertexDesc(vertexDesc_in),
                            varyingDesc(varyingDesc_in),
                            E_IT   (E_IT_in),
                            vertexOffset(vertexOffset_in),
                            tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeRestrictedEdge(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *E_IT, int vertexOffset, int tableOffset,
    int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBRestrictedEdgeKernel kernel(vertex, varying, vertexDesc, varyingDesc, E_IT,
                                   vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBVertexKernelA {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *V_ITa;
    float const  *V_W;
    int           vertexOffset;
    int           tableOffset;
    int           pass;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vertexDesc.length == 4 && varying == NULL) {
            ComputeVertexAKernel<4>(vertex, V_ITa, V_W, vertexOffset, tableOffset,
                                 r.begin(), r.end(), pass);
        }
        else if (vertexDesc.length == 8 && varying == NULL) {
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

                if (not pass) {
                    clear(vertex, dstIndex, vertexDesc);
                    clear(varying, dstIndex, varyingDesc);
                }

                if (eidx0 == -1 || (pass == 0 && (n == -1))) {
                    addWithWeight(vertex, dstIndex, p, weight, vertexDesc);
                } else {
                    addWithWeight(vertex, dstIndex, p, weight * 0.75f, vertexDesc);
                    addWithWeight(vertex, dstIndex, eidx0, weight * 0.125f, vertexDesc);
                    addWithWeight(vertex, dstIndex, eidx1, weight * 0.125f, vertexDesc);
                }

                if (not pass)
                    addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
            }
        }
    }

    TBBVertexKernelA(TBBVertexKernelA const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->V_ITa  = other.V_ITa;
        this->V_W    = other.V_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
        this->pass         = other.pass;
    }

    TBBVertexKernelA(float                     *vertex_in,
                     float                     *varying_in,
                     OsdVertexBufferDescriptor const &vertexDesc_in,
                     OsdVertexBufferDescriptor const &varyingDesc_in,
                     int const                 *V_ITa_in,
                     float const               *V_W_in,
                     int                        vertexOffset_in,
                     int                        tableOffset_in,
                     int                        pass_in) :
                     vertex (vertex_in),
                     varying(varying_in),
                     vertexDesc(vertexDesc_in),
                     varyingDesc(varyingDesc_in),
                     V_ITa  (V_ITa_in),
                     V_W    (V_W_in),
                     vertexOffset(vertexOffset_in),
                     tableOffset(tableOffset_in),
                     pass(pass_in)
    {};
};

void OsdTbbComputeVertexA(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *V_ITa, float const *V_W, int vertexOffset, int tableOffset,
    int start, int end, int pass) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBVertexKernelA kernel(vertex, varying, vertexDesc, varyingDesc,
                            V_ITa, V_W,
                            vertexOffset, tableOffset, pass);
    tbb::parallel_for(range, kernel);
}

class TBBVertexKernelB {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *V_ITa;
    int const    *V_IT;
    float const  *V_W;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vertexDesc.length == 4 && varying == NULL) {
            ComputeVertexBKernel<4>(vertex, V_ITa, V_IT, V_W,
                vertexOffset, tableOffset, r.begin(), r.end());
        }
        else if(vertexDesc.length == 8 && varying == NULL) {
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
                clear(vertex, dstIndex, vertexDesc);
                clear(varying, dstIndex, varyingDesc);

                addWithWeight(vertex, dstIndex, p, weight * wv, vertexDesc);

                for (int j = 0; j < n; ++j) {
                    addWithWeight(vertex, dstIndex, V_IT[h+j*2], weight * wp, vertexDesc);
                    addWithWeight(vertex, dstIndex, V_IT[h+j*2+1], weight * wp, vertexDesc);
                }
                addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
            }
        }
    }

    TBBVertexKernelB(TBBVertexKernelB const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->V_ITa  = other.V_ITa;
        this->V_IT   = other.V_IT;
        this->V_W    = other.V_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBVertexKernelB(float                     *vertex_in,
                     float                     *varying_in,
                     OsdVertexBufferDescriptor const &vertexDesc_in,
                     OsdVertexBufferDescriptor const &varyingDesc_in,
                     int const                 *V_ITa_in,
                     int const                 *V_IT_in,
                     float const               *V_W_in,
                     int                        vertexOffset_in,
                     int                        tableOffset_in) :
                     vertex (vertex_in),
                     varying(varying_in),
                     vertexDesc(vertexDesc_in),
                     varyingDesc(varyingDesc_in),
                     V_ITa  (V_ITa_in),
                     V_IT   (V_IT_in),
                     V_W    (V_W_in),
                     vertexOffset(vertexOffset_in),
                     tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeVertexB(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *V_ITa, int const *V_IT, float const *V_W,
    int vertexOffset, int tableOffset, int start, int end) {

    tbb::blocked_range<int> range(start, end, grain_size);
    TBBVertexKernelB kernel(vertex, varying, vertexDesc, varyingDesc,
                            V_ITa, V_IT, V_W,
                            vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBRestrictedVertexKernelA {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *V_ITa;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
            int p     = V_ITa[5*i+2];
            int eidx0 = V_ITa[5*i+3];
            int eidx1 = V_ITa[5*i+4];

            int dstIndex = i + vertexOffset - tableOffset;

            clear(vertex, dstIndex, vertexDesc);
            clear(varying, dstIndex, varyingDesc);
            addWithWeight(vertex, dstIndex, p, 0.75f, vertexDesc);
            addWithWeight(vertex, dstIndex, eidx0, 0.125f, vertexDesc);
            addWithWeight(vertex, dstIndex, eidx1, 0.125f, vertexDesc);
            addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
        }
    }

    TBBRestrictedVertexKernelA(TBBRestrictedVertexKernelA const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->V_ITa  = other.V_ITa;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBRestrictedVertexKernelA(float                     *vertex_in,
                               float                     *varying_in,
                               OsdVertexBufferDescriptor const &vertexDesc_in,
                               OsdVertexBufferDescriptor const &varyingDesc_in,
                               int const                 *V_ITa_in,
                               int                        vertexOffset_in,
                               int                        tableOffset_in) :
                               vertex (vertex_in),
                               varying(varying_in),
                               vertexDesc(vertexDesc_in),
                               varyingDesc(varyingDesc_in),
                               V_ITa  (V_ITa_in),
                               vertexOffset(vertexOffset_in),
                               tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeRestrictedVertexA(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *V_ITa, int vertexOffset, int tableOffset,
    int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBRestrictedVertexKernelA kernel(vertex, varying, vertexDesc, varyingDesc,
                                      V_ITa,
                                      vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBRestrictedVertexKernelB1 {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *V_ITa;
    int const    *V_IT;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
            int h = V_ITa[5*i];
            int p = V_ITa[5*i+2];

            int dstIndex = i + vertexOffset - tableOffset;
            clear(vertex, dstIndex, vertexDesc);
            clear(varying, dstIndex, varyingDesc);

            addWithWeight(vertex, dstIndex, p, 0.5f, vertexDesc);

            for (int j = 0; j < 8; ++j)
                addWithWeight(vertex, dstIndex, V_IT[h+j], 0.0625f, vertexDesc);
            addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
        }
    }

    TBBRestrictedVertexKernelB1(TBBRestrictedVertexKernelB1 const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->V_ITa  = other.V_ITa;
        this->V_IT   = other.V_IT;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBRestrictedVertexKernelB1(float                     *vertex_in,
                                float                     *varying_in,
                                OsdVertexBufferDescriptor const &vertexDesc_in,
                                OsdVertexBufferDescriptor const &varyingDesc_in,
                                int const                 *V_ITa_in,
                                int const                 *V_IT_in,
                                int                        vertexOffset_in,
                                int                        tableOffset_in) :
                                vertex (vertex_in),
                                varying(varying_in),
                                vertexDesc(vertexDesc_in),
                                varyingDesc(varyingDesc_in),
                                V_ITa  (V_ITa_in),
                                V_IT   (V_IT_in),
                                vertexOffset(vertexOffset_in),
                                tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeRestrictedVertexB1(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *V_ITa, int const *V_IT,
    int vertexOffset, int tableOffset, int start, int end) {

    tbb::blocked_range<int> range(start, end, grain_size);
    TBBRestrictedVertexKernelB1 kernel(vertex, varying, vertexDesc, varyingDesc,
                                       V_ITa, V_IT,
                                       vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBRestrictedVertexKernelB2 {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *V_ITa;
    int const    *V_IT;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
            int h = V_ITa[5*i];
            int n = V_ITa[5*i+1];
            int p = V_ITa[5*i+2];

            float wp = 1.0f/static_cast<float>(n*n);
            float wv = (n-2.0f) * n * wp;

            int dstIndex = i + vertexOffset - tableOffset;
            clear(vertex, dstIndex, vertexDesc);
            clear(varying, dstIndex, varyingDesc);

            addWithWeight(vertex, dstIndex, p, wv, vertexDesc);

            for (int j = 0; j < n; ++j) {
                addWithWeight(vertex, dstIndex, V_IT[h+j*2], wp, vertexDesc);
                addWithWeight(vertex, dstIndex, V_IT[h+j*2+1], wp, vertexDesc);
            }
            addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
        }
    }

    TBBRestrictedVertexKernelB2(TBBRestrictedVertexKernelB2 const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->V_ITa  = other.V_ITa;
        this->V_IT   = other.V_IT;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBRestrictedVertexKernelB2(float                     *vertex_in,
                                float                     *varying_in,
                                OsdVertexBufferDescriptor const &vertexDesc_in,
                                OsdVertexBufferDescriptor const &varyingDesc_in,
                                int const                 *V_ITa_in,
                                int const                 *V_IT_in,
                                int                        vertexOffset_in,
                                int                        tableOffset_in) :
                                vertex (vertex_in),
                                varying(varying_in),
                                vertexDesc(vertexDesc_in),
                                varyingDesc(varyingDesc_in),
                                V_ITa  (V_ITa_in),
                                V_IT   (V_IT_in),
                                vertexOffset(vertexOffset_in),
                                tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeRestrictedVertexB2(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *V_ITa, int const *V_IT,
    int vertexOffset, int tableOffset, int start, int end) {

    tbb::blocked_range<int> range(start, end, grain_size);
    TBBRestrictedVertexKernelB2 kernel(vertex, varying, vertexDesc, varyingDesc,
                                       V_ITa, V_IT,
                                       vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBLoopVertexKernelB {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *V_ITa;
    int const    *V_IT;
    float const  *V_W;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vertexDesc.length == 4 && varying == NULL) {
            ComputeLoopVertexBKernel<4>(vertex, V_ITa, V_IT, V_W, vertexOffset,
                                  tableOffset, r.begin(), r.end());
        }
        else if(vertexDesc.length == 8 && varying == NULL) {
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
                clear(vertex, dstIndex, vertexDesc);
                clear(varying, dstIndex, varyingDesc);

                addWithWeight(vertex, dstIndex, p, weight * (1.0f - (beta * n)), vertexDesc);

                for (int j = 0; j < n; ++j)
                    addWithWeight(vertex, dstIndex, V_IT[h+j], weight * beta, vertexDesc);

                addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
            }
        }
    }

    TBBLoopVertexKernelB(TBBLoopVertexKernelB const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->V_ITa  = other.V_ITa;
        this->V_IT   = other.V_IT;
        this->V_W    = other.V_W;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBLoopVertexKernelB(float                     *vertex_in,
                         float                     *varying_in,
                         OsdVertexBufferDescriptor const &vertexDesc_in,
                         OsdVertexBufferDescriptor const &varyingDesc_in,
                         int const                 *V_ITa_in,
                         int const                 *V_IT_in,
                         float const               *V_W_in,
                         int                        vertexOffset_in,
                         int                        tableOffset_in) :
                         vertex (vertex_in),
                         varying(varying_in),
                         vertexDesc(vertexDesc_in),
                         varyingDesc(varyingDesc_in),
                         V_ITa  (V_ITa_in),
                         V_IT   (V_IT_in),
                         V_W    (V_W_in),
                         vertexOffset(vertexOffset_in),
                         tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeLoopVertexB(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *V_ITa, int const *V_IT, float const *V_W,
    int vertexOffset, int tableOffset, int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBLoopVertexKernelB kernel(vertex, varying, vertexDesc, varyingDesc,
                                V_ITa, V_IT, V_W,
                                vertexOffset, tableOffset);

    tbb::parallel_for(range, kernel);
}

class TBBBilinearEdgeKernel {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *E_IT;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        if(vertexDesc.length == 4 && varying == NULL) {
            ComputeBilinearEdgeKernel<4>(vertex, E_IT, vertexOffset, tableOffset,
                                         r.begin(), r.end());
        }
        else if(vertexDesc.length == 8 && varying == NULL) {
            ComputeBilinearEdgeKernel<8>(vertex, E_IT, vertexOffset, tableOffset,
                                         r.begin(), r.end());
        }
        else {
            for (int i = r.begin() + tableOffset; i < r.end() + tableOffset; i++) {
                int eidx0 = E_IT[2*i+0];
                int eidx1 = E_IT[2*i+1];

                int dstIndex = i + vertexOffset - tableOffset;
                clear(vertex, dstIndex, vertexDesc);
                clear(varying, dstIndex, varyingDesc);

                addWithWeight(vertex, dstIndex, eidx0, 0.5f, vertexDesc);
                addWithWeight(vertex, dstIndex, eidx1, 0.5f, vertexDesc);

                addWithWeight(varying, dstIndex, eidx0, 0.5f, varyingDesc);
                addWithWeight(varying, dstIndex, eidx1, 0.5f, varyingDesc);
            }
        }
    }

    TBBBilinearEdgeKernel(TBBBilinearEdgeKernel const &other)
    {
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->E_IT   = other.E_IT;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBBilinearEdgeKernel(float                     *vertex_in,
                          float                     *varying_in,
                          OsdVertexBufferDescriptor const &vertexDesc_in,
                          OsdVertexBufferDescriptor const &varyingDesc_in,
                          int const                 *E_IT_in,
                          int                        vertexOffset_in,
                          int                        tableOffset_in) :
                          vertex (vertex_in),
                          varying(varying_in),
                          vertexDesc(vertexDesc_in),
                          varyingDesc(varyingDesc_in),
                          E_IT   (E_IT_in),
                          vertexOffset(vertexOffset_in),
                          tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeBilinearEdge(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *E_IT, int vertexOffset, int tableOffset, int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBBilinearEdgeKernel kernel(vertex, varying, vertexDesc, varyingDesc,
                                 E_IT, vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

class TBBBilinearVertexKernel {
    float        *vertex;
    float        *varying;
    OsdVertexBufferDescriptor vertexDesc;
    OsdVertexBufferDescriptor varyingDesc;
    int const    *V_ITa;
    int           vertexOffset;
    int           tableOffset;

public:
    void operator() (tbb::blocked_range<int> const &r) const {
        int numVertexElements  = vertexDesc.length;
        int numVaryingElements = varyingDesc.length;
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
        this->vertex = other.vertex;
        this->varying= other.varying;
        this->vertexDesc = other.vertexDesc;
        this->varyingDesc = other.varyingDesc;
        this->V_ITa  = other.V_ITa;
        this->vertexOffset = other.vertexOffset;
        this->tableOffset  = other.tableOffset;
    }

    TBBBilinearVertexKernel(float                     *vertex_in,
                            float                     *varying_in,
                            OsdVertexBufferDescriptor const &vertexDesc_in,
                            OsdVertexBufferDescriptor const &varyingDesc_in,
                            int const                 *V_ITa_in,
                            int                        vertexOffset_in,
                            int                        tableOffset_in) :
                            vertex (vertex_in),
                            varying(varying_in),
                            vertexDesc(vertexDesc_in),
                            varyingDesc(varyingDesc_in),
                            V_ITa  (V_ITa_in),
                            vertexOffset(vertexOffset_in),
                            tableOffset(tableOffset_in)
    {};
};

void OsdTbbComputeBilinearVertex(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    int const *V_ITa, int vertexOffset, int tableOffset, int start, int end) {
    tbb::blocked_range<int> range(start, end, grain_size);
    TBBBilinearVertexKernel kernel(vertex, varying, vertexDesc, varyingDesc,
                                   V_ITa, vertexOffset, tableOffset);
    tbb::parallel_for(range, kernel);
}

void OsdTbbEditVertexAdd(
    float *vertex,
    OsdVertexBufferDescriptor const &vertexDesc,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    unsigned int const *editIndices, float const *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {

        if (vertex) {
            int editIndex = editIndices[i] + vertexOffset;
            float *dst = vertex + editIndex * vertexDesc.stride + primVarOffset;

            for (int j = 0; j < primVarWidth; ++j) {
                dst[j] += editValues[j];
            }
        }
    }
}

void OsdTbbEditVertexSet(
    float *vertex,
    OsdVertexBufferDescriptor const &vertexDesc,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    unsigned int const *editIndices, float const *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {

        if (vertex) {
            int editIndex = editIndices[i] + vertexOffset;
            float *dst = vertex + editIndex * vertexDesc.stride + primVarOffset;

            for (int j = 0; j < primVarWidth; ++j) {
                dst[j] = editValues[j];
            }
        }
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
