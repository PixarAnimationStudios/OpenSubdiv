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

#pragma once
#include "buffer.h"

struct TopologyImpl;

namespace shim {

    struct BoundaryMode {
        enum e {
            NONE,
            EDGE_ONLY,
            EDGE_AND_CORNER,
            ALWAYS_SHARP,
        };
    };

    class Subdivider;

    class Topology {
    public:
        Topology(const shim::HomogeneousBuffer& indices,
                 const shim::HomogeneousBuffer& valences);
        ~Topology();

        void copyAnnotationsFrom(const Topology& topo);
 
        void finalize();
        
        BoundaryMode::e getBoundaryMode() const;
        void setBoundaryMode(BoundaryMode::e bm);

        int getNumVertices() const;
        float getVertexSharpness(int vertex) const;
        void setVertexSharpness(int vertex, float sharpness);
        
        int getNumFaces() const;
        bool getFaceHole(int face) const;
        void setFaceHole(int face, bool isHole);

        int getNumEdges(int face) const;
        float getEdgeSharpness(int face, int edge) const;
        void setEdgeSharpness(int face, int edge, float sharpness);

    private:
        TopologyImpl *self;
        friend class shim::Subdivider;
    };

}
