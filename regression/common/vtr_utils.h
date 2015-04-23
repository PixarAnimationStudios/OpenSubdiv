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

#ifndef VTR_UTILS_H
#define VTR_UTILS_H

#include <far/topologyRefinerFactory.h>
#include <far/types.h>

#include "../../regression/common/shape_utils.h"

//------------------------------------------------------------------------------

inline OpenSubdiv::Sdc::SchemeType
GetSdcType(Shape const & shape) {

    OpenSubdiv::Sdc::SchemeType type=OpenSubdiv::Sdc::SCHEME_CATMARK;

    switch (shape.scheme) {
        case kBilinear: type = OpenSubdiv::Sdc::SCHEME_BILINEAR; break;
        case kCatmark : type = OpenSubdiv::Sdc::SCHEME_CATMARK; break;
        case kLoop    : type = OpenSubdiv::Sdc::SCHEME_LOOP; break;
    }
    return type;
}

inline OpenSubdiv::Sdc::Options
GetSdcOptions(Shape const & shape) {

    typedef OpenSubdiv::Sdc::Options Options;

    Options result;

    result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_EDGE_ONLY);
    result.SetCreasingMethod(Options::CREASE_UNIFORM);
    result.SetTriangleSubdivision(Options::TRI_SUB_CATMARK);

    for (int i=0; i<(int)shape.tags.size(); ++i) {

        Shape::tag * t = shape.tags[i];

        if (t->name=="interpolateboundary") {
            if ((int)t->intargs.size()!=1) {
                printf("expecting 1 integer for \"interpolateboundary\" tag n. %d\n", i);
                continue;
            }
            switch( t->intargs[0] ) {
                case 0 : result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_NONE); break;
                case 1 : result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_EDGE_AND_CORNER); break;
                case 2 : result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_EDGE_ONLY); break;
                default: printf("unknown interpolate boundary : %d\n", t->intargs[0] ); break;
            }
        } else if (t->name=="facevaryinginterpolateboundary") {
            if ((int)t->intargs.size()!=1) {
                printf("expecting 1 integer for \"facevaryinginterpolateboundary\" tag n. %d\n", i);
                continue;
            }
            switch( t->intargs[0] ) {
                case 0 : result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_NONE); break;
                case 1 : result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_CORNERS_ONLY); break;
                case 2 : result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_CORNERS_PLUS1); break;
                case 3 : result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_CORNERS_PLUS2); break;
                case 4 : result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_BOUNDARIES); break;
                case 5 : result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_ALL); break;
                default: printf("unknown interpolate boundary : %d\n", t->intargs[0] ); break;
            }
        } else if (t->name=="facevaryingpropagatecorners") {
            if ((int)t->intargs.size()==1) {
                // XXXX no propagate corners in Options
                assert(0);
            } else
                printf( "expecting single int argument for \"facevaryingpropagatecorners\"\n" );
        } else if (t->name=="creasemethod") {

            if ((int)t->stringargs.size()==0) {
                printf("the \"creasemethod\" tag expects a string argument\n");
                continue;
            }

            if (t->stringargs[0]=="normal") {
                result.SetCreasingMethod(Options::CREASE_UNIFORM);
            } else if (t->stringargs[0]=="chaikin") {
                result.SetCreasingMethod(Options::CREASE_CHAIKIN);
            } else {
                printf("the \"creasemethod\" tag only accepts \"normal\" or \"chaikin\" as value (%s)\n", t->stringargs[0].c_str());
            }
        } else if (t->name=="smoothtriangles") {

            if (shape.scheme!=kCatmark) {
                printf("the \"smoothtriangles\" tag can only be applied to Catmark meshes\n");
                continue;
            }
            if (t->stringargs[0]=="catmark") {
                result.SetTriangleSubdivision(Options::TRI_SUB_CATMARK);
            } else if (t->stringargs[0]=="smooth") {
                result.SetTriangleSubdivision(Options::TRI_SUB_SMOOTH);
            } else {
                printf("the \"smoothtriangles\" tag only accepts \"catmark\" or \"smooth\" as value (%s)\n", t->stringargs[0].c_str());
            }
        }
    }

    return result;
}

void
InterpolateFVarData(OpenSubdiv::Far::TopologyRefiner & refiner,
    Shape const & shape, std::vector<float> & fvarData);
//------------------------------------------------------------------------------

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

template <>
inline bool
TopologyRefinerFactory<Shape>::resizeComponentTopology(
    Far::TopologyRefiner & refiner, Shape const & shape) {

    int nfaces = shape.GetNumFaces(),
        nverts = shape.GetNumVertices();

    refiner.setNumBaseFaces(nfaces);
    for (int i=0; i<nfaces; ++i) {

        int nv = shape.nvertsPerFace[i];
        refiner.setNumBaseFaceVertices(i, nv);
    }

    // Vertices and vert-faces and vert-edges
    refiner.setNumBaseVertices(nverts);

    return true;
}

//----------------------------------------------------------
template <>
inline bool
TopologyRefinerFactory<Shape>::assignComponentTopology(
    Far::TopologyRefiner & refiner, Shape const & shape) {

    { // Face relations:
        int nfaces = refiner.GetNumFaces(0);

        for (int i=0, ofs=0; i < nfaces; ++i) {

            Far::IndexArray dstFaceVerts = refiner.setBaseFaceVertices(i);
            //IndexArray dstFaceEdges = refiner.setBaseFaceEdges(i);

            for (int j=0; j<dstFaceVerts.size(); ++j) {
                dstFaceVerts[j] = shape.faceverts[ofs++];
            }
        }
    }
    return true;
}

//----------------------------------------------------------
template <>
inline bool
TopologyRefinerFactory<Shape>::assignFaceVaryingTopology(
    Far::TopologyRefiner & refiner, Shape const & shape) {

    // UV layyout (we only parse 1 channel)
    if (not shape.faceuvs.empty()) {

        int nfaces = refiner.GetNumFaces(0),
           channel = refiner.createBaseFVarChannel( (int)shape.faceuvs.size() );

        for (int i=0, ofs=0; i < nfaces; ++i) {

            Far::IndexArray dstFaceUVs =
                refiner.setBaseFVarFaceValues(i, channel);

            for (int j=0; j<dstFaceUVs.size(); ++j) {
                dstFaceUVs[j] = shape.faceuvs[ofs++];
            }
        }
    }
    return true;
}

//----------------------------------------------------------
template <>
inline bool
TopologyRefinerFactory<Shape>::assignComponentTags(
    Far::TopologyRefiner & refiner, Shape const & shape) {


    for (int i=0; i<(int)shape.tags.size(); ++i) {

        Shape::tag * t = shape.tags[i];

        if (t->name=="crease") {

            for (int j=0; j<(int)t->intargs.size()-1; j += 2) {

                OpenSubdiv::Vtr::Index edge = refiner.FindEdge(/*level*/0, t->intargs[j], t->intargs[j+1]);
                if (edge==OpenSubdiv::Vtr::INDEX_INVALID) {
                    printf("cannot find edge for crease tag (%d,%d)\n", t->intargs[j], t->intargs[j+1] );
                    return false;
                } else {
                    int nfloat = (int) t->floatargs.size();
                    refiner.setBaseEdgeSharpness(edge,
                        std::max(0.0f, ((nfloat > 1) ? t->floatargs[j] : t->floatargs[0])));
                }
            }
        } else if (t->name=="corner") {

            for (int j=0; j<(int)t->intargs.size(); ++j) {
                int vertex = t->intargs[j];
                if (vertex<0 or vertex>=refiner.GetNumVertices(/*level*/0)) {
                    printf("cannot find vertex for corner tag (%d)\n", vertex );
                    return false;
                } else {
                    int nfloat = (int) t->floatargs.size();
                    refiner.setBaseVertexSharpness(vertex,
                        std::max(0.0f, ((nfloat > 1) ? t->floatargs[j] : t->floatargs[0])));
                }
            }
        }
    }
    { // Hole tags
        for (int i=0; i<(int)shape.tags.size(); ++i) {
            Shape::tag * t = shape.tags[i];
            if (t->name=="hole") {
                for (int j=0; j<(int)t->intargs.size(); ++j) {
                    refiner.setBaseFaceHole(t->intargs[j], true);
                }
            }
        }
    }
    return true;
}

template <>
inline void
TopologyRefinerFactory<Shape>::reportInvalidTopology(
    TopologyRefinerFactory::TopologyError /* errCode */, char const * msg, Shape const & /* shape */) {
    Warning(msg);
}

} // namespace Far

} // namespace OPENSUBDIV_VERSION
} // namespace OpenSubdiv

//------------------------------------------------------------------------------

#endif /* VTR_UTILS_H */
