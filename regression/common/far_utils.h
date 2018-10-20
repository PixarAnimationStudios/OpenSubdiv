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

#ifndef FAR_UTILS_H
#define FAR_UTILS_H

#include "shape_utils.h"

#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/types.h>

#include <cstdio>


//------------------------------------------------------------------------------

inline Scheme
ConvertSdcTypeToShapeScheme(OpenSubdiv::Sdc::SchemeType sdcScheme) {

    switch (sdcScheme) {
        case OpenSubdiv::Sdc::SCHEME_BILINEAR: return kBilinear;
        case OpenSubdiv::Sdc::SCHEME_CATMARK:  return kCatmark;
        case OpenSubdiv::Sdc::SCHEME_LOOP:     return kLoop;
        default: printf("unknown Sdc::SchemeType : %d\n", (int)sdcScheme); break;
    }
    return kCatmark;
}

inline OpenSubdiv::Sdc::SchemeType
ConvertShapeSchemeToSdcType(Scheme shapeScheme) {

    switch (shapeScheme) {
        case kBilinear: return OpenSubdiv::Sdc::SCHEME_BILINEAR;
        case kCatmark:  return OpenSubdiv::Sdc::SCHEME_CATMARK;
        case kLoop:     return OpenSubdiv::Sdc::SCHEME_LOOP;
        default: printf("unknown Shape Scheme : %d\n", (int)shapeScheme); break;
    }
    return OpenSubdiv::Sdc::SCHEME_CATMARK;
}

inline OpenSubdiv::Sdc::SchemeType
GetSdcType(Shape const & shape) {

    return ConvertShapeSchemeToSdcType(shape.scheme);
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

//------------------------------------------------------------------------------

void
InterpolateFVarData(OpenSubdiv::Far::TopologyRefiner & refiner,
    Shape const & shape, std::vector<float> & fvarData);

//------------------------------------------------------------------------------

template <class T>
OpenSubdiv::Far::TopologyRefiner *
InterpolateFarVertexData(Shape const & shape, int maxlevel, std::vector<T> &data) {

    typedef OpenSubdiv::Far::TopologyRefiner FarTopologyRefiner;
    typedef OpenSubdiv::Far::TopologyRefinerFactory<Shape> FarTopologyRefinerFactory;

    // Far interpolation
    FarTopologyRefiner * refiner =
        FarTopologyRefinerFactory::Create(shape,
            FarTopologyRefinerFactory::Options(
                GetSdcType(shape), GetSdcOptions(shape)));
    assert(refiner);

    FarTopologyRefiner::UniformOptions options(maxlevel);
    options.fullTopologyInLastLevel=true;
    refiner->RefineUniform(options);

    // populate coarse mesh positions
    data.resize(refiner->GetNumVerticesTotal());
    for (int i=0; i<refiner->GetLevel(0).GetNumVertices(); i++) {
        data[i].SetPosition(shape.verts[i*3+0],
                            shape.verts[i*3+1],
                            shape.verts[i*3+2]);
    }

    T * srcVerts = &data[0];
    T * dstVerts = srcVerts + refiner->GetLevel(0).GetNumVertices();
    OpenSubdiv::Far::PrimvarRefiner primvarRefiner(*refiner);

    for (int i = 1; i <= refiner->GetMaxLevel(); ++i) {
        primvarRefiner.Interpolate(i, srcVerts, dstVerts);
        srcVerts = dstVerts;
        dstVerts += refiner->GetLevel(i).GetNumVertices();
    }
    return refiner;
}

template <class T>
OpenSubdiv::Far::TopologyRefiner *
InterpolateFarVertexData(const char *shapeStr, Scheme scheme, int maxlevel,
    std::vector<T> &data) {

    Shape const * shape = Shape::parseObj(shapeStr, scheme);

    OpenSubdiv::Far::TopologyRefiner * refiner =
            InterpolateFarVertexData(*shape, maxlevel, data);

    delete shape;
    return refiner;
}


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

    setNumBaseFaces(refiner, nfaces);
    for (int i=0; i<nfaces; ++i) {

        int nv = shape.nvertsPerFace[i];
        setNumBaseFaceVertices(refiner, i, nv);
    }

    // Vertices and vert-faces and vert-edges
    setNumBaseVertices(refiner, nverts);

    return true;
}

//----------------------------------------------------------
template <>
inline bool
TopologyRefinerFactory<Shape>::assignComponentTopology(
    Far::TopologyRefiner & refiner, Shape const & shape) {

    { // Face relations:
        int nfaces = getNumBaseFaces(refiner);

        for (int i=0, ofs=0; i < nfaces; ++i) {

            Far::IndexArray dstFaceVerts = getBaseFaceVertices(refiner, i);

            if (shape.isLeftHanded) {
                dstFaceVerts[0] = shape.faceverts[ofs++];
                for (int j=dstFaceVerts.size()-1; j>0; --j) {
                    dstFaceVerts[j] = shape.faceverts[ofs++];
                }
            } else {
                for (int j=0; j<dstFaceVerts.size(); ++j) {
                    dstFaceVerts[j] = shape.faceverts[ofs++];
                }
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

    // UV layout (we only parse 1 channel)
    if (! shape.faceuvs.empty()) {

        int nfaces = getNumBaseFaces(refiner),
           channel = createBaseFVarChannel(refiner, (int)shape.uvs.size()/2 );

        for (int i=0, ofs=0; i < nfaces; ++i) {

            Far::IndexArray dstFaceUVs = getBaseFaceFVarValues(refiner, i, channel);

            if (shape.isLeftHanded) {
                dstFaceUVs[0] = shape.faceuvs[ofs++];
                for (int j=dstFaceUVs.size()-1; j > 0; --j) {
                    dstFaceUVs[j] = shape.faceuvs[ofs++];
                }
            } else {
                for (int j=0; j<dstFaceUVs.size(); ++j) {
                    dstFaceUVs[j] = shape.faceuvs[ofs++];
                }
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

                OpenSubdiv::Far::Index edge = findBaseEdge(refiner, t->intargs[j], t->intargs[j+1]);
                if (edge==OpenSubdiv::Far::INDEX_INVALID) {
                    printf("cannot find edge for crease tag (%d,%d)\n", t->intargs[j], t->intargs[j+1] );
                    return false;
                } else {
                    int nfloat = (int) t->floatargs.size();
                    setBaseEdgeSharpness(refiner, edge,
                        std::max(0.0f, ((nfloat > 1) ? t->floatargs[j] : t->floatargs[0])));
                }
            }
        } else if (t->name=="corner") {

            for (int j=0; j<(int)t->intargs.size(); ++j) {
                int vertex = t->intargs[j];
                if (vertex<0 || vertex>=getNumBaseVertices(refiner)) {
                    printf("cannot find vertex for corner tag (%d)\n", vertex );
                    return false;
                } else {
                    int nfloat = (int) t->floatargs.size();
                    setBaseVertexSharpness(refiner, vertex,
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
                    setBaseFaceHole(refiner, t->intargs[j], true);
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

#endif /* FAR_UTILS_H */
