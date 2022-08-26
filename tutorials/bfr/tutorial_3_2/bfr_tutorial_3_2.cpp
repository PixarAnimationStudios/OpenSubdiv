//
//   Copyright 2021 Pixar
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

//------------------------------------------------------------------------------
//  Tutorial description:
//
//      This tutorial is a variation of tutorials showing simple uniform
//      tessellation. Rather than constructing and evaluating a Surface at
//      a time, this tutorial shows how Surfaces can be created and saved
//      for repeated use.
//
//      A simple SurfaceCache class is created that creates and stores the
//      Surface for each face, along with the patch points associated with
//      it. The main tessellation function remains essentially the same,
//      but here it access the Surfaces from the SurfaceCache rather than
//      computing them locally.
//
//      Note that while this example illustrated the retention of all
//      Surfaces for a mesh, this behavior is not recommended. It does not
//      scale well for large meshes and undermines the memory savings that
//      transient use of Surfaces is designed to achieve. Rather than
//      storing Surfaces for all faces, maintaining a priority queue for a
//      fixed number may be a reasonable compromise.
//

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>
#include <opensubdiv/bfr/tessellation.h>

#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <cstdio>

//  Local headers with support for this tutorial in "namespace tutorial"
#include "./meshLoader.h"
#include "./objWriter.h"

using namespace OpenSubdiv;

//
//  Simple command line arguments to provide input and run-time options:
//
class Args {
public:
    std::string     inputObjFile;
    std::string     outputObjFile;
    Sdc::SchemeType schemeType;
    int             tessUniformRate;
    bool            tessQuadsFlag;

public:
    Args(int argc, char * argv[]) :
        inputObjFile(),
        outputObjFile(),
        schemeType(Sdc::SCHEME_CATMARK),
        tessUniformRate(5),
        tessQuadsFlag(false) {

        for (int i = 1; i < argc; ++i) {
            if (strstr(argv[i], ".obj")) {
                if (inputObjFile.empty()) {
                    inputObjFile = std::string(argv[i]);
                } else {
                    fprintf(stderr,
                        "Warning: Extra Obj file '%s' ignored\n", argv[i]);
                }
            } else if (!strcmp(argv[i], "-o")) {
                if (++i < argc) outputObjFile = std::string(argv[i]);
            } else if (!strcmp(argv[i], "-bilinear")) {
                schemeType = Sdc::SCHEME_BILINEAR;
            } else if (!strcmp(argv[i], "-catmark")) {
                schemeType = Sdc::SCHEME_CATMARK;
            } else if (!strcmp(argv[i], "-loop")) {
                schemeType = Sdc::SCHEME_LOOP;
            } else if (!strcmp(argv[i], "-res")) {
                if (++i < argc) tessUniformRate = atoi(argv[i]);
            } else if (!strcmp(argv[i], "-quads")) {
                tessQuadsFlag = true;
            } else {
                fprintf(stderr,
                    "Warning: Unrecognized argument '%s' ignored\n", argv[i]);
            }
        }
    }

private:
    Args() { }
};

//
//  This simple class creates and dispenses Surfaces for all faces of
//  a mesh. It consists primarily of an array of simple structs (entries)
//  for each face and a single array of patch points for all Surfaces
//  created.
//
//  There are many ways to create such a cache depending on requirements.
//  This is a simple example, but the interface presents some options that
//  are worth considering. A SurfaceCache is constructed here given the
//  following:
//
//      - a reference to the SurfaceFactory:
//          - the cache could just as easily take a reference to the mesh
//            and construct the SurfaceFactory internally
//
//      - the position data for the mesh:
//          - this is needed to compute patch points for the Surfaces
//          - if caching UVs or any other primvar, other data needs to be
//            provided -- along with the interpolation type for that data
//            (vertex, face-varying, etc.)
//
//      - option to "cache patch points":
//          - the cache could store the Surfaces only or also include
//            their patch points
//          - storing patch points takes more memory but will eliminate
//            any preparation time for evaluation of the Surface
//
//      - option to "cache all surfaces":
//          - the benefits to caching simple linear or regular surfaces
//            are minimal -- and may even be detrimental
//          - so only caching non-linear irregular surfaces is an option
//            worth considering
//
//  The SurfaceCache implementation here provides the options noted above.
//  But for simplicity, the actual usage of the SurfaceCache does not deal
//  with the permutations of additional work that is necessary when the
//  Surfaces or their patch points are not cached.
//
class SurfaceCache {
public:
    typedef Bfr::Surface<float>          Surface;
    typedef Bfr::RefinerSurfaceFactory<> SurfaceFactory;

public:
    SurfaceCache(SurfaceFactory     const & surfaceFactory,
                 std::vector<float> const & meshPoints,
                 bool                       cachePatchPoints = true,
                 bool                       cacheAllSurfaces = true);
    SurfaceCache() = delete;
    ~SurfaceCache() = default;

    //
    //  Public methods to retrieved cached Surfaces and their pre-computed
    //  patch points:
    //
    bool FaceHasLimitSurface(int face) { return _entries[face].hasLimit; }

    Surface const * GetSurface(int face) { return _entries[face].surface.get();}

    float const * GetPatchPoints(int face) { return getPatchPoints(face); }

private:
    //  Simple struct to keep track of Surface and more for each face:
    struct FaceEntry {
        FaceEntry() : surface(), hasLimit(false), pointOffset(-1) { }

        std::unique_ptr<Surface const> surface;
        bool hasLimit;
        int  pointOffset;
    };

    //  Non-const version to be used internally to aide assignment:
    float * getPatchPoints(int face) {
        return (_entries[face].surface && !_points.empty()) ?
               (_points.data() + _entries[face].pointOffset * 3) : 0;
    }

private:
    std::vector<FaceEntry> _entries;
    std::vector<float>     _points;
};

SurfaceCache::SurfaceCache(SurfaceFactory     const & surfaceFactory,
                           std::vector<float> const & meshPoints,
                           bool                       cachePatchPoints,
                           bool                       cacheAllSurfaces) {

    int numFaces = surfaceFactory.GetNumFaces();

    _entries.resize(numFaces);

    int numPointsInCache = 0;
    for (int face = 0; face < numFaces; ++face) {
        Surface * s = surfaceFactory.CreateVertexSurface<float>(face);
        if (s) {
            FaceEntry & entry = _entries[face];
            entry.hasLimit = true;

            if (cacheAllSurfaces || (!s->IsRegular() && !s->IsLinear())) {
                entry.surface.reset(s);
                entry.pointOffset = numPointsInCache;

                numPointsInCache += s->GetNumPatchPoints();
            } else {
                delete s;
            }
        }
    }

    if (cachePatchPoints) {
        _points.resize(numPointsInCache * 3);
        for (int face = 0; face < numFaces; ++face) {
            float * patchPoints = getPatchPoints(face);
            if (patchPoints) {
                GetSurface(face)->PreparePatchPoints(meshPoints.data(), 3,
                                                     patchPoints, 3);
            }
        }
    }
}

//
//  The main tessellation function:  given a mesh and vertex positions,
//  tessellate each face -- writing results in Obj format.
//
void
tessellateToObj(Far::TopologyRefiner const & meshTopology,
                std::vector<float>   const & meshVertexPositions,
                Args                 const & options) {

    //
    //  Use simpler local type names for the Surface and its factory:
    //
    typedef Bfr::RefinerSurfaceFactory<> SurfaceFactory;
    typedef Bfr::Surface<float>          Surface;

    //
    //  Initialize the SurfaceFactory for the given base mesh (very low
    //  cost in terms of both time and space) and tessellate each face
    //  independently (i.e. no shared vertices):
    //
    //  Note that the SurfaceFactory is not thread-safe by default due to
    //  use of an internal cache.  Creating a separate instance of the
    //  SurfaceFactory for each thread is one way to safely parallelize
    //  this loop.  Another (preferred) is to assign a thread-safe cache
    //  to the single instance.
    //
    //  First declare any evaluation options when initializing (though
    //  none are used in this simple case):
    //
    SurfaceFactory::Options surfaceOptions;

    SurfaceFactory meshSurfaceFactory(meshTopology, surfaceOptions);

    //
    //  Initialize a SurfaceCache to construct Surfaces for all faces.
    //  From this point forward the SurfaceFactory is no longer used to
    //  access Surfaces. Note also that usage below is specific to the
    //  options used to initialize the SurfaceCache:
    //
    bool cachePatchPoints = true;
    bool cacheAllSurfaces = true;
    SurfaceCache surfaceCache(meshSurfaceFactory, meshVertexPositions,
                              cachePatchPoints, cacheAllSurfaces);

    //
    //  As with previous tutorials, output data associated with the face
    //  can be declared in the scope local to each face. But since dynamic
    //  memory is involved with these variables, it is preferred to declare
    //  them outside that loop to preserve and reuse that dynamic memory.
    //
    std::vector<float> outCoords;
    std::vector<float> outPos, outDu, outDv;
    std::vector<int>   outFacets;

    //
    //  Assign Tessellation Options applied for all faces.  Tessellations
    //  allow the creating of either 3- or 4-sided faces -- both of which
    //  are supported here via a command line option:
    //
    int const tessFacetSize = 3 + options.tessQuadsFlag;

    Bfr::Tessellation::Options tessOptions;
    tessOptions.SetFacetSize(tessFacetSize);
    tessOptions.PreserveQuads(options.tessQuadsFlag);

    //
    //  Process each face, writing the output of each in Obj format:
    //
    tutorial::ObjWriter objWriter(options.outputObjFile);

    int numFaces = meshSurfaceFactory.GetNumFaces();
    for (int faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
        //
        //  Retrieve the Surface for this face when present:
        //
        if (!surfaceCache.FaceHasLimitSurface(faceIndex)) continue;

        Surface const & faceSurface = * surfaceCache.GetSurface(faceIndex);

        //
        //  Declare a simple uniform Tessellation for the Parameterization
        //  of this face and identify coordinates of the points to evaluate:
        //
        Bfr::Tessellation tessPattern(faceSurface.GetParameterization(),
                                      options.tessUniformRate, tessOptions);

        int numOutCoords = tessPattern.GetNumCoords();

        outCoords.resize(numOutCoords * 2);

        tessPattern.GetCoords(outCoords.data());

        //
        //  Retrieve the patch points for the Surface, then use them to
        //  evaluate output points for all identified coordinates:
        //
        float const * facePatchPoints = surfaceCache.GetPatchPoints(faceIndex);

        int pointSize = 3;

        outPos.resize(numOutCoords * pointSize);
        outDu.resize(numOutCoords * pointSize);
        outDv.resize(numOutCoords * pointSize);

        for (int i = 0, j = 0; i < numOutCoords; ++i, j += pointSize) {
            faceSurface.Evaluate(&outCoords[i*2],
                                 facePatchPoints, pointSize,
                                 &outPos[j], &outDu[j], &outDv[j]);
        }

        //
        //  Identify the faces of the Tessellation:
        //
        //  Note the need to offset vertex indices for the output faces --
        //  using the number of vertices generated prior to this face. One
        //  of several Tessellation methods to transform the facet indices
        //  simply translates all indices by the desired offset.
        //
        int objVertexIndexOffset = objWriter.GetNumVertices();

        int numFacets = tessPattern.GetNumFacets();
        outFacets.resize(numFacets * tessFacetSize);
        tessPattern.GetFacets(outFacets.data());

        tessPattern.TransformFacetCoordIndices(outFacets.data(),
                                               objVertexIndexOffset);

        //
        //  Write the evaluated points and faces connecting them as Obj:
        //
        objWriter.WriteGroupName("baseFace_", faceIndex);

        objWriter.WriteVertexPositions(outPos);
        objWriter.WriteVertexNormals(outDu, outDv);

        objWriter.WriteFaces(outFacets, tessFacetSize, true, false);
    }
}

//
//  Load command line arguments, specified or default geometry and process:
//
int
main(int argc, char * argv[]) {

    Args args(argc, argv);

    Far::TopologyRefiner * meshTopology = 0;
    std::vector<float>     meshVtxPositions;
    std::vector<float>     meshFVarUVs;

    meshTopology = tutorial::createTopologyRefiner(
            args.inputObjFile, args.schemeType, meshVtxPositions, meshFVarUVs);
    if (meshTopology == 0) {
        return EXIT_FAILURE;
    }

    tessellateToObj(*meshTopology, meshVtxPositions, args);

    delete meshTopology;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
