//
//   Copyright 2018 DreamWorks Animation LLC.
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
//      This tutorial shows how to manage the limit surface of a potentially
//      large mesh by creating groups of patches for selected faces of the
//      mesh.  Familiarity with construction and evaluation of a PatchTable
//      is assumed (see tutorial_5_1).
//
//      When the patches for a mesh do not need to be retained for further
//      use, e.g. when simply computing points for a tessellation, the time
//      and space required to construct a single large PatchTable can be
//      considerable.  By constructing, evaluating and discarding smaller
//      PatchTables for subsets of the mesh, the high transient memory cost
//      can be avoided when computed serially.  When computed in parallel,
//      there may be little memory savings, but the construction time can
//      then be distributed.
//
//      This tutorial creates simple geometry (currently a lattice of cubes)
//      that can be expanded in complexity with a simple multiplier.  The
//      collection of faces are then divided into a specified number of groups
//      from which patches will be constructed and evaluated.  A simple
//      tessellation (a triangle fan around the midpoint of each face) is then
//      written in Obj format to the standard output.
//

#include "../../../regression/common/arg_utils.h"
#include "../../../regression/common/far_utils.h"

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/ptexIndices.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace OpenSubdiv;

using Far::Index;


//
//  Global utilities in this namespace are not relevant to the tutorial.
//  They simply serve to construct some default geometry to be processed
//  in the form of a TopologyRefiner and vector of vertex positions.
//
namespace {
    //
    //  Simple structs for (x,y,z) position and a 3-tuple for the set
    //  of vertices of a triangle:
    //
    struct Pos {
        Pos() { }
        Pos(float x, float y, float z) { p[0] = x, p[1] = y, p[2] = z; }

        Pos operator+(Pos const & op) const {
            return Pos(p[0] + op.p[0], p[1] + op.p[1], p[2] + op.p[2]);
        }

        //  Clear() and AddWithWeight() required for interpolation:
        void Clear( void * =0 ) { p[0] = p[1] = p[2] = 0.0f; }

        void AddWithWeight(Pos const & src, float weight) {
            p[0] += weight * src.p[0];
            p[1] += weight * src.p[1];
            p[2] += weight * src.p[2];
        }

        float p[3];
    };
    typedef std::vector<Pos> PosVector;

    struct Tri {
        Tri() { }
        Tri(int a, int b, int c) { v[0] = a, v[1] = b, v[2] = c; }

        int v[3];
    };
    typedef std::vector<Tri> TriVector;


    //
    //  Functions to populate the topology and geometry arrays with simple
    //  shapes that we can multiply to increase complexity:
    //
    void
    appendDefaultPrimitive(Pos const &          origin,
                           std::vector<int> &   vertsPerFace,
                           std::vector<Index> & faceVerts,
                           std::vector<Pos> &   positionsPerVert) {

        //  Local topology and position of a cube centered at origin:
        static float const cubePositions[8][3] = { { -0.5f, -0.5f, -0.5f },
                                                   { -0.5f,  0.5f, -0.5f },
                                                   { -0.5f,  0.5f,  0.5f },
                                                   { -0.5f, -0.5f,  0.5f },
                                                   {  0.5f, -0.5f, -0.5f },
                                                   {  0.5f,  0.5f, -0.5f },
                                                   {  0.5f,  0.5f,  0.5f },
                                                   {  0.5f, -0.5f,  0.5f } };

        static int const cubeFaceVerts[6][4] = { { 0, 3, 2, 1 },
                                                 { 4, 5, 6, 7 },
                                                 { 0, 4, 7, 3 },
                                                 { 1, 2, 6, 5 },
                                                 { 0, 1, 5, 4 },
                                                 { 3, 7, 6, 2 } };

        //  Identify the next vertex before appending vertex positions:
        int baseVertex = (int) positionsPerVert.size();

        for (int i = 0; i < 8; ++i) {
            float const * p = cubePositions[i];
            positionsPerVert.push_back(origin + Pos(p[0], p[1], p[2]));
        }

        //  Append number of verts-per-face and face-vertices for each face:
        for (int i = 0; i < 6; ++i) {
            vertsPerFace.push_back(4);
            for (int j = 0; j < 4; ++j) {
                faceVerts.push_back(baseVertex + cubeFaceVerts[i][j]);
            }
        }
    }

    void
    createDefaultGeometry(int multiplier,
                          std::vector<int> &   vertsPerFace,
                          std::vector<Index> & faceVerts,
                          std::vector<Pos> &   positionsPerVert) {

        //  Default primitive is currently a cube:
        int const vertsPerPrimitive = 8;
        int const facesPerPrimitive = 6;
        int const faceVertsPerPrimitive = 24;

        int nPrimitives = multiplier * multiplier * multiplier;

        positionsPerVert.reserve(nPrimitives * vertsPerPrimitive);
        vertsPerFace.reserve(nPrimitives * facesPerPrimitive);
        faceVerts.reserve(nPrimitives * faceVertsPerPrimitive);

        for (int x = 0; x < multiplier; ++x) {
            for (int y = 0; y < multiplier; ++y) {
                for (int z = 0; z < multiplier; ++z) {
                    appendDefaultPrimitive(
                        Pos((float)x * 2.0f, (float)y * 2.0f, (float)z * 2.0f),
                        vertsPerFace, faceVerts, positionsPerVert);
                }
            }
        }
    }

    //
    //  Create a TopologyRefiner from default geometry created above:
    //
    Far::TopologyRefiner *
    createTopologyRefinerDefault(int multiplier,
                                 PosVector & posVector) {

        std::vector<int>   topVertsPerFace;
        std::vector<Index> topFaceVerts;

        createDefaultGeometry(
            multiplier, topVertsPerFace, topFaceVerts, posVector);

        typedef Far::TopologyDescriptor Descriptor;

        Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

        Sdc::Options options;
        options.SetVtxBoundaryInterpolation(
            Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);

        Descriptor desc;
        desc.numVertices = (int) posVector.size();
        desc.numFaces = (int) topVertsPerFace.size();
        desc.numVertsPerFace = &topVertsPerFace[0];
        desc.vertIndicesPerFace = &topFaceVerts[0];

        //  Instantiate a Far::TopologyRefiner from the descriptor.
        Far::TopologyRefiner * refiner =
            Far::TopologyRefinerFactory<Descriptor>::Create(desc,
                Far::TopologyRefinerFactory<Descriptor>::Options(
                    type, options));

        if (refiner == 0) {
            exit(EXIT_FAILURE);
        }

        bool dumpDefaultGeometryToObj = false;
        if (dumpDefaultGeometryToObj) {
            int nVerts = (int) posVector.size();
            for (int i = 0; i < nVerts; ++i) {
                float const * p = posVector[i].p;
                printf("v %f %f %f\n", p[0], p[1], p[2]);
            }

            int const * fVerts = &topFaceVerts[0];
            int nFaces = (int) topVertsPerFace.size();
            for (int i = 0; i < nFaces; ++i) {
                printf("f");
                for (int j = 0; j < topVertsPerFace[i]; ++j) {
                    printf(" %d", 1 + *fVerts++);
                }
                printf("\n");
            }
            exit(EXIT_SUCCESS);
        }
        return refiner;
    }

    //
    //  Create a TopologyRefiner from a specified Obj file:
    //  geometry created internally:
    //
    Far::TopologyRefiner *
    createTopologyRefinerFromObj(std::string const & objFileName,
                                 Sdc::SchemeType schemeType,
                                 PosVector & posVector) {

        const char *  filename = objFileName.c_str();
        const Shape * shape = 0;

        std::ifstream ifs(filename);
        if (ifs) {
            std::stringstream ss;
            ss << ifs.rdbuf();
            ifs.close();
            std::string shapeString = ss.str();

            shape = Shape::parseObj(shapeString.c_str(),
                ConvertSdcTypeToShapeScheme(schemeType), false);
            if (shape == 0) {
                fprintf(stderr, "Error:  Cannot create Shape "
                    "from .obj file '%s'\n", filename);
                return 0;
            }
        } else {
            fprintf(stderr, "Error:  Cannot open .obj file '%s'\n", filename);
            return 0;
        }

        Sdc::SchemeType sdcType    = GetSdcType(*shape);
        Sdc::Options    sdcOptions = GetSdcOptions(*shape);

        Far::TopologyRefiner * refiner =
            Far::TopologyRefinerFactory<Shape>::Create(*shape,
                Far::TopologyRefinerFactory<Shape>::Options(
                    sdcType, sdcOptions));
        if (refiner == 0) {
            fprintf(stderr, "Error:  Unable to construct TopologyRefiner "
                "from .obj file '%s'\n", filename);
            return 0;
        }

        int numVertices = refiner->GetNumVerticesTotal();
        posVector.resize(numVertices);
        std::memcpy(&posVector[0].p[0], &shape->verts[0],
                    numVertices * 3 * sizeof(float));

        delete shape;
        return refiner;
    }
} // end namespace


//
//  The PatchGroup bundles objects used to create and evaluate a sparse set
//  of patches.  Its construction creates a PatchTable and all other objects
//  necessary to evaluate patches associated with the specified subset of
//  faces provided.  A simple method to tessellate a specified face is
//  provided.
//
//  Note that, since the data buffers for the base level and refined levels
//  are separate (we want to avoid copying primvar data for the base level
//  of a potentially large mesh), that patch evaluation needs to account
//  for the separation when combining control points.
//
struct PatchGroup {
    PatchGroup(Far::PatchTableFactory::Options patchOptions,
               Far::TopologyRefiner const &    baseRefinerArg,
               Far::PtexIndices const &        basePtexIndicesArg,
               std::vector<Pos> const &        basePositionsArg,
               std::vector<Index> const &      baseFacesArg);
    ~PatchGroup();

    void TessellateBaseFace(int face, PosVector & tessPoints,
                                      TriVector & tessTris) const;

    //  Const reference members:
    Far::TopologyRefiner const & baseRefiner;
    Far::PtexIndices const &     basePtexIndices;
    std::vector<Pos> const &     basePositions;
    std::vector<Index> const &   baseFaces;

    //  Members constructed to evaluate patches:
    Far::PatchTable *   patchTable;
    Far::PatchMap *     patchMap;
    int                 patchFaceSize;
    std::vector<Pos>    localPositions;
};

PatchGroup::PatchGroup(Far::PatchTableFactory::Options patchOptions,
                       Far::TopologyRefiner const &    baseRefinerArg,
                       Far::PtexIndices const &        basePtexIndicesArg,
                       std::vector<Pos> const &        basePositionsArg,
                       std::vector<Index> const &      baseFacesArg) :
        baseRefiner(baseRefinerArg), 
        basePtexIndices(basePtexIndicesArg), 
        basePositions(basePositionsArg), 
        baseFaces(baseFacesArg) {

    //  Create a local refiner (sharing the base level), apply adaptive
    //  refinement to the given subset of base faces, and construct a patch
    //  table (and its associated map) for the same set of faces:
    //
    Far::ConstIndexArray groupFaces(&baseFaces[0], (int)baseFaces.size());

    Far::TopologyRefiner *localRefiner =
        Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(
            baseRefiner);

    localRefiner->RefineAdaptive(
        patchOptions.GetRefineAdaptiveOptions(), groupFaces);

    patchTable = Far::PatchTableFactory::Create(*localRefiner, patchOptions,
        groupFaces);

    patchMap = new Far::PatchMap(*patchTable);

    patchFaceSize =
        Sdc::SchemeTypeTraits::GetRegularFaceSize(baseRefiner.GetSchemeType());

    //  Compute the number of refined and local points needed to evaluate the
    //  patches, allocate and interpolate.  This varies from tutorial_5_1 in
    //  that the primvar buffer for the base vertices is separate from the
    //  refined vertices and local patch points (which must also be accounted
    //  for when evaluating the patches).
    //
    int nBaseVertices    = localRefiner->GetLevel(0).GetNumVertices();
    int nRefinedVertices = localRefiner->GetNumVerticesTotal() - nBaseVertices;
    int nLocalPoints     = patchTable->GetNumLocalPoints();

    localPositions.resize(nRefinedVertices + nLocalPoints);

    if (nRefinedVertices) {
        Far::PrimvarRefiner primvarRefiner(*localRefiner);

        Pos const * src = &basePositions[0];
        Pos * dst = &localPositions[0];
        for (int level = 1; level < localRefiner->GetNumLevels(); ++level) {
            primvarRefiner.Interpolate(level, src, dst);
            src = dst;
            dst += localRefiner->GetLevel(level).GetNumVertices();
        }
    }
    if (nLocalPoints) {
        patchTable->GetLocalPointStencilTable()->UpdateValues(
                &basePositions[0], nBaseVertices, &localPositions[0],
                &localPositions[nRefinedVertices]);
    }

    delete localRefiner;
}

PatchGroup::~PatchGroup() {
    delete patchTable;
    delete patchMap;
}

void
PatchGroup::TessellateBaseFace(int face, PosVector & tessPoints,
                                         TriVector & tessTris) const {

    //  Tesselate the face with points at the midpoint of the face and at
    //  each corner, and triangles connecting the midpoint to each edge.
    //  Irregular faces require an aribrary number of corners points, but
    //  all are at the origin of the child face of the irregular base face:
    //
    float const quadPoints[5][2] = { { 0.5f, 0.5f },
                                     { 0.0f, 0.0f },
                                     { 1.0f, 0.0f },
                                     { 1.0f, 1.0f },
                                     { 0.0f, 1.0f } };

    float const triPoints[4][2] = { { 0.5f, 0.5f },
                                    { 0.0f, 0.0f },
                                    { 1.0f, 0.0f },
                                    { 0.0f, 1.0f } };

    float const irregPoints[4][2] = { { 1.0f, 1.0f },
                                      { 0.0f, 0.0f } };

    //  Determine the topology of the given base face and the resulting
    //  tessellation points and faces to generate:
    //
    int baseFace = baseFaces[face];
    int faceSize = baseRefiner.GetLevel(0).GetFaceVertices(baseFace).size();

    bool faceIsIrregular = (faceSize != patchFaceSize);

    int nTessPoints = faceSize + 1;
    int nTessFaces  = faceSize;

    tessPoints.resize(nTessPoints);
    tessTris.resize(nTessFaces);

    //  Compute the mid and corner points -- remember that for an irregular
    //  face, we must reference the individual ptex faces for each corner:
    //
    int ptexFace = basePtexIndices.GetFaceId(baseFace);

    int numBaseVerts = (int) basePositions.size();

    for (int i = 0; i < nTessPoints; ++i) {
        //  Choose the (s,t) coordinate from the fixed tessellation:
        float const * st = faceIsIrregular ? irregPoints[i != 0]
                         : ((faceSize == 4) ? quadPoints[i] : triPoints[i]);

        //  Locate the patch corresponding to the face ptex idx and (s,t)
        //  and evaluate:
        int patchFace = ptexFace;
        if (faceIsIrregular && (i > 0)) {
            patchFace += i - 1;
        }
        Far::PatchTable::PatchHandle const * handle =
            patchMap->FindPatch(patchFace, st[0], st[1]);
        assert(handle);

        float pWeights[20];
        patchTable->EvaluateBasis(*handle, st[0], st[1], pWeights);

        //  Identify the patch cvs and combine with the evaluated weights --
        //  remember to distinguish cvs in the base level:
        Far::ConstIndexArray cvIndices = patchTable->GetPatchVertices(*handle);

        Pos & pos = tessPoints[i];
        pos.Clear();
        for (int cv = 0; cv < cvIndices.size(); ++cv) {
            int cvIndex = cvIndices[cv];
            if (cvIndex < numBaseVerts) {
                pos.AddWithWeight(basePositions[cvIndex],
                    pWeights[cv]);
            } else {
                pos.AddWithWeight(localPositions[cvIndex - numBaseVerts],
                    pWeights[cv]);
            }
        }
    }

    //  Assign triangles connecting the midpoint of the base face to the
    //  points computed at the ends of each of its edges:
    //
    for (int i = 0; i < nTessFaces; ++i) {
        tessTris[i] = Tri(0, 1 + i, 1 + ((i + 1) % faceSize));
    }
}


//
//  Command line arguments parsed to provide run-time options:
//
class Args {
public:
    std::string     inputObjFile;
    Sdc::SchemeType schemeType;
    int             geoMultiplier;
    int             maxPatchDepth;
    int             numPatchGroups;
    bool            noTessFlag;
    bool            noOutputFlag;

public:
    Args(int argc, char ** argv) :
        inputObjFile(),
        schemeType(Sdc::SCHEME_CATMARK),
        geoMultiplier(10),
        maxPatchDepth(3),
        numPatchGroups(10),
        noTessFlag(false),
        noOutputFlag(false) {

        //  Parse and assign standard arguments and Obj files:
        ArgOptions args;
        args.Parse(argc, argv);

        maxPatchDepth = args.GetLevel();
        schemeType = ConvertShapeSchemeToSdcType(args.GetDefaultScheme());

        const std::vector<const char *> objFiles = args.GetObjFiles();
        if (!objFiles.empty()) {
            for (size_t i = 1; i < objFiles.size(); ++i) {
                fprintf(stderr,
                    "Warning: .obj file '%s' ignored\n", objFiles[i]);
            }
            inputObjFile = std::string(objFiles[0]);
        }

        //  Parse remaining arguments specific to this example:
        const std::vector<const char *> &rargs = args.GetRemainingArgs();
        for (size_t i = 0; i < rargs.size(); ++i) {
            if (!strcmp(rargs[i], "-groups")) {
                if (++i < rargs.size()) numPatchGroups = atoi(rargs[i]);
            } else if (!strcmp(rargs[i], "-mult")) {
                if (++i < rargs.size()) geoMultiplier = atoi(rargs[i]);
            } else if (!strcmp(rargs[i], "-notess")) {
                noTessFlag = true;
            } else if (!strcmp(rargs[i], "-nooutput")) {
                noOutputFlag = true;
            } else {
                fprintf(stderr, "Warning: Argument '%s' ignored\n", rargs[i]);
            }
        }
    }

private:
    Args() { }
};


//
//  Load command line arguments and geometry, then divide the mesh into groups
//  of faces from which to create and tessellate patches:
//
int
main(int argc, char **argv) {

    Args args(argc, argv);

    //
    //  Create or load the base geometry (command line arguments allow a
    //  .obj file to be specified).  In addition to the TopologyRefiner
    //  and set of positions for the base vertices, a set of PtexIndices is
    //  also required to evaluate patches, so build it here once for use
    //  elsewhere:
    //
    std::vector<Pos> basePositions;

    Far::TopologyRefiner * baseRefinerPtr = args.inputObjFile.empty() ?
        createTopologyRefinerDefault(args.geoMultiplier, basePositions) :
        createTopologyRefinerFromObj(args.inputObjFile, args.schemeType,
            basePositions);
    assert(baseRefinerPtr);
    Far::TopologyRefiner & baseRefiner = *baseRefinerPtr;

    Far::PtexIndices basePtexIndices(baseRefiner);

    //
    //  Determine the sizes of the patch groups specified -- there will be
    //  two sizes that differ by one to account for unequal division:
    //
    int numBaseFaces = baseRefiner.GetNumFacesTotal();

    int numPatchGroups = args.numPatchGroups;
    if (numPatchGroups > numBaseFaces) {
        numPatchGroups = numBaseFaces;
    } else if (numPatchGroups < 1) {
        numPatchGroups = 1;
    }
    int lesserGroupSize = numBaseFaces / numPatchGroups;
    int numLargerGroups = numBaseFaces - (numPatchGroups * lesserGroupSize);

    //
    //  Define the options used to construct the patches for each group.
    //  Unless suppressed, a tessellation in Obj format will also be printed
    //  to standard output, so keep track of the vertex indices.
    //
    Far::PatchTableFactory::Options patchOptions(args.maxPatchDepth);
    patchOptions.generateVaryingTables = false;
    patchOptions.shareEndCapPatchPoints = false;
    patchOptions.endCapType =
        Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;

    int objVertCount = 0;

    PosVector tessPoints;
    TriVector tessFaces;

    for (int i = 0; i < numPatchGroups; ++i) {

        //
        //  Initialize a vector with a group of base faces from which to
        //  create and evaluate patches:
        //
        Index minFace = i * lesserGroupSize + std::min(i, numLargerGroups);
        Index maxFace = minFace + lesserGroupSize + (i < numLargerGroups);

        std::vector<Far::Index> baseFaces(maxFace - minFace);
        for (int face = minFace; face < maxFace; ++face) {
            baseFaces[face - minFace] = face;
        }

        //
        //  Declare a PatchGroup and tessellate its base faces -- generating
        //  vertices and faces in Obj format to standard output:
        //
        PatchGroup patchGroup(patchOptions,
            baseRefiner, basePtexIndices, basePositions, baseFaces);

        if (args.noTessFlag) continue;

        if (!args.noOutputFlag) {
            printf("g patchGroup_%d\n", i);
        }

        for (int j = 0; j < (int) baseFaces.size(); ++j) {
            patchGroup.TessellateBaseFace(j, tessPoints, tessFaces);

            if (!args.noOutputFlag) {
                int nVerts = (int) tessPoints.size();
                for (int k = 0; k < nVerts; ++k) {
                    float const * p = tessPoints[k].p;
                    printf("v %f %f %f\n", p[0], p[1], p[2]);
                }

                int nTris = (int) tessFaces.size();
                int vBase = 1 + objVertCount;
                for (int k = 0; k < nTris; ++k) {
                    int const * v = tessFaces[k].v;
                    printf("f %d %d %d\n",
                        vBase + v[0], vBase + v[1], vBase + v[2]);
                }
                objVertCount += nVerts;
            }
        }
    }
    delete baseRefinerPtr;

    return EXIT_SUCCESS;
}
