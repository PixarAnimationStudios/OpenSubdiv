//
//   Copyright 2020 DreamWorks Animation LLC.
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
//      This tutorial shows how to use a Far::LimitStenciTable to repeatedly
//      and efficiently evaluate a set of points (and optionally derivatives)
//      on the limit surface.
//
//      A LimitStencilTable derives from StencilTable but is specialized to
//      factor the evaluation of limit positions and derivatives into stencils.
//      This allows a set of limit properties to be efficiently recomputed in
//      response to changes to the vertices of the base mesh.  Constructing
//      the different kinds of StencilTables can have a high cost, so whether
//      that cost is worth it will depend on your usage (e.g. if points are
//      only computed once, using stencil tables is typically not worth the
//      added cost).
//
//      Any points on the limit surface can be identified for evaluation. In
//      this example we create a crude tessellation similar to tutorial_5_2.
//      The midpoint of each face and points near the corners of the face are
//      evaluated and a triangle fan connects them.
//

#include "../../../regression/common/arg_utils.h"
#include "../../../regression/common/far_utils.h"

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/stencilTableFactory.h>
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
    //  Functions to populate the topology and geometry arrays a simple
    //  shape whose positions may be transformed:
    //
    void
    createCube(std::vector<int> &   vertsPerFace,
               std::vector<Index> & faceVertsPerFace,
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

        //  Initialize verts-per-face and face-vertices for each face:
        vertsPerFace.resize(6);
        faceVertsPerFace.resize(24);
        for (int i = 0; i < 6; ++i) {
            vertsPerFace[i] = 4;
            for (int j = 0; j < 4; ++j) {
                faceVertsPerFace[i*4+j] = cubeFaceVerts[i][j];
            }
        }

        //  Initialize vertex positions:
        positionsPerVert.resize(8);
        for (int i = 0; i < 8; ++i) {
            float const * p = cubePositions[i];
            positionsPerVert[i] = Pos(p[0], p[1], p[2]);
        }
    }

    //
    //  Create a TopologyRefiner from default geometry created above:
    //
    Far::TopologyRefiner *
    createTopologyRefinerDefault(PosVector & posVector) {

        std::vector<int>   topVertsPerFace;
        std::vector<Index> topFaceVerts;

        createCube(topVertsPerFace, topFaceVerts, posVector);

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
                Far::TopologyRefinerFactory<Descriptor>::Options(type,options));
        assert(refiner);
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
                fprintf(stderr,
                    "Error:  Cannot create Shape from .obj file '%s'\n",
                    filename);
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


    //
    //  Simple function to export an Obj file for the limit points -- which
    //  provides a simple tessllation similar to tutorial_5_2.
    //
    int writeToObj(
        Far::TopologyLevel const & baseLevel,
        std::vector<Pos> const & vertexPositions,
        int nextObjVertexIndex) {

        for (size_t i = 0; i < vertexPositions.size(); ++i) {
            float const * p = vertexPositions[i].p;
            printf("v %f %f %f\n", p[0], p[1], p[2]);
        }

        //
        //  Connect the sequences of limit points (center followed by corners)
        //  into triangle fans for each base face:
        //
        for (int i = 0; i < baseLevel.GetNumFaces(); ++i) {
            int faceSize = baseLevel.GetFaceVertices(i).size();

            int vCenter = nextObjVertexIndex + 1;
            int vCorner = vCenter + 1;
            for (int k = 0; k < faceSize; ++k) {
                printf("f %d %d %d\n",
                    vCenter, vCorner + k, vCorner + ((k + 1) % faceSize));
            }
            nextObjVertexIndex += faceSize + 1;
        }
        return nextObjVertexIndex;
    }
} // end namespace


//
//  Command line arguments parsed to provide run-time options:
//
class Args {
public:
    std::string     inputObjFile;
    Sdc::SchemeType schemeType;
    int             maxPatchDepth;
    int             numPoses;
    Pos             poseOffset;
    bool            deriv1Flag;
    bool            noPatchesFlag;
    bool            noOutputFlag;

public:
    Args(int argc, char ** argv) :
        inputObjFile(),
        schemeType(Sdc::SCHEME_CATMARK),
        maxPatchDepth(3),
        numPoses(0),
        poseOffset(1.0f, 0.0f, 0.0f),
        deriv1Flag(false),
        noPatchesFlag(false),
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
            if (!strcmp(rargs[i], "-d1")) {
                deriv1Flag = true;
            } else if (!strcmp(rargs[i], "-nopatches")) {
                noPatchesFlag = true;
            } else if (!strcmp(rargs[i], "-poses")) {
                if (++i < rargs.size()) numPoses = atoi(rargs[i]);
            } else if (!strcmp(rargs[i], "-offset")) {
                if (++i < rargs.size()) poseOffset.p[0] = (float)atof(rargs[i]);
                if (++i < rargs.size()) poseOffset.p[1] = (float)atof(rargs[i]);
                if (++i < rargs.size()) poseOffset.p[2] = (float)atof(rargs[i]);
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
//  Assemble the set of locations for the limit points.  The resulting
//  vector of LocationArrays can contain arbitrary locations on the limit
//  surface -- with multiple locations for the same patch grouped into a
//  single array.
//
//  In this case, for each base face, coordinates for the center and its
//  corners are specified -- from which we will construct a triangle fan
//  providing a crude tessellation (similar to tutorial_5_2).
//
typedef Far::LimitStencilTableFactory::LocationArray   LocationArray;

int assembleLimitPointLocations(Far::TopologyRefiner const & refiner,
                                std::vector<LocationArray> & locations) {
    //
    //  Coordinates for the center of the face and its corners (slightly
    //  inset).  Unlike most of the public interface for patches, the
    //  LocationArray refers to parameteric coordinates as (s,t), so that
    //  convention will be followed here.
    //
    //  Note that the (s,t) coordinates in a LocationArray are referred to
    //  by reference.  The memory holding these (s,t) values must persist
    //  while the LimitStencilTable is constructed -- the arrays here are
    //  declared as static for that purpose.
    //
    static float const quadSCoords[5] = { 0.5f, 0.05f, 0.95f, 0.95f, 0.05f };
    static float const quadTCoords[5] = { 0.5f, 0.05f, 0.05f, 0.95f, 0.95f };

    static float const triSCoords[4] = { 0.33f, 0.05f, 0.95f, 0.05f };
    static float const triTCoords[4] = { 0.33f, 0.05f, 0.00f, 0.95f };

    static float const irregSCoords[2] = { 1.0f, 0.05f };
    static float const irregTCoords[2] = { 1.0f, 0.05f };

    //
    //  Since these are references to patches to be evaluated, we require
    //  use of the Ptex indices to identify the top-most parameterized
    //  patch, which is essential to dealing with non-quad faces (in the
    //  case of Catmark).
    //
    Far::TopologyLevel const & baseLevel = refiner.GetLevel(0);

    Far::PtexIndices basePtexIndices(refiner);

    int regFaceSize = Sdc::SchemeTypeTraits::GetRegularFaceSize(
        refiner.GetSchemeType());


    //
    //  For each base face, simply refer to the (s,t) arrays for regular quad
    //  and triangular patches with a single LocationArray.  Otherwise, for
    //  irregular faces, the corners of the face come from different patches
    //  and so must be referenced in separate LocationArrays.
    //
    locations.clear();

    int numLimitPoints = 0;
    for (int i = 0; i < baseLevel.GetNumFaces(); ++i) {
        int baseFaceSize = baseLevel.GetFaceVertices(i).size();
        int basePtexId   = basePtexIndices.GetFaceId(i); 

        bool faceIsRegular = (baseFaceSize == regFaceSize);
        if (faceIsRegular) {
            //  All coordinates are on the same top-level patch:
            LocationArray loc;
            loc.ptexIdx = basePtexId;
            loc.numLocations = baseFaceSize + 1;
            if (baseFaceSize == 4) {
                loc.s = quadSCoords;
                loc.t = quadTCoords;
            } else {
                loc.s = triSCoords;
                loc.t = triTCoords;
            }
            locations.push_back(loc);
        } else {
            //  Center coordinate is on the first sub-patch while those on
            //  near the corners are on each successive sub-patch:
            LocationArray loc;
            loc.numLocations = 1;
            for (int j = 0; j <= baseFaceSize; ++j) {
                bool isPerimeter = (j > 0);
                loc.ptexIdx = basePtexId + (isPerimeter ? (j-1) : 0);
                loc.s = &irregSCoords[isPerimeter];
                loc.t = &irregTCoords[isPerimeter];

                locations.push_back(loc);
            }
        }
        numLimitPoints += baseFaceSize + 1;
    }
    return numLimitPoints;
}


//
//  Load command line arguments and geometry, build the LimitStencilTable
//  for a set of points on the limit surface and compute those points for
//  several orientations of the mesh:
//
int
main(int argc, char **argv) {

    Args args(argc, argv);

    //
    //  Create or load the base geometry (command line arguments allow a
    //  .obj file to be specified), providing a TopologyRefiner and a set
    //  of base vertex positions to work with:
    //
    std::vector<Pos> basePositions;

    Far::TopologyRefiner * refinerPtr = args.inputObjFile.empty() ?
            createTopologyRefinerDefault(basePositions) :
            createTopologyRefinerFromObj(args.inputObjFile, args.schemeType,
                                         basePositions);
    assert(refinerPtr);
    Far::TopologyRefiner & refiner = *refinerPtr;

    Far::TopologyLevel const & baseLevel = refiner.GetLevel(0);

    //
    //  Use of LimitStencilTable requires either explicit or implicit use
    //  of a PatchTable.  A PatchTable is not required to construct a
    //  LimitStencilTable -- one will be constructed internally for use
    //  and discarded -- but explicit construction is recommended to control
    //  the many legacy options for PatchTable, rather than relying on
    //  internal defaults.  Adaptive refinement is required in both cases
    //  to indicate the accuracy of the patches.
    //
    //  Note that if a TopologyRefiner and PatchTable are not used for
    //  any other purpose than computing the limit points, that specifying
    //  the subset of faces containing those limit points in the adaptive
    //  refinement and PatchTable construction can avoid unnecessary
    //  overhead.
    //
    Far::PatchTable * patchTablePtr = 0;

    if (args.noPatchesFlag) {
        refiner.RefineAdaptive(
            Far::TopologyRefiner::AdaptiveOptions(args.maxPatchDepth));
    } else {
        Far::PatchTableFactory::Options patchOptions(args.maxPatchDepth);
        patchOptions.useInfSharpPatch = true;
        patchOptions.generateLegacySharpCornerPatches = false;
        patchOptions.generateVaryingTables = false;
        patchOptions.generateFVarTables = false;
        patchOptions.endCapType =
            Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;

        refiner.RefineAdaptive(patchOptions.GetRefineAdaptiveOptions());

        patchTablePtr = Far::PatchTableFactory::Create(refiner, patchOptions);
        assert(patchTablePtr);
    }

    //
    //  Assemble the set of locations for the limit points.  For each base
    //  face, coordinates for the center and its corners are specified --
    //  from which we will construct a triangle fan providing a crude
    //  tessellation (similar to tutorial_5_2).
    //
    std::vector<LocationArray> locations;

    int numLimitPoints = assembleLimitPointLocations(refiner, locations);

    //
    //  Construct a LimitStencilTable from the refiner, patch table (optional)
    //  and the collection of limit point locations.  Stencils can optionally
    //  be created for computing dervatives -- the default is to compute 1st
    //  derivative stencils, so be sure to disable that if not necessary:
    //
    Far::LimitStencilTableFactory::Options limitOptions;
    limitOptions.generate1stDerivatives = args.deriv1Flag;

    Far::LimitStencilTable const * limitStencilTablePtr =
        Far::LimitStencilTableFactory::Create(refiner, locations,
            0,             // optional StencilTable for the refined points
            patchTablePtr, // optional PatchTable
            limitOptions);
    assert(limitStencilTablePtr);
    Far::LimitStencilTable const & limitStencilTable = *limitStencilTablePtr;

    //
    //  Apply the constructed LimitStencilTable to compute limit positions
    //  from the base level vertex positions.  This is trivial if computing
    //  all positions in one invokation.  The UpdateValues method (and those
    //  for derivatives) are overloaded to optionally accept a subrange of
    //  indices to distribute the computation:
    //
    std::vector<Pos> limitPositions(numLimitPoints);

    limitStencilTable.UpdateValues(basePositions, limitPositions);

    //  Call with the optional subrange:
    limitStencilTable.UpdateValues(basePositions, limitPositions,
                                   0, numLimitPoints / 2);
    limitStencilTable.UpdateValues(basePositions, limitPositions,
                                   (numLimitPoints / 2) + 1, numLimitPoints);

    // Write vertices and faces in Obj format for the original limit points:
    int objVertCount = 0;

    if (!args.noOutputFlag) {
        printf("g base_mesh\n");
        objVertCount = writeToObj(baseLevel, limitPositions, objVertCount);
    }

    //
    //  Recompute the limit points and output faces for different "poses" of
    //  the original mesh -- in this case simply translated.  Also optionally
    //  compute 1st derivatives (though they are not used here):
    //
    std::vector<Pos> posePositions(basePositions);

    std::vector<Pos> limitDu(args.deriv1Flag ? numLimitPoints : 0);
    std::vector<Pos> limitDv(args.deriv1Flag ? numLimitPoints : 0);

    for (int i = 0; i < args.numPoses; ++i) {
        // Trivially transform the base vertex positions and re-compute:
        for (size_t j = 0; j < basePositions.size(); ++j) {
            posePositions[j] = posePositions[j] + args.poseOffset;
        }

        limitStencilTable.UpdateValues(posePositions, limitPositions);
        if (args.deriv1Flag) {
            limitStencilTable.UpdateDerivs(posePositions, limitDu, limitDv);
        }

        if (!args.noOutputFlag) {
            printf("\ng pose_%d\n", i);
            objVertCount = writeToObj(baseLevel, limitPositions, objVertCount);
        }
    }
    delete refinerPtr;
    delete patchTablePtr;
    delete limitStencilTablePtr;

    return EXIT_SUCCESS;
}
