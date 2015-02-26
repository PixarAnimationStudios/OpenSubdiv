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

#ifndef FAR_PATCH_TABLES_FACTORY_H
#define FAR_PATCH_TABLES_FACTORY_H

#include "../version.h"

#include "../far/patchTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

//  Forward declarations (for internal implementation purposes):
namespace Vtr { class Level; }

namespace Far {

class StencilTables;
class TopologyRefiner;



/// \brief A specialized factory for feature adaptive PatchTables
///
/// PatchTables contain the lists of vertices for each patch of an adaptive
/// mesh representation.
///
class PatchTablesFactory {

public:

    struct Options {

        Options(unsigned int maxIsolation=10) :
             generateAllLevels(false),
             triangulateQuads(false),
             useSingleCreasePatch(false),
             maxIsolationLevel(maxIsolation),
             generateFVarTables(false),
             useFVarQuadEndCaps(true), // XXXX change to false when FVar Gregory is ready
             numFVarChannels(-1),
             fvarChannelIndices(0),
             adaptiveStencilTables(0) { }

        unsigned int generateAllLevels    : 1, ///< Include levels from 'firstLevel' to 'maxLevel' (Uniform mode only)
                     triangulateQuads     : 1, ///< Triangulate 'QUADS' primitives (Uniform mode only)
                     useSingleCreasePatch : 1, ///< Use single crease patch
                     maxIsolationLevel    : 4, ///< Cap adaptive feature isolation to the given level (max. 10)

                     // face-varying
                     generateFVarTables   : 1, ///< Generate face-varying patch tables
                     useFVarQuadEndCaps   : 1; ///< Use bilinear quads as end-caps around extraordinary vertices

        int          numFVarChannels;          ///< Number of channel indices and interpolation modes passed
        int const *  fvarChannelIndices;       ///< List containing the indices of the channels selected for the factory

        StencilTables const * adaptiveStencilTables;  ///< Passing a valid stencil table allows the factory to generate
                                                      ///< stencils for gregory patches and replace them with a much
                                                      ///< more efficient basis.
    };

    /// \brief Factory constructor for PatchTables
    ///
    /// @param refiner  TopologyRefiner from which to generate patches
    ///
    /// @param options       Options controlling the creation of the tables
    ///
    /// @return              A new instance of PatchTables
    ///
    static PatchTables * Create(TopologyRefiner const & refiner, Options options=Options());

private:

    //
    // Private helper structures
    //

    struct AdaptiveContext;

private:

    static PatchTables * createUniform(TopologyRefiner const & refiner, Options options);

    static PatchTables * createAdaptive(TopologyRefiner const & refiner, Options options);

    //
    //  High-level methods for identifying and populating patches associated with faces:
    //

    static void identifyAdaptivePatches(AdaptiveContext & state);

    static void populateAdaptivePatches(AdaptiveContext & state);

    //
    //  Methods for allocating and managing the patch table data arrays:
    //

    static void allocateVertexTables(PatchTables * tables, int nlevels, bool hasSharpness);

    static void allocateFVarChannels(TopologyRefiner const & refiner,
         Options options, int npatches, PatchTables * tables);

    static PatchParam * computePatchParam(TopologyRefiner const & refiner,
        int level, int face, int rotation, PatchParam * coord);

    static int gatherFVarData(AdaptiveContext & state,
        int level, Index faceIndex, Index levelFaceOffset, int rotation, Index const * levelOffsets, Index fofss, Index ** fptrs);

    static void getQuadOffsets(Vtr::Level const & level, int face, unsigned int * result);

private:
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_PATCH_TABLES_FACTORY_H */
