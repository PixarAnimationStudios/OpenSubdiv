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

#ifndef OPENSUBDIV3_FAR_PATCH_TABLE_FACTORY_H
#define OPENSUBDIV3_FAR_PATCH_TABLE_FACTORY_H

#include "../version.h"

#include "../far/patchTable.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//  Forward declarations (for internal implementation purposes):
class TopologyRefiner;

/// \brief Factory for constructing a PatchTable from a TopologyRefiner
///
class PatchTableFactory {
public:

    /// \brief Public options for the PatchTable factory
    ///
    struct Options {

        enum EndCapType {
            ENDCAP_NONE = 0,             ///< no endcap
            ENDCAP_BILINEAR_BASIS,       ///< use bilinear quads (4 cp) as end-caps
            ENDCAP_BSPLINE_BASIS,        ///< use BSpline basis patches (16 cp) as end-caps
            ENDCAP_GREGORY_BASIS,        ///< use Gregory basis patches (20 cp) as end-caps
            ENDCAP_LEGACY_GREGORY        ///< use legacy (2.x) Gregory patches (4 cp + valence table) as end-caps
        };

        Options(unsigned int maxIsolation=10) :
             generateAllLevels(false),
             triangulateQuads(false),
             useSingleCreasePatch(false),
             useInfSharpPatch(false),
             maxIsolationLevel(maxIsolation),
             endCapType(ENDCAP_GREGORY_BASIS),
             shareEndCapPatchPoints(true),
             generateFVarTables(false),
             generateFVarLegacyLinearPatches(true),
             generateLegacySharpCornerPatches(true),
             numFVarChannels(-1),
             fvarChannelIndices(0)
        { }

        /// \brief Get endcap patch type
        EndCapType GetEndCapType() const { return (EndCapType)endCapType; }

        /// \brief Set endcap patch type
        void SetEndCapType(EndCapType e) { endCapType = e; }

        unsigned int generateAllLevels    : 1, ///< Include levels from 'firstLevel' to 'maxLevel' (Uniform mode only)
                     triangulateQuads     : 1, ///< Triangulate 'QUADS' primitives (Uniform mode only)
                     useSingleCreasePatch : 1, ///< Use single crease patch
                     useInfSharpPatch     : 1, ///< Use infinitely-sharp patch
                     maxIsolationLevel    : 4, ///< Cap adaptive feature isolation to the given level (max. 10)

                     // end-capping
                     endCapType              : 3, ///< EndCapType
                     shareEndCapPatchPoints  : 1, ///< Share endcap patch points among adjacent endcap patches.
                                                  ///< currently only work with GregoryBasis.

                     // face-varying
                     generateFVarTables  : 1, ///< Generate face-varying patch tables

                     // legacy behaviors (default to true)
                     generateFVarLegacyLinearPatches  : 1, ///< Generate all linear face-varying patches (legacy)
                     generateLegacySharpCornerPatches : 1; ///< Generate sharp regular patches at smooth corners (legacy)

        int          numFVarChannels;          ///< Number of channel indices and interpolation modes passed
        int const *  fvarChannelIndices;       ///< List containing the indices of the channels selected for the factory
    };

    /// \brief Instantiates a PatchTable from a client-provided TopologyRefiner.
    ///
    ///  A PatchTable can be constructed from a TopologyRefiner that has been
    ///  either adaptively or uniformly refined.  In both cases, the resulting
    ///  patches reference vertices in the various refined levels by index,
    ///  and those indices accumulate with the levels in different ways.
    ///
    ///  For adaptively refined patches, patches are defined at different levels,
    ///  including the base level, so the indices of patch vertices include
    ///  vertices from all levels.
    ///
    ///  For uniformly refined patches, all patches are completely defined within
    ///  the last level.  There is often no use for intermediate levels and they
    ///  can usually be ignored.  Indices of patch vertices might therefore be
    ///  expected to be defined solely within the last level.  While this is true
    ///  for face-varying patches, for historical reasons it is not the case for
    ///  vertex and varying patches.  Indices for vertex and varying patches include
    ///  the base level in addition to the last level while indices for face-varying
    ///  patches include only the last level.
    ///
    /// @param refiner              TopologyRefiner from which to generate patches
    ///
    /// @param options              Options controlling the creation of the table
    ///
    /// @return                     A new instance of PatchTable
    ///
    static PatchTable * Create(TopologyRefiner const & refiner,
                               Options options=Options());

public:
    //  PatchFaceTag
    //  This simple struct was previously used within the factory to take inventory of
    //  various kinds of patches to fully allocate buffers prior to populating them.  It
    //  was not intended to be exposed as part of the public interface.
    //
    //  It is no longer used internally and is being kept here to respect preservation
    //  of the public interface, but it will be deprecated at the earliest opportunity.
    //
    struct PatchFaceTag {
    public:
        unsigned int   _hasPatch        : 1;
        unsigned int   _isRegular       : 1;
        unsigned int   _transitionMask  : 4;
        unsigned int   _boundaryMask    : 4;
        unsigned int   _boundaryIndex   : 2;
        unsigned int   _boundaryCount   : 3;
        unsigned int   _hasBoundaryEdge : 3;
        unsigned int   _isSingleCrease  : 1;

        void clear();
        void assignBoundaryPropertiesFromEdgeMask(int boundaryEdgeMask);
        void assignBoundaryPropertiesFromVertexMask(int boundaryVertexMask);
        void assignTransitionPropertiesFromEdgeMask(int boundaryVertexMask);
    };
    typedef std::vector<PatchFaceTag> PatchTagVector;
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv


#endif /* OPENSUBDIV3_FAR_PATCH_TABLE_FACTORY_H */
