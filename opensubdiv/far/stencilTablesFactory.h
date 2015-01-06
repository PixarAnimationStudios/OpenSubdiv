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

#ifndef FAR_STENCILTABLE_FACTORY_H
#define FAR_STENCILTABLE_FACTORY_H

#include "../version.h"

#include "../far/kernelBatch.h"
#include "../far/patchTables.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

class TopologyRefiner;

class Stencil;
class StencilTables;
class LimitStencil;
class LimitStencilTables;

/// \brief A specialized factory for StencilTables
///
class StencilTablesFactory {

public:

    enum Mode {
        INTERPOLATE_VERTEX=0,
        INTERPOLATE_VARYING
    };

    struct Options {

        Options() : interpolationMode(INTERPOLATE_VERTEX),
                    generateOffsets(false),
                    generateControlVerts(false),
                    generateIntermediateLevels(true),
                    factorizeIntermediateLevels(true),
                    maxLevel(10) { }

        unsigned int interpolationMode           : 2, ///< interpolation mode
                     generateOffsets             : 1, ///< populate optional "_offsets" field
                     generateControlVerts        : 1, ///< generate stencils for control-vertices
                     generateIntermediateLevels  : 1, ///< vertices at all levels or highest only
                     factorizeIntermediateLevels : 1, ///< accumulate stencil weights from control
                                                      ///  vertices or from the stencils of the
                                                      ///  previous level
                     maxLevel                    : 4; ///< generate stencils up to 'maxLevel'
    };

    /// \brief Instantiates StencilTables from TopologyRefiner that have been
    ///        refined uniformly or adaptively.
    ///
    /// \note The factory only creates stencils for vertices that have already
    ///       been refined in the TopologyRefiner. Use RefineUniform() or
    ///       RefineAdaptive() before constructing the stencils.
    ///
    /// @param refiner  The TopologyRefiner containing the topology
    ///
    /// @param options  Options controlling the creation of the tables
    ///
    static StencilTables const * Create(TopologyRefiner const & refiner,
        Options options = Options());


    /// \brief Instantiates StencilTables by concatenating an array of existing
    ///        stencil tables.
    ///
    /// \note This factory checks that the stencil tables point to the same set
    ///       of supporting control vertices - no re-indexing is done.
    ///       GetNumControlVertices() *must* return the same value for all input
    ///       tables.
    ///
    /// @param numTables Number of input StencilTables
    ///
    /// @param tables    Array of input StencilTables
    ///
    static StencilTables const * Create(int numTables, StencilTables const ** tables);

    /// \brief Returns a KernelBatch applying all the stencil in the tables
    ///        to primvar data.
    ///
    /// @param stencilTables The stencil tables to batch
    ///
    static KernelBatch Create(StencilTables const &stencilTables);

private:

    // Generate stencils for the coarse control-vertices (single weight = 1.0f)
    static void generateControlVertStencils(int numControlVerts, Stencil & dst);
};

/// \brief A specialized factory for LimitStencilTables
///
/// The LimitStencilTablesFactory creates tables of limit stencils. Limit
/// stencils can interpolate any arbitrary location on the limit surface.
/// The stencils will be bilinear if the surface is refined uniformly, and
/// bicubic if feature adaptive isolation is used instead.
///
/// Surface locations are expressed as a combination of ptex face index and
/// normalized (s,t) patch coordinates. The factory exposes the LocationArray
/// struct as a container for these location descriptors.
///
class LimitStencilTablesFactory {

public:

    /// \brief Descriptor for limit surface locations
    struct LocationArray {

        LocationArray() : ptexIdx(-1), numLocations(0), s(0), t(0) { }

        int ptexIdx,        ///< ptex face index
            numLocations;   ///< number of (u,v) coordinates in the array

        float const * s,    ///< array of u coordinates
                    * t;    ///< array of v coordinates
    };

    typedef std::vector<LocationArray> LocationArrayVec;

    /// \brief Instantiates LimitStencilTables from a TopologyRefiner that has
    ///        been refined either uniformly or adaptively.
    ///
    /// @param refiner          The TopologyRefiner containing the topology
    ///
    /// @param locationArrays   An array of surface location descriptors
    ///                         (see LocationArray)
    ///
    /// @param cvStencils       A set of StencilTables generated from the
    ///                         TopologyRefiner (optional: prevents redundant
    ///                         instanciation of the tables if available)
    ///
    /// @param patchTables      A set of PatchTables generated from the
    ///                         TopologyRefiner (optional: prevents redundant
    ///                         instanciation of the tables if available)
    ///
    static LimitStencilTables const * Create(TopologyRefiner const & refiner,
        LocationArrayVec const & locationArrays,
            StencilTables const * cvStencils=0,
                PatchTables const * patchTables=0);
};


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_STENCILTABLE_FACTORY_H
