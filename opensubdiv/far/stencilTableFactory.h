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

#ifndef OPENSUBDIV3_FAR_STENCILTABLE_FACTORY_H
#define OPENSUBDIV3_FAR_STENCILTABLE_FACTORY_H

#include "../version.h"

#include "../far/patchTable.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

class TopologyRefiner;

class Stencil;
class StencilTable;
class LimitStencil;
class LimitStencilTable;

/// \brief A specialized factory for StencilTable
///
class StencilTableFactory {

public:

    enum Mode {
        INTERPOLATE_VERTEX=0,           ///< vertex primvar stencils
        INTERPOLATE_VARYING,            ///< varying primvar stencils
        INTERPOLATE_FACE_VARYING        ///< face-varying primvar stencils
    };

    struct Options {

        Options() : interpolationMode(INTERPOLATE_VERTEX),
                    generateOffsets(false),
                    generateControlVerts(false),
                    generateIntermediateLevels(true),
                    factorizeIntermediateLevels(true),
                    maxLevel(10),
                    fvarChannel(0) { }

        unsigned int interpolationMode           : 2, ///< interpolation mode
                     generateOffsets             : 1, ///< populate optional "_offsets" field
                     generateControlVerts        : 1, ///< generate stencils for control-vertices
                     generateIntermediateLevels  : 1, ///< vertices at all levels or highest only
                     factorizeIntermediateLevels : 1, ///< accumulate stencil weights from control
                                                      ///  vertices or from the stencils of the
                                                      ///  previous level
                     maxLevel                    : 4; ///< generate stencils up to 'maxLevel'
        unsigned int fvarChannel;                     ///< face-varying channel to use
                                                      ///  when generating face-varying stencils
    };

    /// \brief Instantiates StencilTable from TopologyRefiner that have been
    ///        refined uniformly or adaptively.
    ///
    /// \note The factory only creates stencils for vertices that have already
    ///       been refined in the TopologyRefiner. Use RefineUniform() or
    ///       RefineAdaptive() before constructing the stencils.
    ///
    /// @param refiner  The TopologyRefiner containing the topology
    ///
    /// @param options  Options controlling the creation of the table
    ///
    static StencilTable const * Create(TopologyRefiner const & refiner,
        Options options = Options());


    /// \brief Instantiates StencilTable by concatenating an array of existing
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
    static StencilTable const * Create(int numTables, StencilTable const ** tables);


    /// \brief Utility function for stencil splicing for local point stencils.
    ///
    /// @param refiner              The TopologyRefiner containing the topology
    ///
    /// @param baseStencilTable     Input StencilTable for refined vertices
    ///
    /// @param localPointStencilTable
    ///                             StencilTable for the change of basis patch points.
    ///
    /// @param factorize            If factorize set to true, endcap stencils will be
    ///                             factorized with supporting vertices from baseStencil
    ///                             table so that the endcap points can be computed
    ///                             directly from control vertices.
    ///
    static StencilTable const * AppendLocalPointStencilTable(
        TopologyRefiner const &refiner,
        StencilTable const *baseStencilTable,
        StencilTable const *localPointStencilTable,
        bool factorize = true);

    /// \brief Utility function for stencil splicing for local point
    /// face-varying stencils.
    ///
    /// @param refiner              The TopologyRefiner containing the topology
    ///
    /// @param baseStencilTable     Input StencilTable for refined vertices
    ///
    /// @param localPointStencilTable
    ///                             StencilTable for the change of basis patch points.
    ///
    /// @param channel              face-varying channel
    ///
    /// @param factorize            If factorize sets to true, endcap stencils will be
    ///                             factorized with supporting vertices from baseStencil
    ///                             table so that the endcap points can be computed
    ///                             directly from control vertices.
    ///
    static StencilTable const * AppendLocalPointStencilTableFaceVarying(
        TopologyRefiner const &refiner,
        StencilTable const *baseStencilTable,
        StencilTable const *localPointStencilTable,
        int channel = 0,
        bool factorize = true);

private:

    // Generate stencils for the coarse control-vertices (single weight = 1.0f)
    static void generateControlVertStencils(int numControlVerts, Stencil & dst);

    // Internal method to splice local point stencils
    static StencilTable const * appendLocalPointStencilTable(
        TopologyRefiner const &refiner,
        StencilTable const * baseStencilTable,
        StencilTable const * localPointStencilTable,
        int channel,
        bool factorize);
};

/// \brief A specialized factory for LimitStencilTable
///
/// The LimitStencilTableFactory creates a table of limit stencils. Limit
/// stencils can interpolate any arbitrary location on the limit surface.
/// The stencils will be bilinear if the surface is refined uniformly, and
/// bicubic if feature adaptive isolation is used instead.
///
/// Surface locations are expressed as a combination of ptex face index and
/// normalized (s,t) patch coordinates. The factory exposes the LocationArray
/// struct as a container for these location descriptors.
///
class LimitStencilTableFactory {

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

    struct Options {

        Options() : generate1stDerivatives(true),
                    generate2ndDerivatives(false) { }

        unsigned int generate1stDerivatives      : 1, ///< Generate weights for 1st derivatives
                     generate2ndDerivatives      : 1; ///< Generate weights for 2nd derivatives
    };

    /// \brief Instantiates LimitStencilTable from a TopologyRefiner that has
    ///        been refined either uniformly or adaptively.
    ///
    /// @param refiner          The TopologyRefiner containing the topology
    ///
    /// @param locationArrays   An array of surface location descriptors
    ///                         (see LocationArray)
    ///
    /// @param cvStencils       A set of StencilTable generated from the
    ///                         TopologyRefiner (optional: prevents redundant
    ///                         instantiation of the table if available)
    ///
    /// @param patchTable       A set of PatchTable generated from the
    ///                         TopologyRefiner (optional: prevents redundant
    ///                         instantiation of the table if available)
    ///
    /// @param options          Options controlling the creation of the table
    ///
    static LimitStencilTable const * Create(TopologyRefiner const & refiner,
        LocationArrayVec const & locationArrays,
            StencilTable const * cvStencils=0,
              PatchTable const * patchTable=0,
                         Options options=Options());

};


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OPENSUBDIV3_FAR_STENCILTABLE_FACTORY_H
