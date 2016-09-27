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

#ifndef OPENSUBDIV3_FAR_PATCH_TABLE_H
#define OPENSUBDIV3_FAR_PATCH_TABLE_H

#include "../version.h"

#include "../far/patchDescriptor.h"
#include "../far/patchParam.h"
#include "../far/stencilTable.h"

#include "../sdc/options.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Container for arrays of parametric patches
///
/// PatchTable contain topology and parametric information about the patches
/// generated by the Refinement process. Patches in the table are sorted into
/// arrays based on their PatchDescriptor Type.
///
/// Note : PatchTable can be accessed either using a PatchHandle or a
///        combination of array and patch indices.
///
/// XXXX manuelk we should add a PatchIterator that can dereference into
///              a PatchHandle for fast linear traversal of the table
///
class PatchTable {

public:

    /// \brief Handle that can be used as unique patch identifier within PatchTable
    class PatchHandle {
    // XXXX manuelk members will eventually be made private
    public:

        friend class PatchTable;
        friend class PatchMap;

        Index arrayIndex, // Array index of the patch
              patchIndex, // Absolute Index of the patch
              vertIndex;  // Relative offset to the first CV of the patch in array
    };

public:

    /// \brief Copy constructor
    PatchTable(PatchTable const & src);

    /// \brief Destructor
    ~PatchTable();

    /// \brief True if the patches are of feature adaptive types
    bool IsFeatureAdaptive() const;

    /// \brief Returns the total number of control vertex indices in the table
    int GetNumControlVerticesTotal() const {
        return (int)_patchVerts.size();
    }

    /// \brief Returns the total number of patches stored in the table
    int GetNumPatchesTotal() const;

    /// \brief Returns max vertex valence
    int GetMaxValence() const { return _maxValence; }

    /// \brief Returns the total number of ptex faces in the mesh
    int GetNumPtexFaces() const { return _numPtexFaces; }


    //@{
    ///  @name Individual patches
    ///
    /// \anchor individual_patches
    ///
    /// \brief Accessors for individual patches
    ///

    /// \brief Returns the PatchDescriptor for the patch identified by \p handle
    PatchDescriptor GetPatchDescriptor(PatchHandle const & handle) const;

    /// \brief Returns the control vertex indices for the patch identified by \p handle
    ConstIndexArray GetPatchVertices(PatchHandle const & handle) const;

    /// \brief Returns a PatchParam for the patch identified by \p handle
    PatchParam GetPatchParam(PatchHandle const & handle) const;

    /// \brief Returns the control vertex indices for \p patch in \p array
    ConstIndexArray GetPatchVertices(int array, int patch) const;

    /// \brief Returns the PatchParam for \p patch in \p array
    PatchParam GetPatchParam(int array, int patch) const;
    //@}


    //@{
    ///  @name Arrays of patches
    ///
    /// \anchor arrays_of_patches
    ///
    /// \brief Accessors for arrays of patches of the same type
    ///

    /// \brief Returns the number of patch arrays in the table
    int GetNumPatchArrays() const;

    /// \brief Returns the number of patches in \p array
    int GetNumPatches(int array) const;

    /// \brief Returns the number of control vertices in \p array
    int GetNumControlVertices(int array) const;

    /// \brief Returns the PatchDescriptor for the patches in \p array
    PatchDescriptor GetPatchArrayDescriptor(int array) const;

    /// \brief Returns the control vertex indices for the patches in \p array
    ConstIndexArray GetPatchArrayVertices(int array) const;

    /// \brief Returns the PatchParams for the patches in \p array
    ConstPatchParamArray const GetPatchParams(int array) const;
    //@}


    //@{
    ///  @name Change of basis patches
    ///
    /// \anchor change_of_basis_patches
    ///
    /// \brief Accessors for change of basis patches
    ///
    ///

    /// \brief Returns the number of local vertex points.
    int GetNumLocalPoints() const;

    /// \brief Updates local point vertex values.
    ///
    /// @param src       Buffer with primvar data for the coarse vertex values
    ///                  and refined vertex values
    ///
    /// @param dst       Destination buffer for the computed local point
    ///                  vertex values
    ///
    ///
    template <class T> void
    ComputeLocalPointValues(T const *src, T *dst) const;

    /// \brief Returns the stencil table to compute local point vertex values
    StencilTable const *GetLocalPointStencilTable() const {
        return _localPointStencils;
    }

    /// \brief Returns the number of local varying points.
    int GetNumLocalPointsVarying() const;

    /// \brief Updates local point varying values.
    ///
    /// @param src       Buffer with primvar data for the coarse varying values
    ///                  and refined varying values
    ///
    /// @param dst       Destination buffer for the computed local point
    ///                  varying values
    ///
    ///
    template <class T> void
    ComputeLocalPointValuesVarying(T const *src, T *dst) const;

    /// \brief Returns the stencil table to compute local point varying values
    StencilTable const *GetLocalPointVaryingStencilTable() const {
        return _localPointVaryingStencils;
    }

    /// \brief Returns the number of local face-varying points for \p channel
    int GetNumLocalPointsFaceVarying(int channel = 0) const;

    /// \brief Updates local point face-varying values.
    ///
    /// @param src       Buffer with primvar data for the coarse face-varying
    ///                  values and refined face-varying values
    ///
    /// @param dst       Destination buffer for the computed local point
    ///                  face-varying values
    ///
    ///
    template <class T> void
    ComputeLocalPointValuesFaceVarying(T const *src, T *dst, int channel = 0) const;

    /// \brief Returns the stencil table to compute local point face-varying values
    StencilTable const *GetLocalPointFaceVaryingStencilTable(int channel = 0) const {
        if (channel >= 0 && channel < (int)_localPointFaceVaryingStencils.size()) {
            return _localPointFaceVaryingStencils[channel];
        }
        return NULL;
    }
    //@}


    //@{
    ///  @name Legacy gregory patch evaluation buffers

    /// \brief Accessors for the gregory patch evaluation buffers.
    ///        These methods will be deprecated.
    ///
    typedef Vtr::ConstArray<unsigned int> ConstQuadOffsetsArray;

    /// \brief Returns the 'QuadOffsets' for the Gregory patch identified by \p handle
    ConstQuadOffsetsArray GetPatchQuadOffsets(PatchHandle const & handle) const;

    typedef std::vector<Index> VertexValenceTable;

    /// \brief Returns the 'VertexValences' table (vertex neighborhoods table)
    VertexValenceTable const & GetVertexValenceTable() const {
        return _vertexValenceTable;
    }
    //@}


    //@{
    ///  @name Single-crease patches
    ///
    /// \anchor single_crease_patches
    ///
    /// \brief Accessors for single-crease patch edge sharpness
    ///

    /// \brief Returns the crease sharpness for the patch identified by \p handle
    ///        if it is a single-crease patch, or 0.0f
    float GetSingleCreasePatchSharpnessValue(PatchHandle const & handle) const;

    /// \brief Returns the crease sharpness for the \p patch in \p array
    ///        if it is a single-crease patch, or 0.0f
    float GetSingleCreasePatchSharpnessValue(int array, int patch) const;
    //@}


    //@{
    ///  @name Varying data
    ///
    /// \anchor varying_data
    ///
    /// \brief Accessors for varying data
    ///

    /// \brief Returns the varying vertex indices for a given patch
    ConstIndexArray GetPatchVaryingVertices(PatchHandle const & handle) const;

    /// \brief Returns the varying vertex indices for a given patch
    ConstIndexArray GetPatchVaryingVertices(int array, int patch) const;

    /// \brief Returns the varying vertex indices for the patches in \p array
    ConstIndexArray GetPatchArrayVaryingVertices(int array) const;

    /// \brief Returns an array of varying vertex indices for the patches.
    ConstIndexArray GetVaryingVertices() const;
    //@}


    //@{
    ///  @name Face-varying channels
    ///
    /// \anchor face_varying_channels
    ///
    /// \brief Accessors for face-varying channels
    ///

    /// \brief Returns the number of face-varying channels
    int GetNumFVarChannels() const;

    /// \brief Deprecated Returns the interpolation mode for \p channel
    Sdc::Options::FVarLinearInterpolation GetFVarChannelLinearInterpolation(int channel = 0) const;

    /// \brief Returns the patch descriptor for \p channel
    PatchDescriptor GetFVarChannelPatchDescriptor(int channel = 0) const;

    /// \brief Returns the value indices for a given patch in \p channel
    ConstIndexArray GetPatchFVarValues(PatchHandle const & handle, int channel = 0) const;

    /// \brief Returns the value indices for a given patch in \p channel
    ConstIndexArray GetPatchFVarValues(int array, int patch, int channel = 0) const;

    /// \brief Returns the value indices for the patches in \p array in \p channel
    ConstIndexArray GetPatchArrayFVarValues(int array, int channel = 0) const;

    /// \brief Returns an array of value indices for the patches in \p channel
    ConstIndexArray GetFVarValues(int channel = 0) const;

    /// \brief Returns the value indices for a given patch in \p channel
    PatchParam GetPatchFVarPatchParam(PatchHandle const & handle, int channel = 0) const;

    /// \brief Returns the face-varying params for a given patch \p channel
    PatchParam GetPatchFVarPatchParam(int array, int patch, int channel = 0) const;

    /// \brief Returns the face-varying for a given patch in \p array in \p channel
    ConstPatchParamArray GetPatchArrayFVarPatchParam(int array, int channel = 0) const;

    /// \brief Returns an array of face-varying patch param for \p channel
    ConstPatchParamArray GetFVarPatchParam(int channel = 0) const;
    //@}


    //@{
    ///  @name Direct accessors
    ///
    /// \warning These direct accessors are left for convenience, but they are
    ///          likely going to be deprecated in future releases
    ///

    typedef std::vector<Index> PatchVertsTable;

    /// \brief Get the table of patch control vertices
    PatchVertsTable const & GetPatchControlVerticesTable() const { return _patchVerts; }

    /// \brief Returns the PatchParamTable (PatchParams order matches patch array sorting)
    PatchParamTable const & GetPatchParamTable() const { return _paramTable; }

    /// \brief Returns a sharpness index table for each patch (if exists)
    std::vector<Index> const &GetSharpnessIndexTable() const { return _sharpnessIndices; }

    /// \brief Returns sharpness values table
    std::vector<float> const &GetSharpnessValues() const { return _sharpnessValues; }

    typedef std::vector<unsigned int> QuadOffsetsTable;

    /// \brief Returns the quad-offsets table
    QuadOffsetsTable const & GetQuadOffsetsTable() const {
        return _quadOffsetsTable;
    }
    //@}

    /// debug helper
    void print() const;

public:

    //@{
    ///  @name Evaluation methods
    ///

    /// \brief Evaluate basis functions for position and derivatives at a
    /// given (s,t) parametric location of a patch.
    ///
    /// @param handle  A patch handle indentifying the sub-patch containing the
    ///                (s,t) location
    ///
    /// @param s       Patch coordinate (in coarse face normalized space)
    ///
    /// @param t       Patch coordinate (in coarse face normalized space)
    ///
    /// @param wP      Weights (evaluated basis functions) for the position
    ///
    /// @param wDs     Weights (evaluated basis functions) for derivative wrt s
    ///
    /// @param wDt     Weights (evaluated basis functions) for derivative wrt t
    ///
    /// @param wDss    Weights (evaluated basis functions) for derivative wrt ss
    ///
    /// @param wDst    Weights (evaluated basis functions) for derivative wrt st
    ///
    /// @param wDtt    Weights (evaluated basis functions) for derivative wrt tt
    ///
    void EvaluateBasis(PatchHandle const & handle, float s, float t,
        float wP[], float wDs[] = 0, float wDt[] = 0,
        float wDss[] = 0, float wDst[] = 0, float wDtt[] = 0) const;

    /// \brief Evaluate basis functions for a varying value and
    /// derivatives at a given (s,t) parametric location of a patch.
    ///
    /// @param handle  A patch handle indentifying the sub-patch containing the
    ///                (s,t) location
    ///
    /// @param s       Patch coordinate (in coarse face normalized space)
    ///
    /// @param t       Patch coordinate (in coarse face normalized space)
    ///
    /// @param wP      Weights (evaluated basis functions) for the position
    ///
    /// @param wDs     Weights (evaluated basis functions) for derivative wrt s
    ///
    /// @param wDt     Weights (evaluated basis functions) for derivative wrt t
    ///
    /// @param wDss    Weights (evaluated basis functions) for derivative wrt ss
    ///
    /// @param wDst    Weights (evaluated basis functions) for derivative wrt st
    ///
    /// @param wDtt    Weights (evaluated basis functions) for derivative wrt tt
    ///
    void EvaluateBasisVarying(PatchHandle const & handle, float s, float t,
        float wP[], float wDs[] = 0, float wDt[] = 0,
        float wDss[] = 0, float wDst[] = 0, float wDtt[] = 0) const;

    /// \brief Evaluate basis functions for a face-varying value and
    /// derivatives at a given (s,t) parametric location of a patch.
    ///
    /// @param handle  A patch handle indentifying the sub-patch containing the
    ///                (s,t) location
    ///
    /// @param s       Patch coordinate (in coarse face normalized space)
    ///
    /// @param t       Patch coordinate (in coarse face normalized space)
    ///
    /// @param wP      Weights (evaluated basis functions) for the position
    ///
    /// @param wDs     Weights (evaluated basis functions) for derivative wrt s
    ///
    /// @param wDt     Weights (evaluated basis functions) for derivative wrt t
    ///
    /// @param wDss    Weights (evaluated basis functions) for derivative wrt ss
    ///
    /// @param wDst    Weights (evaluated basis functions) for derivative wrt st
    ///
    /// @param wDtt    Weights (evaluated basis functions) for derivative wrt tt
    ///
    /// @param channel face-varying channel
    ///
    void EvaluateBasisFaceVarying(PatchHandle const & handle, float s, float t,
        float wP[], float wDs[] = 0, float wDt[] = 0,
        float wDss[] = 0, float wDst[] = 0, float wDtt[] = 0,
        int channel = 0) const;
    //@}

protected:

    friend class PatchTableFactory;

    // Factory constructor
    PatchTable(int maxvalence);

    Index getPatchIndex(int array, int patch) const;

    PatchParamArray getPatchParams(int arrayIndex);

    Index * getSharpnessIndices(Index arrayIndex);
    float * getSharpnessValues(Index arrayIndex);

private:

    //
    // Patch arrays
    //

    struct PatchArray;
    typedef std::vector<PatchArray> PatchArrayVector;

    PatchArray & getPatchArray(Index arrayIndex);
    PatchArray const & getPatchArray(Index arrayIndex) const;

    void reservePatchArrays(int numPatchArrays);
    void pushPatchArray(PatchDescriptor desc, int npatches,
        Index * vidx, Index * pidx, Index * qoidx=0);

    IndexArray getPatchArrayVertices(int arrayIndex);

    Index findPatchArray(PatchDescriptor desc);


    //
    // Varying patch arrays
    //
    IndexArray getPatchArrayVaryingVertices(int arrayIndex);

    void allocateVaryingVertices(
        PatchDescriptor desc, int numPatches);
    void populateVaryingVertices();

    //
    // Face-varying patch channels
    //

    struct FVarPatchChannel;
    typedef std::vector<FVarPatchChannel> FVarPatchChannelVector;

    FVarPatchChannel & getFVarPatchChannel(int channel);
    FVarPatchChannel const & getFVarPatchChannel(int channel) const;

    void allocateFVarPatchChannels(int numChannels);
    void allocateFVarPatchChannelValues(
        PatchDescriptor desc, int numPatches, int channel);

    // deprecated
    void setFVarPatchChannelLinearInterpolation(
        Sdc::Options::FVarLinearInterpolation interpolation, int channel);

    IndexArray getFVarValues(int channel);
    ConstIndexArray getPatchFVarValues(int patch, int channel) const;

    PatchParamArray getFVarPatchParam(int channel);
    PatchParam getPatchFVarPatchParam(int patch, int channel) const;

private:

    //
    // Topology
    //

    int _maxValence,   // highest vertex valence found in the mesh
        _numPtexFaces; // total number of ptex faces

    PatchArrayVector     _patchArrays;  // Vector of descriptors for arrays of patches

    std::vector<Index>   _patchVerts;   // Indices of the control vertices of the patches

    PatchParamTable      _paramTable;   // PatchParam bitfields (one per patch)

    //
    // Extraordinary vertex closed-form evaluation / endcap basis conversion
    //
    // XXXtakahito: these data will probably be replaced with mask coefficient or something
    //              SchemeWorker populates.
    //
    QuadOffsetsTable     _quadOffsetsTable;   // Quad offsets (for Gregory patches)
    VertexValenceTable   _vertexValenceTable; // Vertex valence table (for Gregory patches)
    StencilTable const * _localPointStencils;  // endcap basis conversion stencils
    StencilTable const * _localPointVaryingStencils; // endcap varying stencils (for convenience)

    //
    // Varying data
    //

    PatchDescriptor _varyingDesc;

    std::vector<Index>   _varyingVerts;

    //
    // Face-varying data
    //

    FVarPatchChannelVector _fvarChannels;

    std::vector<StencilTable const *> _localPointFaceVaryingStencils;

    //
    // 'single-crease' patch sharpness tables
    //

    std::vector<Index>   _sharpnessIndices; // Indices of single-crease sharpness (one per patch)
    std::vector<float>   _sharpnessValues;  // Sharpness values.
};

template <class T>
inline void
PatchTable::ComputeLocalPointValues(T const *src, T *dst) const {
    if (_localPointStencils) {
        _localPointStencils->UpdateValues(src, dst);
    }
}

template <class T>
inline void
PatchTable::ComputeLocalPointValuesVarying(T const *src, T *dst) const {
    if (_localPointVaryingStencils) {
        _localPointVaryingStencils->UpdateValues(src, dst);
    }
}

template <class T>
inline void
PatchTable::ComputeLocalPointValuesFaceVarying(T const *src, T *dst, int channel) const {
    if (channel >= 0 && channel < (int)_localPointFaceVaryingStencils.size()) {
        if (_localPointFaceVaryingStencils[channel]) {
            _localPointFaceVaryingStencils[channel]->UpdateValues(src, dst);
        }
    }
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PATCH_TABLE */
