//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
#ifndef OSD_CPU_EVAL_LIMIT_CONTEXT_H
#define OSD_CPU_EVAL_LIMIT_CONTEXT_H

#include "../version.h"

#include "../osd/evalLimitContext.h"
#include "../osd/vertexDescriptor.h"
#include "../far/patchTables.h"

#include <map>
#include <stdio.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCpuEvalLimitContext : public OsdEvalLimitContext {
public:
    /// \brief Factory
    /// Returns an EvalLimitContext from the given farmesh.
    /// Note : the farmesh is expected to be feature-adaptive and have ptex
    ///        coordinates tables.
    static OsdCpuEvalLimitContext * Create(FarMesh<OsdVertex> const * farmesh);

    /// Destructor
    virtual ~OsdCpuEvalLimitContext();

    /// Limit evaluation data descriptor
    class EvalData {
    public:
        OsdVertexBufferDescriptor const & GetInputDesc() const {
            return _inDesc;
        }

        float const * GetInputData() const {
            return _inQ;
        }
        
        OsdVertexBufferDescriptor const & GetOutputDesc() const {
            return _outDesc;
        }
        
        float const * GetOutputData(int index=0) const {
            return _outQ + index * _outDesc.stride;
        }
        
        float const * GetOutputDU(int index=0) const {
            return _outdQu + index * _outDesc.stride;
        }

        float const * GetOutputDV(int index=0) const {
            return _outdQv + index * _outDesc.stride;
        }
        
        bool IsBound() const {
            return _inQ and _outQ;
        }

    private:
        friend class OsdCpuEvalLimitContext;
        
        OsdVertexBufferDescriptor _inDesc; // input data
        float * _inQ;     
        
        OsdVertexBufferDescriptor _outDesc; // output data
        float * _outQ,    
              * _outdQu,   // U derivative of output data
              * _outdQv;   // V derivative of output data
              
        /// Binds the data buffers.
        ///
        /// @param inDesc vertex / varying data descriptor shared by all input data buffers
        ///
        /// @param inQ input subidivision data
        ///
        /// @param outDesc vertex buffer data descriptor shared by all output data buffers
        ///
        /// @param outQ output vertex data
        ///
        /// @param outdQu optional output derivative along "u" of the vertex data
        ///
        /// @param outdQv optional output derivative along "v" of the vertex data
        ///
        template<class INPUT_BUFFER, class OUTPUT_BUFFER>
        void Bind( OsdVertexBufferDescriptor const & inDesc, INPUT_BUFFER *inQ,
                   OsdVertexBufferDescriptor const & outDesc, OUTPUT_BUFFER *outQ, 
                                                              OUTPUT_BUFFER *outdQu=0,
                                                              OUTPUT_BUFFER *outdQv=0);
        
        /// Resets the descriptors & pointers
        void Unbind();
    };
    
    EvalData const & GetVertexData() const {
        return _vertexData;
    }

    EvalData const & GetVaryingData() const {
        return _varyingData;
    }


    /// Binds the vertex-interpolated data buffers.
    ///
    /// @param inDesc vertex buffer data descriptor shared by all input data buffers
    ///
    /// @param inQ input vertex data
    ///
    /// @param outDesc vertex buffer data descriptor shared by all output data buffers
    ///
    /// @param outQ output vertex data
    ///
    /// @param outdQu optional output derivative along "u" of the vertex data
    ///
    /// @param outdQv optional output derivative along "v" of the vertex data
    ///
    template<class VERTEX_BUFFER, class OUTPUT_BUFFER>
    void BindVertexBuffers( OsdVertexBufferDescriptor const & inDesc, VERTEX_BUFFER *inQ,
                            OsdVertexBufferDescriptor const & outDesc, OUTPUT_BUFFER *outQ, 
                                                                       OUTPUT_BUFFER *outdQu=0, 
                                                                       OUTPUT_BUFFER *outdQv=0) {
        _vertexData.Bind( inDesc, inQ, outDesc, outQ, outdQu, outdQv );
    }

    /// Unbind the vertex data buffers
    void UnbindVertexBuffers();

    /// Binds the varying-interpolated data buffers.
    ///
    /// @param inDesc varying buffer data descriptor shared by all input data buffers
    ///
    /// @param inQ input varying data
    ///
    /// @param outDesc varying buffer data descriptor shared by all output data buffers
    ///
    /// @param outQ output varying data
    ///
    /// @param outdQu optional output derivative along "u" of the varying data
    ///
    /// @param outdQv optional output derivative along "v" of the varying data
    ///
    template<class VARYING_BUFFER, class OUTPUT_BUFFER>
    void BindVaryingBuffers( OsdVertexBufferDescriptor const & inDesc, VARYING_BUFFER *inQ,
                             OsdVertexBufferDescriptor const & outDesc, OUTPUT_BUFFER *outQ) {
        _varyingData.Bind( inDesc, inQ, outDesc, outQ );
    }

    /// Unbind the varying data buffers
    void UnbindVaryingBuffers();

    
    /// Returns the vector of patch arrays
    const FarPatchTables::PatchArrayVector & GetPatchArrayVector() const {
        return _patchArrays;
    }
    
    /// Returns the vector of per-patch parametric data
    const std::vector<FarPatchParam::BitField> & GetPatchBitFields() const {
        return _patchBitFields;
    }

    /// The ordered array of control vertex indices for all the patches
    const std::vector<unsigned int> & GetControlVertices() const {
        return _patches;
    }

    /// Returns the vertex-valence buffer used for Gregory patch computations
    const int * GetVertexValenceBuffer() const {
        return &_vertexValenceBuffer[0];
    }

    /// Returns the Quad-Offsets buffer used for Gregory patch computations
    const unsigned int *GetQuadOffsetBuffer() const {
        return &_quadOffsetBuffer[0];
    }

    /// Returns a map object that can connect a faceId to a list of children patches
    const FarPatchTables::PatchMap * GetPatchesMap() const {
        return _patchMap;
    }

    /// Returns the highest valence of the vertices in the buffers
    int GetMaxValence() const {
        return _maxValence;
    }

protected:
    explicit OsdCpuEvalLimitContext(FarMesh<OsdVertex> const * farmesh);

private:

    // Topology data for a mesh
    FarPatchTables::PatchArrayVector     _patchArrays;    // patch descriptor for each patch in the mesh
    FarPatchTables::PTable               _patches;        // patch control vertices
    std::vector<FarPatchParam::BitField> _patchBitFields; // per-patch parametric info
    
    FarPatchTables::VertexValenceTable   _vertexValenceBuffer; // extra Gregory patch data buffers
    FarPatchTables::QuadOffsetTable      _quadOffsetBuffer;

    FarPatchTables::PatchMap * _patchMap; // map of the sub-patches given a face index

    EvalData _vertexData,
             _varyingData;

    int _maxValence;
};

template<class INPUT_BUFFER, class OUTPUT_BUFFER> void 
OsdCpuEvalLimitContext::EvalData::Bind( OsdVertexBufferDescriptor const & inDesc,
                                        INPUT_BUFFER *inQ,
                                        OsdVertexBufferDescriptor const & outDesc,
                                        OUTPUT_BUFFER *outQ,
                                        OUTPUT_BUFFER *outdQu,
                                        OUTPUT_BUFFER *outdQv) {
    _inDesc = inDesc;
    _inQ = inQ ? inQ->BindCpuBuffer() : 0;

    _outDesc = outDesc;
    _outQ = outQ ? outQ->BindCpuBuffer() : 0 ;
    _outdQu = outdQu ? outdQu->BindCpuBuffer() : 0 ;
    _outdQv = outdQv ? outdQv->BindCpuBuffer() : 0 ;
}
        



} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_EVAL_LIMIT_CONTEXT_H */
