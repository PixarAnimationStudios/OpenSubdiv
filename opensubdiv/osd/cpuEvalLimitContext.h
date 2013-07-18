//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#ifndef OSD_CPU_EVAL_LIMIT_CONTEXT_H
#define OSD_CPU_EVAL_LIMIT_CONTEXT_H

#include "../version.h"

#include "../osd/evalLimitContext.h"
#include "../osd/vertexDescriptor.h"
#include "../far/patchTables.h"
#include "../far/patchMap.h"

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
    /// 
    /// @param farmesh          a pointer to an initialized farmesh
    ///
    /// @param requireFVarData  flag for generating face-varying data
    ///
    static OsdCpuEvalLimitContext * Create(FarMesh<OsdVertex> const * farmesh, 
                                           bool requireFVarData=false);

    virtual ~OsdCpuEvalLimitContext();



    /// A container able to bind vertex buffer data as input or output streams.
    class DataStream {
    public:
        /// Constructor
        DataStream() : _data(0) {  }

        /// Binds the stream to the context (and moves the data to the appropriate
        /// compute device)
        ///
        /// @param data  a valid OsdVertexBuffer 
        ///
        template <class BUFFER> void Bind( BUFFER * data ) {
            _data = data ? data->BindCpuBuffer() : 0;
        }

        /// True if the stream has been bound
        bool IsBound() const {
            return (_data!=NULL);
        }

        /// Unbinds the stream
        void Unbind() {
            _data=0;
        }

    protected:
        float * _data;
    };

    /// \brief Input (const) data stream
    class InputDataStream : public DataStream {
    public:
        /// Const accessor
        float const * GetData() const {
            return _data;
        }
    };

    /// \brief Output (const) data stream
    class OutputDataStream : public DataStream {
    public:
        /// Non-cont accessor
        float * GetData() {
            return _data;
        }
    };

    /// Vertex-interpolated streams
    struct VertexData {
    
        /// input vertex-interpolated data descriptor
        OsdVertexBufferDescriptor inDesc;

        /// input vertex-interpolated data stream
        InputDataStream in;

        /// output vertex-interpolated data descriptor
        OsdVertexBufferDescriptor outDesc;

        /// output vertex-interpolated data stream and parametric derivative streams
        OutputDataStream out,
                         outDu,
                         outDv;
                   
        /// Binds the vertex-interpolated data streams
        ///
        /// @param iDesc   data descriptor shared by all input data buffers
        ///
        /// @param inQ     input vertex data
        ///
        /// @param oDesc   data descriptor shared by all output data buffers
        ///
        /// @param outQ    output vertex data
        ///
        /// @param outdQu  output derivative along "u" of the vertex data (optional)
        ///
        /// @param outdQv  output derivative along "v" of the vertex data (optional)
        ///
        template<class VERTEX_BUFFER, class OUTPUT_BUFFER>
        void Bind( OsdVertexBufferDescriptor const & iDesc, VERTEX_BUFFER *inQ,
                   OsdVertexBufferDescriptor const & oDesc, OUTPUT_BUFFER *outQ,
                                                            OUTPUT_BUFFER *outdQu=0,
                                                            OUTPUT_BUFFER *outdQv=0) {
            inDesc = iDesc;
            in.Bind( inQ );

            outDesc = oDesc;
            out.Bind( outQ );
            outDu.Bind( outdQu );
            outDv.Bind( outdQv );
        }
        
        /// True if both the mandatory input and output streams have been bound
        bool IsBound() const {
            return in.IsBound() and out.IsBound();
        }
    
        /// Unbind the vertex data streams
        void Unbind();
    };

    /// Returns an Eval data descriptor of the vertex-interpolated data currently
    /// bound to this EvalLimitContext.
    VertexData & GetVertexData() {
        return _vertexData;
    }




    /// Varying-interpolated streams
    struct VaryingData {

        /// input varying-interpolated data descriptor
        OsdVertexBufferDescriptor inDesc;

        /// input varying-interpolated data stream
        InputDataStream in;

        /// output varying-interpolated data descriptor
        OsdVertexBufferDescriptor outDesc;

        /// output varying-interpolated data stream
        OutputDataStream out;

        /// Binds the varying-interpolated data streams
        ///
        /// @param iDesc  data descriptor shared by all input data buffers
        ///
        /// @param inQ    input varying data
        ///
        /// @param oDesc  data descriptor shared by all output data buffers
        ///
        /// @param outQ   output varying data
        ///
        template<class VARYING_BUFFER, class OUTPUT_BUFFER>
        void Bind( OsdVertexBufferDescriptor const & iDesc, VARYING_BUFFER *inQ,
                   OsdVertexBufferDescriptor const & oDesc, OUTPUT_BUFFER *outQ ) {
            inDesc = iDesc;
            in.Bind( inQ );

            outDesc = oDesc;
            out.Bind( outQ );
        }
        
        /// True if both the mandatory input and output streams have been bound
        bool IsBound() const {
            return in.IsBound() and out.IsBound();
        }
    
        /// Unbind the vertex data streams
        void Unbind();
    };
    
    /// Returns an Eval data descriptor of the varying-interpolated data currently
    /// bound to this EvalLimitContext.
    VaryingData & GetVaryingData() {
        return _varyingData;
    }



    /// Face-Varying-interpolated streams
    struct FaceVaryingData {
    
        /// input face-varying-interpolated data descriptor
        OsdVertexBufferDescriptor inDesc;

        /// output face-varying-interpolated data descriptor
        OsdVertexBufferDescriptor outDesc;

        /// output face-varying-interpolated data stream and parametric derivative streams
        OutputDataStream out;

        /// Binds the face-varying-interpolated data streams
        ///
        /// Note : currently we only support bilinear boundary interpolation rules
        /// for face-varying data. Although Hbr supports 3 addition smooth rule sets,
        /// the feature-adaptive patch interpolation code currently does not support
        /// them, and neither does this EvalContext
        ///
        /// @param iDesc  data descriptor shared by all input data buffers
        ///
        /// @param oDesc  data descriptor shared by all output data buffers
        ///
        /// @param outQ   output face-varying data
        ///
        template<class OUTPUT_BUFFER>
        void Bind( OsdVertexBufferDescriptor const & iDesc,
                   OsdVertexBufferDescriptor const & oDesc, OUTPUT_BUFFER *outQ ) {
            inDesc = iDesc;

            outDesc = oDesc;
            out.Bind( outQ );
        }

        /// True if the output stream has been bound
        bool IsBound() const {
            return out.IsBound();
        }
    
        /// Unbind the vertex data streams
        void Unbind();
    };
    
    /// Returns an Eval data descriptor of the face-varying-interpolated data 
    /// currently bound to this EvalLimitContext.
    FaceVaryingData & GetFaceVaryingData() {
        return _faceVaryingData;
    }


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
    FarPatchTables::VertexValenceTable const & GetVertexValenceTable() const {
        return _vertexValenceTable;
    }

    /// Returns the Quad-Offsets buffer used for Gregory patch computations
    FarPatchTables::QuadOffsetTable const & GetQuadOffsetTable() const {
        return _quadOffsetTable;
    }
    
    /// Returns the face-varying data patch table
    FarPatchTables::FVarDataTable const & GetFVarData() const {
        return _fvarData;
    }
    
    /// Returns the number of floats in a datum of the face-varying data table
    int GetFVarWidth() const {
        return _fvarwidth;
    }

    /// Returns a map that can connect a faceId to a list of children patches
    FarPatchMap const & GetPatchMap() const {
        return *_patchMap;
    }

    /// Returns the highest valence of the vertices in the buffers
    int GetMaxValence() const {
        return _maxValence;
    }

protected:
    explicit OsdCpuEvalLimitContext(FarMesh<OsdVertex> const * farmesh, bool requireFVarData);

private:

    // Topology data for a mesh
    FarPatchTables::PatchArrayVector     _patchArrays;    // patch descriptor for each patch in the mesh
    FarPatchTables::PTable               _patches;        // patch control vertices
    std::vector<FarPatchParam::BitField> _patchBitFields; // per-patch parametric info
    
    FarPatchTables::VertexValenceTable   _vertexValenceTable; // extra Gregory patch data buffers
    FarPatchTables::QuadOffsetTable      _quadOffsetTable;

    FarPatchTables::FVarDataTable        _fvarData;

    FarPatchMap * _patchMap;           // map of the sub-patches given a face index

    VertexData       _vertexData;      // vertex-interpolated data descriptor
    VaryingData      _varyingData;     // varying-interpolated data descriptor 
    FaceVaryingData  _faceVaryingData; // face-varying-interpolated data descriptor 

    int _maxValence, 
        _fvarwidth;
};


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_EVAL_LIMIT_CONTEXT_H */
