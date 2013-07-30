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
#ifndef OSD_CPU_EVAL_LIMIT_CONTROLLER_H
#define OSD_CPU_EVAL_LIMIT_CONTROLLER_H

#include "../version.h"

#include "../osd/evalLimitContext.h"
#include "../osd/cpuEvalLimitContext.h"
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief CPU controler for limit surface evaluation.
///
/// A CPU-driven controller that can be called to evaluate samples on the limit
/// surface for a given EvalContext.
///
class OsdCpuEvalLimitController {

public:
    /// Constructor.
    OsdCpuEvalLimitController();

    /// Destructor.
    ~OsdCpuEvalLimitController();
    
    /// \brief Represents a location on the limit surface
    struct OsdEvalCoords {
        int face;   // Ptex unique face ID
        float u,v;  // local u,v
    };
    
    /// \brief Vertex interpolation of samples at the limit
    ///
    /// Evaluates "vertex" interpolation of a sample on the surface limit.
    ///
    /// Warning : this function is re-entrant but it breaks the Osd API pattern
    /// by requiring the client code to bind and unbind the vertex buffers to
    /// the EvalLimitContext. 
    ///
    /// Ex :
    /// \code
    /// evalCtxt->BindVertexBuffers( ... );
    ///
    /// parallel_for( int index=0; i<nsamples; ++index ) {
    ///    evalCtrlr->EvalLimitSample( coord, evalCtxt, index );
    /// }
    ///
    /// evalCtxt->UnbindVertexBuffers();
    /// \endcode
    ///
    /// @param coords   location on the limit surface to be evaluated
    ///
    /// @param context  the EvalLimitContext that the controller will evaluate
    ///
    /// @param index    the index of the vertex in the output buffers bound to the 
    ///                 context
    ///
    /// @return the number of samples found (0 if the location was tagged as a hole
    ///         or the coordinate was invalid)
    ///
    template<class VERTEX_BUFFER, class OUTPUT_BUFFER>
    int EvalLimitSample( OpenSubdiv::OsdEvalCoords const & coords, 
                         OsdCpuEvalLimitContext * context,
                         unsigned int index ) {    
        if (not context)
            return 0;
            
        int n = _EvalLimitSample( coords, context, index );
        
        return n;
    }

private:

    int _EvalLimitSample( OpenSubdiv::OsdEvalCoords const & coords, 
                          OsdCpuEvalLimitContext * context,
                          unsigned int index );

};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_EVAL_LIMIT_CONTROLLER_H */
