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
    /// @param coords location on the limit surface to be evaluated
    ///
    /// @param context the EvalLimitContext that the controller will evaluate
    ///
    /// @param index the index of the vertex in the output buffers bound to the 
    ///              context
    ///
    template<class VERTEX_BUFFER, class OUTPUT_BUFFER>
    int EvalLimitSample( OpenSubdiv::OsdEvalCoords const & coords, 
                         OsdCpuEvalLimitContext * context,
                         unsigned int index
                       ) {    

        if (not context)
            return 0;
            
        int n = _EvalLimitSample( coords, context, index );
        
        return n;
    }

private:

    int _EvalLimitSample( OpenSubdiv::OsdEvalCoords const & coords, 
                          OsdCpuEvalLimitContext const * context,
                          unsigned int index );

};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_EVAL_LIMIT_CONTROLLER_H */
