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

#ifndef OSDUTIL_REFINER_H
#define OSDUTIL_REFINER_H

#include "../version.h"

#include "mesh.h"

#include <string>
#include <vector>


#define HBR_ADAPTIVE

#include "../osd/vertex.h"
#include "../osd/cpuVertexBuffer.h"
#include "../osd/cpuComputeContext.h"
#include "../far/mesh.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

//----------------------------------------------------------------------------
// A simple class that wraps several OpenSubdiv classes for tessellating
// a subdivision surface into quads and extracting position and topology
// data.  Single and multithreaded CPU evaluation is supported.
//
// At initialization time this class takes polygonal mesh topology as
// vectors of ints, constructs an HbrMesh from that with topology checking
// and does uniform subdivision on that to make a FarMesh.
//
// At runtime Osd vertex buffers, compute controllers, and compute contexts
// are used for fast evaluation of the surface given the FarMesh.
//----------------------------------------------------------------------------
class OsdUtilRefiner  {
  public:

    OsdUtilRefiner();

    ~OsdUtilRefiner();    

    // Returns false on error.  If errorMessage is non-NULL it'll
    // be populated upon error.
    //
    // If successful the HbrMesh and FarMesh will be created, and
    // all variables will be populated for later calls to Refine.
    //
    bool Initialize(
       const OsdUtilSubdivTopology &topology, bool adaptive,
       std::string *errorMessage = NULL,
       OsdUtilMesh<OsdVertex>::Scheme scheme = OsdUtilMesh<OsdVertex>::SCHEME_CATMARK);

    // Fetch the topology of the post-refined mesh. The "quads" vector
    // will be filled with 4 ints per quad which index into a vector
    // of positions.
    bool GetRefinedQuads(std::vector<int>* quads,
                         std::string *errorMessage = NULL) const;

    
    // Fetch the U/V coordinates of the refined quads in the U/V space
    // of their parent coarse face
    bool GetRefinedPtexUvs(std::vector<float>* subfaceUvs,
                           std::vector<int>* ptexIndices,
                           std::string *errorMessage = NULL) const;

    // Fetch the face varying attribute values on refined quads
    // Calls through to the lower level mesh class to extract
    // face varying data from hbr.
    void GetRefinedFVData(int level, const std::vector<std::string>& names,
                          std::vector<float>* fvdata) {
        _mesh->GetRefinedFVData(level, names, fvdata);
    }

    // Const access to far mesh
    const OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>*  GetFarMesh() {
        return _farMesh;
    }
    
    const std::string &GetName();

    bool GetAdaptive() { return _adaptive; }

    OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> *GetHbrMesh();

    const OsdUtilSubdivTopology &GetTopology() const {return _mesh->GetTopology();}

    bool IsRefined() {return _isRefined;}

    int GetNumRefinedVertices() { return _numRefinedVerts;}
    int GetFirstVertexOffset() { return _firstVertexOffset;}    
           
  private:

    // If true, feature adaptive refinement is used and _farMesh
    // is populated with bspline and gregory patches.
    // if false, uniform refinement is used by subdividing the entire
    // mesh N times.
    bool _adaptive;
    
    // The next block of member variables are the OpenSubdiv meshe
    // classes used to generate the refined subdivision surface
    //
    
    // The lowest level mesh, it definies the polygonal topology
    // and is used for refinement.  
    OsdUtilMesh<OpenSubdiv::OsdVertex>* _mesh;

    // A mesh of patches (adaptive), or quads (uniform) generated
    // by performing feature adaptive or uniform subdivision on the hbrMesh.
    // Uniform and adaptive each get their own far mesh as uniform and
    // adaptive code in far result in very different meshes
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>*  _farMesh;

    // Cached counts within _farMesh
    /// XXX: Maybe not cache, get from far mesh each time?    
    int _firstVertexOffset; 
    int _firstPatchOffset;  
    
    int _numRefinedVerts;
    
    int _numUniformQuads;  // zero if adaptive = true
    int _numPatches;       // zero if adaptive = false
    
    bool _isRefined;

};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif /* OSDUTIL_REFINER_H */
