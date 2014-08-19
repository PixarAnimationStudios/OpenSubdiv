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

#ifndef OSD_UTIL_MESH_H
#define OSD_UTIL_MESH_H

#include "../version.h"

#include "topology.h"
#include <string>
#include <map>

// We forward declare the HbrMesh templated class here to avoid including
// hbr headers.  Keep the hbr inclusions in the .cpp files to avoid
// conflicts with the amber/lib version of hbr and to reduce included code
// complexity.
//
// Note that a client who wants to access the HbrMesh should include hbr/mesh.h
// like this:
// #define HBR_ADAPTIVE
// #include "../hbr/mesh.h"
//
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {
    template <class T> class HbrMesh;
  }
}



namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {
    
// This class is reponsible for taking a topological description of a mesh
// defined by OsdUtilSubdivTopology and turn that into a halfedge mesh
// with detailed connectivity information for mesh traversal. A OsdUtilMesh
// object is used for uniform and feature adaptive refinement of subdivision
// surfaces (subdivs), which is itself a requirement for fast run-time
// evaluation of subdivs.
//
template <class T>
class OsdUtilMesh {
  public:
    enum Scheme {
      SCHEME_CATMARK,
      SCHEME_BILINEAR,
      SCHEME_LOOP,
    };

    OsdUtilMesh();

    bool Initialize(
        const OsdUtilSubdivTopology &topology,
        std::string *errorMessage = NULL,
        Scheme scheme = SCHEME_CATMARK);

    ~OsdUtilMesh();

    // Fetch the face varying attribute values on refined quads
    // Traverse the hbrMesh gathering face varying data created
    // by a refiner.
    // XXX: this assumes uniform subdivision, should be moved
    // into uniformRefiner?
    void GetRefinedFVData(int level,
                          const std::vector<std::string>& names,
                          std::vector<float>* fvdata);
    
    OpenSubdiv::OPENSUBDIV_VERSION::HbrMesh<T> *GetHbrMesh() { return _hmesh;}

    bool IsValid() { return _valid;}

    const std::string &GetName() { return _t.name;}

    const OsdUtilSubdivTopology &GetTopology() const {return _t;}

private:

    OsdUtilSubdivTopology _t;

    std::vector<int> _fvarwidths;
    std::vector<int> _fvarindices;
    std::map<std::string, int> _fvaroffsets;

    OpenSubdiv::OPENSUBDIV_VERSION::HbrMesh<T> *_hmesh;

    bool _valid;
    
};


}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv
    

#endif /* OSD_UTIL_MESH_H */
