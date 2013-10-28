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

#ifndef PX_OSD_UTIL_MESH_H
#define PX_OSD_UTIL_MESH_H

#include <hbr/mesh.h>
#include <osd/vertex.h>

#include <string>
#include <map>


// A value struct that holds annotations on a subdivision surface
// such as creases, boundaries, holes, corners, hierarchical edits, etc.
//
// For OpenSubdiv documentation on tags, see:
// See http://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html#hierarchical-edits
//
struct PxOsdUtilTagData {
    std::vector<std::string> tags;
    std::vector<int> numArgs;
    std::vector<int> intArgs;
    std::vector<float> floatArgs;
    std::vector<std::string> stringArgs;
};

// A value struct intended to hold within it topology for the base mesh
// of a subdivision surface, and any annotation tags.
// It is used to initialize classes that create and operate on subdivs.
//
class PxOsdUtilSubdivTopology {
  public:

    PxOsdUtilSubdivTopology();
    ~PxOsdUtilSubdivTopology();    

    // XXX Would be great for these members to be private with accessors
    std::string name;
    int numVertices;
    int maxLevels;
    std::vector<int> indices;
    std::vector<int> nverts;
    std::vector<std::string> vvNames;
    std::vector<std::string> fvNames;
    std::vector<float> fvData;
    PxOsdUtilTagData tagData;


    // Initialize using raw types. 
    //
    // This is useful for automated tests initializing with data like:
    // int nverts[] = { 4, 4, 4, 4, 4, 4};
    //
    bool Initialize(
        int numVertices,
        const int *nverts, int numFaces,
        const int *indices, int indicesLen,
        int levels,
        std::string *errorMessage);

    // checks indices etc to ensure that mesh isn't in a
    // broken state. Returns false on error, and will populate
    // errorMessage (if non-NULL) with a descriptive error message
    bool IsValid(std::string *errorMessage = NULL) const;

    // for debugging, print the contents of the topology to stdout
    void Print() const;
};


// This class is reponsible for taking a topological description of a mesh
// defined by PxOsdUtilSubdivTopology and turn that into a halfedge mesh
// with detailed connectivity information for mesh traversal. A PxOsdUtilMesh
// object is used for uniform and feature adaptive refinement of subdivision
// surfaces (subdivs), which is itself a requirement for fast run-time
// evaluation of subdivs.
//
class PxOsdUtilMesh {
  public:
    
    PxOsdUtilMesh(
        const PxOsdUtilSubdivTopology &topology,
        std::string *errorMessage = NULL);

    ~PxOsdUtilMesh();

    // Fetch the face varying attribute values on refined quads
    // Traverse the hbrMesh gathering face varying data created
    // by a refiner.
    // XXX: this assumes uniform subdivision, should be moved
    // into uniformRefiner?
    void GetRefinedFVData(int subdivisionLevel,
                          const std::vector<std::string>& names,
                          std::vector<float>* fvdata);
    
    OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> *GetHbrMesh() { return _hmesh;}

    bool IsValid() { return _valid;}

    const std::string &GetName() { return _name;}

    const PxOsdUtilSubdivTopology &GetTopology() const {return _t;}

private:

    const PxOsdUtilSubdivTopology &_t;

    std::vector<int> _fvarwidths;
    std::vector<int> _fvarindices;
    std::map<std::string, int> _fvaroffsets;

    OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> *_hmesh;

    std::string _name;

    bool _valid;
    
    static void _ProcessTagsAndFinishMesh(
        OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> *mesh,
        const std::vector<std::string> &tags,
        const std::vector<int> &numArgs,
        const std::vector<int> &intArgs,
        const std::vector<float> &floatArgs,
        const std::vector<std::string> &stringArgs);
};


#endif /* PX_OSD_UTIL_MESH_H */
