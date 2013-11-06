#ifndef PX_OSD_UTIL_TOPOLOGY_H
#define PX_OSD_UTIL_TOPOLOGY_H

#include <vector>
#include <string>



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



#endif /* PX_OSD_UTIL_TOPOLOGY_H */
