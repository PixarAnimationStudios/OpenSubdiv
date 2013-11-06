#include "topology.h"

#include <sstream>
#include <iostream>

using namespace std;

PxOsdUtilSubdivTopology::PxOsdUtilSubdivTopology():
    name("noname"),
    numVertices(0),
    maxLevels(2)  // arbitrary, start with a reasonable subdivision level
{
    std::cout << "Creating subdiv topology object\n";    
}

PxOsdUtilSubdivTopology::~PxOsdUtilSubdivTopology()
{
    std::cout << "Destroying subdiv topology object\n";
}

bool
PxOsdUtilSubdivTopology::Initialize(
    int numVerticesParam,
    const int *nvertsParam, int numFaces,
    const int *indicesParam, int indicesLen,
    int levels,
    string *errorMessage)
{

    numVertices = numVerticesParam;
    maxLevels = levels;

    nverts.resize(numFaces);
    for (int i=0; i<numFaces; ++i) {
        nverts[i] = nvertsParam[i];
    }
        
    indices.resize(indicesLen);    
    for (int i=0; i<indicesLen; ++i) {
        indices[i] = indicesParam[i];
    }

    return IsValid(errorMessage);
}


bool
PxOsdUtilSubdivTopology::IsValid(string *errorMessage) const
{
    if (numVertices == 0) {
        if (errorMessage) {
            stringstream ss;
            ss << "Topology " << name << " has no vertices";
            *errorMessage = ss.str();
        }
        return false;
    }
    
    for (int i=0; i<(int)indices.size(); ++i) {
        if ((indices[i] < 0) or
            (indices[i] >= numVertices)) {
            if (errorMessage) {
                stringstream ss;
                ss << "Topology " << name << " has bad index " << indices[i] << " at index " << i;
                *errorMessage = ss.str();
            }
            return false;
        }

    }

    int totalNumIndices = 0;
    for (int i=0; i< (int)nverts.size(); ++i) {
        if (nverts[i] < 1) {
            if (errorMessage) {
                stringstream ss;
                ss << "Topology " << name << " has bad nverts " << nverts[i] << " at index " << i;
                *errorMessage = ss.str();
            }
            return false;            
        }
        totalNumIndices += nverts[i];
    }
    
    if (totalNumIndices != (int)indices.size()) {
        if (errorMessage) {
            *errorMessage = "Bad indexing for face topology";
        }
  
        return false;
    }
    
    std::cout << "\n";

    return true;
}


void
PxOsdUtilSubdivTopology::Print() const
{
    std::cout << "Mesh " << name << "\n";
    std::cout << "\tnumVertices = " << numVertices << "\n";
    std::cout << "\tmaxLevels = " << maxLevels << "\n";
    std::cout << "\tindices ( " << indices.size() << ") : ";
    for (int i=0; i<(int)indices.size(); ++i) {
        std::cout << indices[i] << ", ";
    }
    std::cout << "\n";

    std::cout << "\tnverts ( " << nverts.size() << ") : ";
    for (int i=0; i<(int)nverts.size(); ++i) {
        std::cout << nverts[i] << ", ";
    }
    std::cout << "\n";    

}

