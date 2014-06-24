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
#include "topology.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdUtilSubdivTopology::OsdUtilSubdivTopology():
    name("noname"),
    numVertices(0),
    refinementLevel(2)  // arbitrary, start with a reasonable subdivision level
{
}

OsdUtilSubdivTopology::~OsdUtilSubdivTopology()
{
}

bool
OsdUtilSubdivTopology::Initialize(
    int numVerticesParam,
    const int *nvertsParam, int numFaces,
    const int *indicesParam, int indicesLen,
    int levels,
    string *errorMessage)
{

    numVertices = numVerticesParam;
    refinementLevel = levels;

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
OsdUtilSubdivTopology::IsValid(string *errorMessage) const
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
    
    return true;
}


void
OsdUtilSubdivTopology::Print() const
{
    std::cout << "Mesh " << name << "\n";
    std::cout << "\tnumVertices = " << numVertices << "\n";
    std::cout << "\trefinementLevel = " << refinementLevel << "\n";
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



bool
OsdUtilSubdivTopology::ReadFromObjFile( char const * fname,
                                          vector<float> *pointPositions,
                                          std::string *errorMessage ) {

    FILE * handle = fopen( fname, "rt" );
    if (not handle) {
        stringstream ss;
        ss << "Could not open .obj file " << fname ;
        *errorMessage = ss.str();
        return false;
    }

    fseek( handle, 0, SEEK_END );
    size_t size = ftell(handle);
    fseek( handle, 0, SEEK_SET );

    char * shapeStr = new char[size+1];

    if ( fread( shapeStr, size, 1, handle)!=1 ) {
        stringstream ss;
        ss << "Error reading .obj file " << fname ;
        *errorMessage = ss.str();
        return false;        
    }

    fclose(handle);

    shapeStr[size]='\0';

    name = fname;
    
    return ParseFromObjString(shapeStr, 1, pointPositions );
}



static char const * sgets( char * s, int size, char ** stream ) {
    for (int i=0; i<size; ++i) {
        if ( (*stream)[i]=='\n' or (*stream)[i]=='\0') {

            memcpy(s, *stream, i);
            s[i]='\0';

            if ((*stream)[i]=='\0')
                return 0;
            else {
                (*stream) += i+1;
                return s;
            }
        }
    }
    return 0;
}

bool
OsdUtilSubdivTopology::ParseFromObjString(
    char const * shapestr, int axis,
    vector<float> *pointPositions,
    std::string * /* errorMessage */)
{
    char * str=const_cast<char *>(shapestr), line[256];
    bool done = false;
    
    while( not done ) {
        
        done = sgets(line, sizeof(line), &str)==0;
        char* end = &line[strlen(line)-1];
        if (*end == '\n')
            *end = '\0'; // strip trailing nl
        
        float x, y, z, u, v;
        switch (line[0]) {
        case 'v': switch (line[1]) {
            case ' ':
                if(sscanf(line, "v %f %f %f", &x, &y, &z) == 3) {
                    pointPositions->push_back(x);
                    switch( axis ) {
                    case 0 : pointPositions->push_back(-z);
                        pointPositions->push_back(y); break;
                    case 1 : pointPositions->push_back(y);
                        pointPositions->push_back(z); break;
                    }
                } break;
            case 't':
                if(sscanf(line, "vt %f %f", &u, &v) == 2) {
                    //XXX:gelder
                    // extract UVs
                    // s->uvs.push_back(u);
                    // s->uvs.push_back(v);
                } break;
            case 'n' :
                break; // skip normals for now
            }
            break;
        case 'f':
            if(line[1] == ' ')  {
                int vi, ti, ni;
                const char* cp = &line[2];
                while (*cp == ' ') cp++;
                int numVerts = 0, nitems=0;
                while( (nitems=sscanf(cp, "%d/%d/%d", &vi, &ti, &ni))>0) {
                    numVerts++;
                    indices.push_back(vi-1);
                    //XXX:gelder
                    // Extract face varying uvs
                    //if(nitems >= 1) s->faceuvs.push_back(ti-1);
                    //if(nitems >= 2) s->facenormals.push_back(ni-1);
                    while (*cp && *cp != ' ') cp++;
                    while (*cp == ' ') cp++;
                }
                nverts.push_back(numVerts);
            }
            break;
//        case 't' : if(line[1] == ' ') {
//                shape::tag * t = tag::parseTag( line );
//                if (t)
//                    s->tags.push_back(t);
//            } break;
        }
    }

    numVertices = (int)pointPositions->size()/3;

    return true;
}



bool OsdUtilTagData::TagTypeFromString(
    OsdUtilTagData::TagType *tagType,
    const std::string &str)
{
    if (str == "corner") {
        *tagType = OsdUtilTagData::CORNER;
    } else if  (str == "crease") {
        *tagType = OsdUtilTagData::CREASE;
    } else if  (str == "creasemethod") {
        *tagType = OsdUtilTagData::CREASE_METHOD;
    } else if  (str == "facevaryinginterpolateboundary") {
        *tagType = OsdUtilTagData::FACE_VARYING_INTERPOLATE_BOUNDARY;
    } else if  (str == "facevaryingpropagatecorners") {
        *tagType = OsdUtilTagData::FACE_VARYING_PROPOGATE_CORNERS; 
    } else if  (str == "hole") {
        *tagType = OsdUtilTagData::HOLE;
    } else if  (str == "interpolateboundary") {
        *tagType = OsdUtilTagData::INTERPOLATE_BOUNDARY;
    } else if  (str == "smoothtriangles") {
        *tagType = OsdUtilTagData::SMOOTH_TRIANGLES;
    } else if  (str == "vertexedit") {
        *tagType = OsdUtilTagData::VERTEX_EDIT;
    } else if  (str == "edgeedit") {
        *tagType = OsdUtilTagData::EDGE_EDIT;
    } else {
        return false;
    }

    return true;
}
    


// If numSharpness == 1, the single sharpness value applies for all vertices
// If numSharpness == numIndices, there is a per-vertex sharpness
bool
OsdUtilTagData::AddCorner(int *indices, int numIndices,
                            float *sharpness, int numSharpness)
{
    tags.push_back(CORNER);
    numArgs.push_back(numIndices);
    numArgs.push_back(numSharpness);
    numArgs.push_back(0); 
    for (int i=0; i<numIndices; ++i) {
        intArgs.push_back(indices[i]);
    }
    for (int i=0; i<numSharpness; ++i) {
        floatArgs.push_back(sharpness[i]);
    }

    return true;
}

// Indices is a sequential series of mesh vertex indices that bound
// the edges to be tagged with sharpness.  
//
// If numSharpness == 1, the single sharpness value applies for all edges
// If numSharpness == numIndices, there is a per-edge sharpness that
// will be interpolated along the crease
bool
OsdUtilTagData::AddCrease(int *indices, int numIndices,
                            float *sharpness, int numSharpness)
{
    tags.push_back(CREASE);
    numArgs.push_back(numIndices);
    numArgs.push_back(numSharpness);
    numArgs.push_back(0); 
    for (int i=0; i<numIndices; ++i) {
        intArgs.push_back(indices[i]);
    }
    for (int i=0; i<numSharpness; ++i) {
        floatArgs.push_back(sharpness[i]);
    }

    return true;
}
        
// Either "normal" or "chaikin"
bool
OsdUtilTagData::AddCreaseMethod(const std::string & /* value */)
{
    return true;
}

// 0 == OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryNone
// 1 == OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeAndCorner
// 2 == OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly    
bool
OsdUtilTagData::AddInterpolateBoundary(int value )
{
    tags.push_back(INTERPOLATE_BOUNDARY);
    numArgs.push_back(1);
    numArgs.push_back(0);
    numArgs.push_back(0);    
    intArgs.push_back(value);

    return true;
}

bool
OsdUtilTagData::AddHole(int *indices, int numIndices)
{
    tags.push_back(HOLE);
    numArgs.push_back(numIndices);
    numArgs.push_back(0);
    numArgs.push_back(0);
    for (int i=0; i<numIndices; ++i) {
        intArgs.push_back(indices[i]);
    }

    return true;
}



bool
OsdUtilSubdivTopology::WriteObjFile(
    const char *filename,
    const float *positions,
    std::string *errorMessage)
{
    
    ofstream file;
    
    file.open (filename);

    if (not file.is_open()) {
        stringstream ss;
        ss << "Could not open .obj file " << filename ;
        *errorMessage = ss.str();
        return false;
    }
    

    for (int i=0; i<numVertices*3; i+=3) {
        file << "v " << positions[i] << " " << positions[i+1]
             << " " << positions[i+2] <<"\n";
    }

    file << "\n";

    int idx = 0;
    for (int i=0; i<(int)nverts.size(); ++i) {
        file << "f";
        for (int j=0; j<nverts[i]; ++j) {
            file << " " << indices[idx+j]+1;
        }
        idx += nverts[i];
        file << "\n";
    }

    file << "\n";    

    file.close();

    return true;
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
