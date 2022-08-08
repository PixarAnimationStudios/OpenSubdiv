//
//   Copyright 2021 Pixar
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

#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cassert>

//  Utilities local to this tutorial:
namespace tutorial {

//
//  Simple class to write vertex positions, normals and faces to a
//  specified Obj file:
//
class ObjWriter {
public:
    ObjWriter(std::string const &filename = 0);
    ~ObjWriter();

    int GetNumVertices() const { return _numVertices; }
    int GetNumFaces()    const { return _numFaces; }

    void WriteVertexPositions(std::vector<float> const & p, int size = 3);
    void WriteVertexNormals(std::vector<float> const & du,
                            std::vector<float> const & dv);
    void WriteVertexUVs(std::vector<float> const & uv);

    void WriteFaces(std::vector<int> const & faceVertices, int faceSize,
                    bool writeNormalIndices = false,
                    bool writeUVIndices = false);

    void WriteGroupName(char const * prefix, int index);

private:
    void getNormal(float N[3], float const du[3], float const dv[3]) const;

private:
    std::string _filename;
    FILE *      _fptr;

    int _numVertices;
    int _numNormals;
    int _numUVs;
    int _numFaces;
};


//
//  Definitions ObjWriter methods:
//
ObjWriter::ObjWriter(std::string const &filename) :
        _fptr(0), _numVertices(0), _numNormals(0), _numUVs(0), _numFaces(0) {

    if (filename != std::string()) {
        _fptr = fopen(filename.c_str(), "w");
        if (_fptr == 0) {
            fprintf(stderr, "Error:  ObjWriter cannot open Obj file '%s'\n",
                filename.c_str());
        }
    }
    if (_fptr == 0) _fptr = stdout;
}

ObjWriter::~ObjWriter() {

    if (_fptr != stdout) fclose(_fptr);
}

void
ObjWriter::WriteVertexPositions(std::vector<float> const & pos, int dim) {

    assert(dim >= 2);
    int numNewVerts = (int)pos.size() / dim;

    float const * P = pos.data();
    for (int i = 0; i < numNewVerts; ++i, P += dim) {
        if (dim == 2) {
            fprintf(_fptr, "v %f %f 0.0\n", P[0], P[1]);
        } else {
            fprintf(_fptr, "v %f %f %f\n", P[0], P[1], P[2]);
        }
    }
    _numVertices += numNewVerts;
}

void
ObjWriter::getNormal(float N[3], float const du[3], float const dv[3]) const {

    N[0] = du[1] * dv[2] - du[2] * dv[1];
    N[1] = du[2] * dv[0] - du[0] * dv[2];
    N[2] = du[0] * dv[1] - du[1] * dv[0];

    float lenSqrd = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
    if (lenSqrd <= 0.0f) {
        N[0] = 0.0f;
        N[1] = 0.0f;
        N[2] = 0.0f;
    } else {
        float lenInv = 1.0f / std::sqrt(lenSqrd);
        N[0] *= lenInv;
        N[1] *= lenInv;
        N[2] *= lenInv;
    }
}

void
ObjWriter::WriteVertexNormals(std::vector<float> const & du,
                              std::vector<float> const & dv) {

    assert(du.size() == dv.size());
    int numNewNormals = (int)du.size() / 3;

    float const * dPdu = &du[0];
    float const * dPdv = &dv[0];
    for (int i = 0; i < numNewNormals; ++i, dPdu += 3, dPdv += 3) {
        float N[3];
        getNormal(N, dPdu, dPdv);
        fprintf(_fptr, "vn %f %f %f\n", N[0], N[1], N[2]);
    }
    _numNormals += numNewNormals;
}

void
ObjWriter::WriteVertexUVs(std::vector<float> const & uv) {

    int numNewUVs = (int)uv.size() / 2;

    for (int i = 0; i < numNewUVs; ++i) {
        fprintf(_fptr, "vt %f %f\n", uv[i*2], uv[i*2+1]);
    }
    _numUVs += numNewUVs;
}

void
ObjWriter::WriteFaces(std::vector<int> const & faceVertices, int faceSize,
                      bool includeNormalIndices, bool includeUVIndices) {

    int numNewFaces = (int)faceVertices.size() / faceSize;

    int const * v = &faceVertices[0];
    for (int i = 0; i < numNewFaces; ++i, v += faceSize) {
        fprintf(_fptr, "f ");
        for (int j = 0; j < faceSize; ++j) {
            if (v[j] >= 0) {
                //  Remember Obj indices start with 1:
                int vIndex = 1 + v[j];

                if (includeNormalIndices && includeUVIndices) {
                    fprintf(_fptr, " %d/%d/%d", vIndex, vIndex, vIndex);
                } else if (includeNormalIndices) {
                    fprintf(_fptr, " %d//%d", vIndex, vIndex);
                } else if (includeUVIndices) {
                    fprintf(_fptr, " %d/%d", vIndex, vIndex);
                } else {
                    fprintf(_fptr, " %d", vIndex);
                } 
            }
        }
        fprintf(_fptr, "\n");
    }
    _numFaces += numNewFaces;
}

void
ObjWriter::WriteGroupName(char const * prefix, int index) {

    fprintf(_fptr, "g %s%d\n", prefix ? prefix : "", index);
}

} // end namespace
