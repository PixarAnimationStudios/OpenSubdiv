//
//   Copyright 2018 DreamWorks Animation LLC.
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


//------------------------------------------------------------------------------
// Tutorial description:
//
// This tutorial illustrates two different styles of defining classes for
// interpolating primvar data with the template methods in Far.  The most
// common usage involves data of a fixed size, so the focus here is on an
// alternative supporting variable length data.
//

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>

#include <cstdio>

using namespace OpenSubdiv;

//
//  Vertex data containers for interpolation:
//      - Coord3 is fixed to support 3 floats
//      - Coord2 is fixed to support 2 floats
//      - CoordBuffer can support a specified number of floats
//
struct Coord3 {
    Coord3() { }
    Coord3(float x, float y, float z) { _xyz[0] = x, _xyz[1] = y, _xyz[2] = z; }

    void Clear() { _xyz[0] = _xyz[1] = _xyz[2] = 0.0f; }

    void AddWithWeight(Coord3 const & src, float weight) {
        _xyz[0] += weight * src._xyz[0];
        _xyz[1] += weight * src._xyz[1];
        _xyz[2] += weight * src._xyz[2];
    }

    float const * Coords() const { return &_xyz[0]; }

private:
    float _xyz[3];
};

struct Coord2 {
    Coord2() { }
    Coord2(float u, float v) { _uv[0] = u, _uv[1] = v; }

    void Clear() { _uv[0] = _uv[1] = 0.0f; }

    void AddWithWeight(Coord2 const & src, float weight) {
        _uv[0] += weight * src._uv[0];
        _uv[1] += weight * src._uv[1];
    }

    float const * Coords() const { return &_uv[0]; }

private:
    float _uv[2];
};

struct CoordBuffer {
    //
    //  The head of an external buffer and stride is specified on construction:
    //
    CoordBuffer(float * data, int size) : _data(data), _size(size) { }
    CoordBuffer() : _data(0), _size(0) { }

    void Clear() {
        for (int i = 0; i < _size; ++i) {
            _data[i] = 0.0f;
        }
    }

    void AddWithWeight(CoordBuffer const & src, float weight) {
        assert(src._size == _size);
        for (int i = 0; i < _size; ++i) {
            _data[i] += weight * src._data[i];
        }
    }

    float const * Coords() const { return _data; }

    //
    //  Defining [] to return a location elsewhere in the buffer is the key
    //  requirement to supporting interpolatible data of varying size
    //
    CoordBuffer operator[](int index) const {
        return CoordBuffer(_data + index * _size, _size);
    }

private:
    float * _data;
    int     _size;
};

//
//  Global cube geometry from catmark_cube.h
//
//  Topology:
static int g_nverts = 8;
static int g_nfaces = 6;

static int g_vertsperface[6] = { 4, 4, 4, 4, 4, 4 };

static int g_vertIndices[24] = { 0, 1, 3, 2,
                                 2, 3, 5, 4,
                                 4, 5, 7, 6,
                                 6, 7, 1, 0,
                                 1, 7, 5, 3,
                                 6, 0, 2, 4  };
//  Primvar data:
static float g_verts[8][3] = {{  0.0f,  0.0f,  1.0f },
                              {  1.0f,  0.0f,  1.0f },
                              {  0.0f,  1.0f,  1.0f },
                              {  1.0f,  1.0f,  1.0f },
                              {  0.0f,  1.0f,  0.0f },
                              {  1.0f,  1.0f,  0.0f },
                              {  0.0f,  0.0f,  0.0f },
                              {  1.0f,  0.0f,  0.0f }};

//
//  Creates Far::TopologyRefiner from raw geometry above (see tutorial_1_1 for
//  more details)
//
static Far::TopologyRefiner *
createFarTopologyRefiner() {

    typedef Far::TopologyDescriptor Descriptor;

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    Descriptor desc;
    desc.numVertices  = g_nverts;
    desc.numFaces     = g_nfaces;
    desc.numVertsPerFace = g_vertsperface;
    desc.vertIndicesPerFace  = g_vertIndices;

    // Instantiate a Far::TopologyRefiner from the descriptor
    Far::TopologyRefiner * refiner =
            Far::TopologyRefinerFactory<Descriptor>::Create(desc,
                    Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    return refiner;
}

//
//  Overview of main():
//      - create a Far::TopologyRefiner and uniformly refine it
//      - allocate separate and combined data buffers for vertex positions and UVs
//      - populate all refined data buffers and compare results
//      - write the result in Obj format
//
//  Disable warnings for exact floating point comparisons:
#ifdef __INTEL_COMPILER
#pragma warning disable 1572
#endif

int main(int, char **) {

    //  Instantiate a Far::TopologyRefiner from the global geometry:
    Far::TopologyRefiner * refiner = createFarTopologyRefiner();

    //  Uniformly refine the topology up to 'maxlevel'
    int maxlevel = 2;

    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

    //  Allocate and populate data buffers for vertex primvar data -- positions and
    //  UVs. We assign UV coordiantes by simply projecting/assigning XY values.
    //  The position and UV buffers use their associated data types, while the
    //  combined buffer uses 5 floats per vertex.
    //
    int numBaseVertices  = g_nverts;
    int numTotalVertices = refiner->GetNumVerticesTotal();

    std::vector<Coord3> posData(numTotalVertices);
    std::vector<Coord2> uvData(numTotalVertices);

    int                 combinedStride = 3 + 2;
    std::vector<float>  combinedData(numTotalVertices * combinedStride);

    for (int i = 0; i < numBaseVertices; ++i) {
        posData[i] = Coord3(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
        uvData[i]  = Coord2(g_verts[i][0], g_verts[i][1]);

        float * coordCombined = &combinedData[i * combinedStride];
        coordCombined[0] = g_verts[i][0];
        coordCombined[1] = g_verts[i][1];
        coordCombined[2] = g_verts[i][2];
        coordCombined[3] = g_verts[i][0];
        coordCombined[4] = g_verts[i][1];
    }

    //  Interpolate vertex primvar data
    Far::PrimvarRefiner primvarRefiner(*refiner);

    Coord3 * posSrc = &posData[0];
    Coord2 * uvSrc  = & uvData[0];

    CoordBuffer combinedSrc(&combinedData[0], combinedStride);

    for (int level = 1; level <= maxlevel; ++level) {
        int numLevelVerts = refiner->GetLevel(level-1).GetNumVertices();

        Coord3 * posDst = posSrc + numLevelVerts;
        Coord2 * uvDst  = uvSrc + numLevelVerts;

        CoordBuffer combinedDst = combinedSrc[numLevelVerts];

        primvarRefiner.Interpolate(level, posSrc, posDst);
        primvarRefiner.Interpolate(level, uvSrc, uvDst);
        primvarRefiner.Interpolate(level, combinedSrc, combinedDst);

        posSrc = posDst;
        uvSrc = uvDst;
        combinedSrc = combinedDst;
    }

    //  Verify that the combined coords match the separate results:
    for (int i = numBaseVertices; i < numTotalVertices; ++i) {
        float const * posCoords = posData[i].Coords();
        float const * uvCoords  = uvData[i].Coords();

        float const * combCoords = &combinedData[combinedStride * i];

        assert(combCoords[0] == posCoords[0]);
        assert(combCoords[1] == posCoords[1]);
        assert(combCoords[2] == posCoords[2]);
        assert(combCoords[3] == uvCoords[0]);
        assert(combCoords[4] == uvCoords[1]);
    }

    //
    //  Output OBJ of the highest level refined:
    //
    Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

    int firstOfLastVerts = numTotalVertices - refLastLevel.GetNumVertices();

    //  Print vertex positions
    printf("#  Vertices:\n");
    for (int vert = firstOfLastVerts; vert < numTotalVertices; ++vert) {
        float const * pos = &combinedData[vert * combinedStride];
        printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
    }

    printf("#  UV coordinates:\n");
    for (int vert = firstOfLastVerts; vert < numTotalVertices; ++vert) {
        float const * uv = &combinedData[vert * combinedStride] + 3;
        printf("vt %f %f\n", uv[0], uv[1]);
    }

    //  Print faces
    int numFaces = refLastLevel.GetNumFaces();

    printf("#  Faces:\n");
    for (int face = 0; face < numFaces; ++face) {
        Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

        printf("f ");
        for (int fvert = 0; fvert < fverts.size(); ++fvert) {
            int objIndex = 1 + fverts[fvert]; // OBJ uses 1-based arrays...
            printf("%d/%d ", objIndex, objIndex);
        }
        printf("\n");
    }

    delete refiner;
    return EXIT_SUCCESS;
}
