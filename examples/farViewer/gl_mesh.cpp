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

#include "glLoader.h"

#include "gl_mesh.h"
#include "gl_fontutils.h"

#include "../common/patchColors.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

//------------------------------------------------------------------------------
// color palettes
static float         g_solidColor[4] = {1.0f,  1.0f,  1.0f, 1.0f},
                   g_ambientColor[4] = {0.1f,  0.1f,  0.1f, 1.0f};

static float    g_levelColors[10][4] = {{1.0f,  1.0f,  1.0f},
                                        {1.0f,  1.0f,  0.0f},
                                        {1.0f,  0.5f,  0.0f},
                                        {0.8f,  0.0f,  0.0f},
                                        {0.0f,  1.0f,  0.5f},
                                        {0.0f,  1.0f,  1.0f},
                                        {0.0f,  0.5f,  1.0f},
                                        {0.0f,  0.5f,  0.5f},
                                        {0.5f,  0.0f,  1.0f},
                                        {1.0f,  0.5f,  1.0f}};

static float g_parentTypeColors[4][4] = {{0.9f,  0.9f,  0.9f},
                                         {0.4f,  0.8f,  0.4f},
                                         {0.8f,  0.8f,  0.4f},
                                         {0.8f,  0.4f,  0.4f}};

//------------------------------------------------------------------------------
void
GLMesh::setSolidColor(float * color) {

    color[0] = _diffuseColor[0];
    color[1] = _diffuseColor[1];
    color[2] = _diffuseColor[2];
}

void
GLMesh::setColorByLevel(int level, float * color) {

    color[0] = g_levelColors[level][0];
    color[1] = g_levelColors[level][1];
    color[2] = g_levelColors[level][2];
}

void
GLMesh::setColorBySharpness(float sharpness, float * color) {

    //  0.0       2.0       4.0
    // green --- yellow --- red
    color[0] = std::min(1.0f, sharpness * 0.5f);
    color[1] = std::min(1.0f, 2.0f - sharpness * 0.5f);
    color[2] = 0;
}

//------------------------------------------------------------------------------
static GLuint g_faceTexture=0;

static GLuint
getFaceTexture() {

#include "face_texture.h"

    if (! g_faceTexture) {
        glGenTextures(1, &g_faceTexture);
        glBindTexture(GL_TEXTURE_2D, g_faceTexture);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FACE_TEXTURE_WIDTH,
            FACE_TEXTURE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, face_texture);
    }

    return g_faceTexture;
}


//------------------------------------------------------------------------------
GLMesh::GLMesh() : _TBOfaceColors(0) {

    for (int i=0; i<COMP_NUM_COMPONENTS; ++i) {
        _VAO[i]=0;
        _VBO[i]=0;
        _EAO[i]=0;

        _numComps[i] = 0;
    }

    memcpy(_ambientColor, g_ambientColor, sizeof(float)*4);
    memcpy(_diffuseColor, g_solidColor, sizeof(float)*4);
}

GLMesh::~GLMesh() {

    for (int i=0; i<COMP_NUM_COMPONENTS; ++i) {
        if (_VAO[i]) {
            glDeleteVertexArrays(1, &_VAO[i]);
        }
        if (_VBO[i]) {
            glDeleteBuffers(1, &_VBO[i]);
        }
        if (_EAO[i]) {
            glDeleteBuffers(1, &_EAO[i]);
        }
    }
}

//------------------------------------------------------------------------------
void
GLMesh::initializeVertexComponentBuffer(float const * vertData, int nverts) {

    std::vector<float> & vbo = _vbo[COMP_VERT];
    vbo.resize(nverts * 6);

    std::vector<int> & eao = _eao[COMP_VERT];
    eao.resize(nverts);

    for (int vert=0; vert<nverts; ++vert) {

        // copy positions
        memcpy(&vbo[vert*6], &vertData[vert*3], 3*sizeof(float));

        // populate EAO
        eao[vert] = vert;
    }
}

//------------------------------------------------------------------------------
void
GLMesh::Initialize(Options /* options */,
    int nverts, int nfaces, int * vertsperface, int * faceverts,
        float const * vertexData) {


    { // vertex color component ----------------------------

        initializeVertexComponentBuffer(vertexData, nverts);

        std::vector<float> & vbo = _vbo[COMP_VERT];

        for (int vert=0, ofs=3; vert<nverts; ++vert, ofs+=6) {
            setSolidColor(&vbo[ofs]);
        }
    }
    { // edge color component ------------------------------
        int nedges = nfaces;

        std::vector<float> & vbo = _vbo[COMP_EDGE];
        vbo.resize(nedges * 2 * 6);

        std::vector<int> & eao = _eao[COMP_EDGE];
        eao.resize(nedges*2);

        for (int edge=0; edge<nedges; ++edge) {

            // edge mode expects faces with 2 verts (aka edges) as input
            assert(vertsperface[edge]==2);

            eao[edge*2  ] = edge*2;
            eao[edge*2+1] = edge*2+1;

            int const * verts = &faceverts[edge*2];

            float * v0 = &vbo[edge*2*6],
                  * v1 = v0+6;

            // copy position
            memcpy(v0, vertexData + verts[0]*3, sizeof(float)*3);
            memcpy(v1, vertexData + verts[1]*3, sizeof(float)*3);

            // default to solid color
            setSolidColor(v0+3);
            setSolidColor(v1+3);
        }
    }
    { // face component ------------------------------------

        std::vector<float> & vbo = _vbo[COMP_FACE];
        vbo.resize(nverts * 3);

        memcpy(&vbo[0], vertexData, nverts*sizeof(float)*3);

        int nfaceverts = 0;
        for (int i=0; i<nfaces; ++i) {
            nfaceverts += vertsperface[i];
        }

        std::vector<int> & eao = _eao[COMP_FACE];
        eao.resize(nfaceverts);

        _faceColors.resize(nfaces*4);

        int const * fverts = faceverts;
        for (int face=0, ofs=0; face<nfaces; ++face) {

            int nverts = vertsperface[face];
            for (int vert=0; vert<nverts; ++vert) {
                eao[ofs++] = fverts[vert];
            }

            setSolidColor(&_faceColors[face*4]);

            fverts += nverts;
        }
    }

    _numComps[COMP_FACE] = (int)_eao[COMP_FACE].size();
    _numComps[COMP_EDGE] = (int)_eao[COMP_EDGE].size();
    _numComps[COMP_VERT] = (int)_eao[COMP_VERT].size();

    InitializeDeviceBuffers();
}

//------------------------------------------------------------------------------
void
GLMesh::Initialize(Options options, TopologyRefiner const & refiner,
    PatchTable const * patchTable, float const * vertexData) {

    if (patchTable) {
        initializeBuffers(options, refiner, *patchTable, vertexData);
    } else {
        initializeBuffers(options, refiner, vertexData);
    }

    _numComps[COMP_FACE] = (int)_eao[COMP_FACE].size();
    _numComps[COMP_EDGE] = (int)_eao[COMP_EDGE].size();
    _numComps[COMP_VERT] = (int)_eao[COMP_VERT].size();

    //InitializeDeviceBuffers();
}

//------------------------------------------------------------------------------
void
GLMesh::initializeBuffers(Options options,
    TopologyRefiner const & refiner, float const * vertexData) {

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    int maxlevel = refiner.GetMaxLevel();

    OpenSubdiv::Far::TopologyLevel const & refLastLevel = refiner.GetLevel(maxlevel);

    int nverts = refLastLevel.GetNumVertices(),
        nedges = refLastLevel.GetNumEdges(),
        nfaces = refLastLevel.GetNumFaces(),
        firstvert = 0;


    for (int i=0; i<maxlevel; ++i) {
        firstvert += refiner.GetLevel(i).GetNumVertices();
    }

    float const * vertData =  &vertexData[firstvert*3];

    { // vertex color component ----------------------------

        initializeVertexComponentBuffer(vertData, nverts);

        std::vector<float> & vbo = _vbo[COMP_VERT];

        // set colors
        if (options.vertColorMode==VERTCOLOR_BY_LEVEL) {

            for (int level=0, ofs=3; level<=maxlevel; ++level) {
                for (int vert=0; vert<refiner.GetLevel(level).GetNumVertices(); ++vert, ofs+=6) {
                    setColorByLevel(level, &vbo[ofs]);
                }
            }
        } else if (options.vertColorMode==VERTCOLOR_BY_SHARPNESS) {

            for (int vert=0, ofs=3; vert<refLastLevel.GetNumVertices(); ++vert, ofs+=6) {
               setColorBySharpness(refLastLevel.GetVertexSharpness(vert), &vbo[ofs]);
            }
        } else if (options.vertColorMode==VERTCOLOR_BY_PARENT_TYPE) {

            int ofs=3;
            if (maxlevel>0) {
                OpenSubdiv::Far::TopologyLevel const & refPrevLevel = refiner.GetLevel(maxlevel-1);

                for (int vert=0; vert<refPrevLevel.GetNumFaces(); ++vert, ofs+=6) {
                    memcpy(&vbo[ofs], g_parentTypeColors[1], sizeof(float)*3);
                }
                for (int vert=0; vert<refPrevLevel.GetNumEdges(); ++vert, ofs+=6) {
                    memcpy(&vbo[ofs], g_parentTypeColors[2], sizeof(float)*3);
                }
                for (int vert=0; vert<refPrevLevel.GetNumVertices(); ++vert, ofs+=6) {
                    memcpy(&vbo[ofs], g_parentTypeColors[3], sizeof(float)*3);
                }
            } else {
                for (int vert=0; vert<refLastLevel.GetNumVertices(); ++vert, ofs+=6) {
                    memcpy(&vbo[ofs], g_parentTypeColors[0], sizeof(float)*3);
                }
            }
        } else {

            for (int vert=0, ofs=3; vert<nverts; ++vert) {
                setSolidColor(&vbo[ofs+=6]);
            }
        }
    }


    { // edge color component ------------------------------

        std::vector<float> & vbo = _vbo[COMP_EDGE];
        vbo.resize(nedges * 2 * 6);

        std::vector<int> & eao = _eao[COMP_EDGE];
        eao.resize(nedges*2);

        for (int edge=0; edge<nedges; ++edge) {

            eao[edge*2  ] = edge*2;
            eao[edge*2+1] = edge*2+1;

            IndexArray const verts = refLastLevel.GetEdgeVertices(edge);

            float * v0 = &vbo[edge*2*6],
                  * v1 = v0+6;

            // copy position
            memcpy(v0, vertData + verts[0]*3, sizeof(float)*3);
            memcpy(v1, vertData + verts[1]*3, sizeof(float)*3);

            // set colors
            if (options.edgeColorMode==EDGECOLOR_BY_LEVEL) {

                setColorByLevel(maxlevel, v0+3);
                setColorByLevel(maxlevel, v1+3);
             } else  if (options.edgeColorMode==EDGECOLOR_BY_SHARPNESS) {

                float sharpness = refLastLevel.GetEdgeSharpness(edge);
                setColorBySharpness(sharpness, v0+3);
                setColorBySharpness(sharpness, v1+3);
            } else {

                // default to solid color
                setSolidColor(v0+3);
                setSolidColor(v1+3);
            }
        }
    }

    { // face component ------------------------------------

        std::vector<float> & vbo = _vbo[COMP_FACE];
        vbo.resize(nverts * 3);

        memcpy(&vbo[0], vertData, nverts*sizeof(float)*3);

        int nfaceverts = refLastLevel.GetNumFaceVertices();

        std::vector<int> & eao = _eao[COMP_FACE];
        eao.resize(nfaceverts);

        _faceColors.resize(nfaces*4);

        for (int face=0, ofs=0; face<nfaces; ++face) {

            IndexArray fverts = refLastLevel.GetFaceVertices(face);
            for (int vert=0; vert<fverts.size(); ++vert) {
                eao[ofs++] = fverts[vert];
            }

            setSolidColor(&_faceColors[face*4]);
        }
    }
}

//------------------------------------------------------------------------------
inline void
setEdge(std::vector<float> & vbo, int edge, float const * vertData, int v0, int v1, float const * color) {

    float * dst0 = &vbo[edge*2*6],
          * dst1 = dst0+6;

    memcpy(dst0, vertData + (v0*3), sizeof(float)*3);
    memcpy(dst1, vertData + (v1*3), sizeof(float)*3);

    memcpy(dst0+3, color, sizeof(float)*3);
    memcpy(dst1+3, color, sizeof(float)*3);
}

//------------------------------------------------------------------------------
void
GLMesh::InitializeFVar(Options options, TopologyRefiner const & refiner,
    PatchTable const * patchTable, int channel, int tessFactor, float const * fvarData) {

    int nverts = refiner.GetNumFVarValuesTotal(channel);

    { // vertex color component ----------------------------

        initializeVertexComponentBuffer(fvarData, nverts);

        std::vector<float> & vbo = _vbo[COMP_VERT];

        if (options.vertColorMode==VERTCOLOR_BY_LEVEL) {

            for (int level=0, ofs=3; level<=refiner.GetMaxLevel(); ++level) {
                for (int vert=0; vert<refiner.GetLevel(level).GetNumFVarValues(channel); ++vert, ofs+=6) {
                    assert(ofs<(int)vbo.size());
                    setColorByLevel(level, &vbo[ofs]);
                }
            }
        } else {

            for (int vert=0, ofs=3; vert<nverts; ++vert) {
                setSolidColor(&vbo[ofs+=6]);
            }
        }
    }

    if (tessFactor>0) {
        // edge color component ------------------------------

        int npatches = patchTable->GetNumPatchesTotal(),
            nvertsperpatch = (tessFactor) * (tessFactor),
            nedgesperpatch = (tessFactor-1) * (tessFactor*2+tessFactor-1),
            //nverts = npatches * nvertsperpatch,
            nedges = npatches * nedgesperpatch;

        std::vector<float> & vbo = _vbo[COMP_EDGE];
        vbo.resize(nedges * 2 * 6);

        std::vector<int> & eao = _eao[COMP_EDGE];
        eao.reserve(nedges*2);


        // default to solid color

        static float quadColor[3] = { 1.0f, 1.0f, 0.0f };
        float const * color = quadColor;

        // wireframe indices
        int * basisedges = (int *)alloca(2*nedgesperpatch*sizeof(int)),
            * ptr = basisedges;
        for (int i=0; i<(tessFactor-1); ++i) {       //  tess pattern :
            for (int j=0; j<(tessFactor-1); ++j) {   //
                *ptr++ = i*tessFactor + j;           //   o---o---o--
                *ptr++ = i*tessFactor + j+1;         //   |\  |\  |
                                                     //   | \ | \ |
                *ptr++ = i * tessFactor + j;         //   |  \|  \|
                *ptr++ = (i+1) * tessFactor + j;     //   o---o---o--
                                                     //   |\  |\  |
                *ptr++ = i * tessFactor + j;         //   | \ | \ |
                *ptr++ = (i+1) * tessFactor + j+1;   //   |  \|  \|
            }                                        //   o---o---o--
            *ptr++ = (i+1) * tessFactor - 1;         //   |   |   |
            *ptr++ = (i+2) * tessFactor - 1;

            *ptr++ = tessFactor * (tessFactor-1) + i;
            *ptr++ = tessFactor * (tessFactor-1) + i+1;
        }

        for (int patch=0, offset=0; patch<npatches; ++patch) {

            assert(color);

            for (int edge=0; edge<nedgesperpatch; ++edge) {

                eao.push_back((int)eao.size());
                eao.push_back((int)eao.size());

                int v0 = offset + basisedges[edge*2],
                    v1 = offset + basisedges[edge*2+1];

                setEdge(vbo, patch*nedgesperpatch+edge, fvarData, v0, v1, color);
            }

            offset += nvertsperpatch;
        }
    }

    _numComps[COMP_FACE] = (int)_eao[COMP_FACE].size();
    _numComps[COMP_EDGE] = (int)_eao[COMP_EDGE].size();
    _numComps[COMP_VERT] = (int)_eao[COMP_VERT].size();

    InitializeDeviceBuffers();
}

//------------------------------------------------------------------------------
// returns the number of edges in a patch with 'numCVs'
inline int
getNumEdges(int numCVs) {

    switch (numCVs) {
        case  4: return  4;
//        case  9: return 12;
        case  9: return 4;
//        case 12: return 17;
        case 12: return 4;
//        case 16: return 24;
        case 16: return  4;
        case 20: return  4;
        default:
            assert(0);
    }
    return -1;
}

//------------------------------------------------------------------------------
int const *
getEdgeList(int numCVs) {

/*
    static int edgeList4[] = { 0, 1, 1, 2, 2, 3, 3, 0 };

    static int edgeList9[] = { 0, 1, 1, 4,
                               3, 2, 2, 5,
                               8, 7, 7, 6,
                               0, 3, 3, 8,
                               1, 2, 2, 7,
                               4, 5, 5, 6 };

    static int edgeList12[] = {  4,  0,  0,  3,  3,  5,
                                11,  1,  1,  2,  2,  6,
                                10,  9,  9,  8,  8,  7,
                                 4, 11, 11, 10,  0,  1,
                                 1,  9,  3,  2,  2,  8,
                                 5,  6,  6,  7 };

    static int edgeList16[] = {  4, 15, 15, 14, 14, 13,
                                 5,  0,  0,  3,  3, 12,
                                 6,  1,  1,  2,  2, 11,
                                 7,  8,  8,  9,  9, 10,
                                 4,  5,  5,  6,  6,  7,
                                15,  0,  0,  1,  1,  8,
                                14,  3,  3,  2,  2,  9,
                                13, 12, 12, 11, 11, 10  };
*/
    static int edgeList4of16[] = { 5, 6, 6, 10, 10, 9, 9, 5 };
    static int edgeList4of20[] = { 0, 5, 5, 10, 10, 15, 15, 0 };

    switch (numCVs) {
        case  4: return edgeList4of20; break;
        case 16: return edgeList4of16; break;
        case 20: return edgeList4of20; break;
        default:
            assert(0);
    }
    return 0;
}

inline int
getRingSize(OpenSubdiv::Far::PatchDescriptor desc) {
    if (desc.GetType()==OpenSubdiv::Far::PatchDescriptor::GREGORY_BASIS) {
        return 4;
    } else {
        return desc.GetNumControlVertices();
    }
}

//------------------------------------------------------------------------------
void
GLMesh::initializeBuffers(Options options, TopologyRefiner const & refiner,
    PatchTable const & patchTable, float const * vertexData) {

    int nverts = refiner.GetNumVerticesTotal();

    { // vertex color component ----------------------------

        initializeVertexComponentBuffer(vertexData, nverts);

        std::vector<float> & vbo = _vbo[COMP_VERT];

        if (options.vertColorMode==VERTCOLOR_BY_LEVEL) {

            for (int level=0, ofs=3; level<=refiner.GetMaxLevel(); ++level) {
                for (int vert=0; vert<refiner.GetLevel(level).GetNumVertices(); ++vert, ofs+=6) {
                    assert(ofs<(int)vbo.size());
                    setColorByLevel(level, &vbo[ofs]);
                }
            }
        } else {

            for (int vert=0, ofs=3; vert<nverts; ++vert) {
                setSolidColor(&vbo[ofs+=6]);
            }
        }
    }

    typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

    { // edge color component ------------------------------

        int nedges = 0;

        for (int array=0; array<(int)patchTable.GetNumPatchArrays(); ++array) {

            int ncvs = getRingSize(patchTable.GetPatchArrayDescriptor(array));

            nedges += patchTable.GetNumPatches(array) * getNumEdges(ncvs);
        }
        std::vector<float> & vbo = _vbo[COMP_EDGE];
        vbo.resize(nedges * 2 * 6);

        std::vector<int> & eao = _eao[COMP_EDGE];
        eao.resize(nedges*2);

        // default to solid color
        float solidColor[3];
        setSolidColor(solidColor);

        float const * color=solidColor;

        for (int array=0, edge=0; array<(int)patchTable.GetNumPatchArrays(); ++array) {

            OpenSubdiv::Far::PatchDescriptor desc =
                patchTable.GetPatchArrayDescriptor(array);

            if (options.edgeColorMode==EDGECOLOR_BY_PATCHTYPE) {
                color = getAdaptivePatchColor(desc);
            }

            int ncvs = getRingSize(desc);

            for (int patch=0; patch<patchTable.GetNumPatches(array); ++patch) {

                OpenSubdiv::Far::ConstIndexArray const cvs =
                    patchTable.GetPatchVertices(array, patch);

                int const * edgeList=getEdgeList(ncvs);

                for (int k=0; k<getNumEdges(cvs.size()); ++k, ++edge) {

                    eao[edge*2  ] = edge*2;
                    eao[edge*2+1] = edge*2+1;

                    int v0 = cvs[edgeList[k*2]],
                        v1 = cvs[edgeList[k*2+1]];
                    setEdge(vbo, edge, vertexData, v0, v1, color);
                }
            }
        }
    }

    { // face color component ------------------------------

        int nfaces = patchTable.GetNumPatchesTotal();

        std::vector<float> & vbo = _vbo[COMP_FACE];
        vbo.resize(nverts*3);
        memcpy(&vbo[0], vertexData, nverts*sizeof(float)*3);

        std::vector<int> & eao = _eao[COMP_FACE];
        eao.resize(nfaces*4);

        _faceColors.resize(nfaces*4, 1.0f);

        // default to solid color
        for (int array=0, face=0; array<(int)patchTable.GetNumPatchArrays(); ++array) {

            OpenSubdiv::Far::PatchDescriptor desc =
               patchTable.GetPatchArrayDescriptor(array);

            //int ncvs = getRingSize(desc);

            for (int patch=0; patch<patchTable.GetNumPatches(array); ++patch, ++face) {

                OpenSubdiv::Far::ConstIndexArray const cvs =
                    patchTable.GetPatchVertices(array, patch);

                if (desc.GetType()==Descriptor::REGULAR) {
                    eao[face*4  ] = cvs[ 5];
                    eao[face*4+1] = cvs[ 6];
                    eao[face*4+2] = cvs[10];
                    eao[face*4+3] = cvs[ 9];
                } else {
                    memcpy(&eao[face*4], cvs.begin(), 4*sizeof(OpenSubdiv::Far::Index));
                }

                if (options.faceColorMode==FACECOLOR_BY_PATCHTYPE) {
                    float const * color = getAdaptivePatchColor(desc);
                    memcpy(&_faceColors[face*4], color, 4*sizeof(float));
                } else {
                    setSolidColor(&_faceColors[face*4]);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename T> static GLuint
createTextureBuffer(T const &data, GLint format, int offset=0) {

    GLuint buffer = 0, texture = 0;

    glGenTextures(1, &texture);
    glGenBuffers(1, &buffer);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER,
        (data.size()-offset)*sizeof(typename T::value_type),
            &data[offset], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glTexBuffer(GL_TEXTURE_BUFFER, format, buffer);

    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glDeleteBuffers(1, &buffer);

    GLUtils::CheckGLErrors("createTextureBuffer");
    return texture;
}


//------------------------------------------------------------------------------
void
GLMesh::InitializeDeviceBuffers() {

    // copy buffers to device
    for (int i=0; i<COMP_NUM_COMPONENTS; ++i) {

        if (! _VAO[i]) {
            glGenVertexArrays(1, &_VAO[i]);
        }
        glBindVertexArray(_VAO[i]);

        if (! _vbo[i].empty()) {
            if (! _VBO[i]) {
                glGenBuffers(1, &_VBO[i]);
            }
            glBindBuffer(GL_ARRAY_BUFFER, _VBO[i]);
            glBufferData(GL_ARRAY_BUFFER, _vbo[i].size()*sizeof(GLfloat), &_vbo[i][0], GL_STATIC_DRAW);

            glEnableVertexAttribArray(0);

            int numelements = (i==COMP_FACE) ? 3 : 6;
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, numelements*sizeof(GLfloat), 0);

            if (i==COMP_FACE) {
                 // face vbo has no color component
                 glDisableVertexAttribArray(1);
             } else {
                 glEnableVertexAttribArray(1);
                 glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), (void*)12);
             }
        }

        if (! _eao[i].empty()) {

            if (! _EAO[i]) {
                glGenBuffers(1, &_EAO[i]);
            }
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EAO[i]);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, _eao[i].size()*sizeof(int), &_eao[i][0], GL_STATIC_DRAW);
        }

        GLUtils::CheckGLErrors("init");
    }

    if (! _faceColors.empty()) {
        _TBOfaceColors = createTextureBuffer(_faceColors, GL_RGBA32F);
    }

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    clearBuffers();
}

//------------------------------------------------------------------------------
void
GLMesh::clearBuffers() {

    for (int i=0; i<COMP_NUM_COMPONENTS; ++i) {
        _vbo[i].clear();
        _eao[i].clear();
    }
}


//------------------------------------------------------------------------------
static const char * g_simpleShaderSrc =
#include "simpleShader.gen.h"
;

static const char * g_faceShaderSrc =
#include "faceShader.gen.h"
;

GLuint g_simpleProgram=0,
       g_faceProgram=0;

// Update and bind transform state
static void
bindProgram( char const * shaderSource,
             GLuint * program,
             GLuint transformUB,
             GLuint lightingUB,
             bool geometry) {

    assert(program);

    GLuint uboIndex=GL_INVALID_INDEX,
           transformBinding=0,
           lightingBinding=1;

    // Update and bind transform state
    if (! *program) {

        *program = glCreateProgram();

        static char const versionStr[] = "#version 330\n",
                          vtxDefineStr[] = "#define VERTEX_SHADER\n",
                          geoDefineStr[] = "#define GEOMETRY_SHADER\n",
                          fragDefineStr[] = "#define FRAGMENT_SHADER\n";

        std::string vsSrc = std::string(versionStr) + vtxDefineStr + shaderSource,
                    gsSrc = std::string(versionStr) + geoDefineStr + shaderSource,
                    fsSrc = std::string(versionStr) + fragDefineStr + shaderSource;

        GLuint vertexShader =
            GLUtils::CompileShader(GL_VERTEX_SHADER, vsSrc.c_str()),
               geometryShader = geometry ?
            GLUtils::CompileShader(GL_GEOMETRY_SHADER, gsSrc.c_str()) : 0,
               fragmentShader =
            GLUtils::CompileShader(GL_FRAGMENT_SHADER, fsSrc.c_str());

        glAttachShader(*program, vertexShader);
        if (geometry) {
            glAttachShader(*program, geometryShader);
        }
        glAttachShader(*program, fragmentShader);

        glLinkProgram(*program);

        GLint status;
        glGetProgramiv(*program, GL_LINK_STATUS, &status);
        if (status == GL_FALSE) {
            GLint infoLogLength;
            glGetProgramiv(*program, GL_INFO_LOG_LENGTH, &infoLogLength);
            char *infoLog = new char[infoLogLength];
            glGetProgramInfoLog(*program, infoLogLength, NULL, infoLog);
            printf("%s\n", infoLog);
            delete[] infoLog;
            exit(1);
        }

        uboIndex = glGetUniformBlockIndex(*program, "Transform");
        if (uboIndex != GL_INVALID_INDEX) {
            glUniformBlockBinding(*program, uboIndex, transformBinding);
        }

        uboIndex = glGetUniformBlockIndex(*program, "Lighting");
        if (uboIndex != GL_INVALID_INDEX) {
            glUniformBlockBinding(*program, uboIndex, lightingBinding);
        }
    }
    glUseProgram(*program);

    if (transformUB) {
        glBindBufferBase(GL_UNIFORM_BUFFER, transformBinding, transformUB);
    }

    if (lightingUB) {
        glBindBufferBase(GL_UNIFORM_BUFFER, lightingBinding, lightingUB);
    }
}

//------------------------------------------------------------------------------
void
GLMesh::Draw(Component comp, GLuint transformUB, GLuint lightingUB) {

    if (comp==COMP_VERT) {

        bindProgram(g_simpleShaderSrc, &g_simpleProgram, transformUB, lightingUB, false);

        glBindVertexArray(_VAO[COMP_VERT]);

        glPointSize(4.0f);
        glDrawElements(GL_POINTS, _numComps[COMP_VERT], GL_UNSIGNED_INT, (void *)0);
        glPointSize(1.0f);

    } else if (comp==COMP_EDGE) {

        bindProgram(g_simpleShaderSrc, &g_simpleProgram, transformUB, lightingUB, false);

        glBindVertexArray(_VAO[COMP_EDGE]);

        glDrawElements(GL_LINES, _numComps[COMP_EDGE], GL_UNSIGNED_INT, (void *)0);

    } else if (comp==COMP_FACE) {

        glEnable(GL_CULL_FACE);

        bindProgram(g_faceShaderSrc, &g_faceProgram, transformUB, lightingUB, true);


        { // set shader parameters
            GLuint diffuseColor = glGetUniformLocation(g_faceProgram, "diffuseColor");
            glProgramUniform4f(g_faceProgram, diffuseColor, _diffuseColor[0],
                _diffuseColor[1], _diffuseColor[2], _diffuseColor[3]);

            GLuint faceColors = glGetUniformLocation(g_faceProgram, "faceColors");
            glUniform1i(faceColors, 0); // GL_TEXTURE0

            GLuint faceTexture = glGetUniformLocation(g_faceProgram, "faceTexture");
            glUniform1i(faceTexture, 1); // GL_TEXTURE1

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_BUFFER, _TBOfaceColors);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, getFaceTexture());
        }

        glBindVertexArray(_VAO[COMP_FACE]);

        glDrawElements(GL_LINES_ADJACENCY, _numComps[COMP_FACE], GL_UNSIGNED_INT, (void *)0);

        glDisable(GL_CULL_FACE);
    }
}

//------------------------------------------------------------------------------
void
GLMesh::SetDiffuseColor(float r, float g, float b, float a) {

    _diffuseColor[0] = r;
    _diffuseColor[1] = g;
    _diffuseColor[2] = b;
    _diffuseColor[3] = a;
}

//------------------------------------------------------------------------------
void
GLMesh::SetFaceColor(int face, float r, float g, float b, float a) {

    assert( (face*4) < (int)_faceColors.size() );

    float * color = &_faceColors[face*4];
    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = a;
}

//------------------------------------------------------------------------------

