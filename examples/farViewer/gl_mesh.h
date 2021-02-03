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

#ifndef GL_MESH_H
#define GL_MESH_H

#include "../../regression/common/far_utils.h"
#include <opensubdiv/far/patchTable.h>

#include "../common/glUtils.h"

#include <algorithm>

// Wrapper class for drawing Far meshes & components
class GLMesh {

public:

    enum Component {
        COMP_FACE=0,
        COMP_EDGE,
        COMP_VERT,
        COMP_NUM_COMPONENTS
    };

    enum VertColorMode {
        VERTCOLOR_SOLID=0,
        VERTCOLOR_BY_LEVEL,
        VERTCOLOR_BY_SHARPNESS,
        VERTCOLOR_BY_PARENT_TYPE
    };

    enum EdgeColorMode {
        EDGECOLOR_SOLID=0,
        EDGECOLOR_BY_LEVEL,
        EDGECOLOR_BY_SHARPNESS,
        EDGECOLOR_BY_PATCHTYPE
    };

    enum FaceColorMode {
        FACECOLOR_SOLID=0,
        FACECOLOR_BY_PATCHTYPE
    };

    struct Options {
        Options() : vertColorMode(0), edgeColorMode(0), faceColorMode(0) {}
        unsigned int vertColorMode:3,
                     edgeColorMode:3,
                     faceColorMode:3;
    };

    // -----------------------------------------------------
    // Raw topology initialization
    void Initialize(Options options,
        int nverts, int nfaces, int * vertsperface, int * faceverts,
            float const * vertexData);

    // -----------------------------------------------------
    // Far initialization
    typedef OpenSubdiv::Far::TopologyRefiner TopologyRefiner;

    typedef OpenSubdiv::Far::PatchTable PatchTable;

    void Initialize(Options options, TopologyRefiner const & refiner,
        PatchTable const * patchTable, float const * vertexData);

    void InitializeFVar(Options options, TopologyRefiner const & refiner,
        PatchTable const * patchTable, int channel, int tessFactor, float const * fvarData);

    void InitializeDeviceBuffers();

    // -----------------------------------------------------

    GLMesh();

    ~GLMesh();

    void Draw(Component comp, GLuint transformUB, GLuint lightingUB);

    void SetDiffuseColor(float r, float g, float b, float a);

    void SetFaceColor(int face, float r, float g, float b, float a);

private:

           void setSolidColor(float * color);

    static void setColorByLevel(int level, float * color);

    static void setColorBySharpness(float sharpness, float * color);

    void initializeVertexComponentBuffer(float const * vertexData, int nverts);

    void initializeBuffers(Options options, TopologyRefiner const & refiner,
        float const * vertexData);

    void initializeBuffers(Options options, TopologyRefiner const & refiner,
        PatchTable const & patchTable, float const * vertexData);

    void clearBuffers();


    int _numComps[COMP_NUM_COMPONENTS];

    GLuint _VAO[COMP_NUM_COMPONENTS],
           _VBO[COMP_NUM_COMPONENTS],
           _EAO[COMP_NUM_COMPONENTS],
           _TBOfaceColors;

    std::vector<float> _vbo[COMP_NUM_COMPONENTS];
    std::vector<int>   _eao[COMP_NUM_COMPONENTS];

    std::vector<float > _faceColors;

    float _ambientColor[4],
          _diffuseColor[4];
};

#endif // GL_MESH_H
