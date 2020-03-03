//
//   Copyright 2015 Pixar
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

#include "glControlMeshDisplay.h"
#include "glUtils.h"

#include <vector>

GLControlMeshDisplay::GLControlMeshDisplay() :
    _displayEdges(true), _displayVertices(false),
    _program(0), _vao(0),
    _vertSharpness(0), _edgeSharpnessTexture(0), _edgeIndices(0),
    _numEdges(0), _numPoints(0) {
}

GLControlMeshDisplay::~GLControlMeshDisplay() {
    if (_program)       glDeleteProgram(_program);
    if (_vertSharpness) glDeleteBuffers(1, &_vertSharpness);
    if (_edgeSharpnessTexture) glDeleteTextures(1, &_edgeSharpnessTexture);
    if (_edgeIndices)   glDeleteBuffers(1, &_edgeIndices);
    if (_vao)           glDeleteVertexArrays(1, &_vao);
}

bool
GLControlMeshDisplay::createProgram() {
    if (_program != 0) glDeleteProgram(_program);

    const std::string glsl_version = GLUtils::GetShaderVersionInclude();

    static const std::string vsSrc =
        glsl_version +
        "in vec3 position;                                         \n"
        "in float vertSharpness;                                   \n"
        "out float sharpness;                                      \n"
        "uniform mat4 mvpMatrix;                                   \n"
        "void main() {                                             \n"
        "  sharpness = vertSharpness;                              \n"
        "  gl_Position = mvpMatrix * vec4(position, 1);            \n"
        "}                                                         \n";

    static const std::string fsSrc =
        glsl_version +
        "in float sharpness;                                       \n"
        "out vec4 color;                                           \n"
        "uniform int drawMode = 0;                                 \n"
        "uniform samplerBuffer edgeSharpness;                      \n"
        "vec4 sharpnessToColor(float s) {                          \n"
        "  //  0.0       2.0       4.0                             \n"
        "  // green --- yellow --- red                             \n"
        "  return vec4(min(1, s * 0.5),                            \n"
        "              min(1, 2 - s * 0.5),                        \n"
        "              0, 1);                                      \n"
        "}                                                         \n"
        "void main() {                                             \n"
        "  float sharp = sharpness;                                \n"
        "  if (drawMode == 1) {                                    \n"
        "    sharp = texelFetch(edgeSharpness, gl_PrimitiveID).x;  \n"
        "  }                                                       \n"
        "  color = sharpnessToColor(sharp);                        \n"
        "}                                                         \n";

    GLuint vertexShader =
        GLUtils::CompileShader(GL_VERTEX_SHADER, vsSrc.c_str());
    GLuint fragmentShader =
        GLUtils::CompileShader(GL_FRAGMENT_SHADER, fsSrc.c_str());

    _program = glCreateProgram();
    glAttachShader(_program, vertexShader);
    glAttachShader(_program, fragmentShader);
    glLinkProgram(_program);

    GLint status;
    glGetProgramiv(_program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &infoLogLength);
        char *infoLog = new char[infoLogLength];
        glGetProgramInfoLog(_program, infoLogLength, NULL, infoLog);
        printf("%s\n", infoLog);
        delete[] infoLog;
        exit(1);
        return false;
    }

    _uniformMvpMatrix =
        glGetUniformLocation(_program, "mvpMatrix");
    _uniformDrawMode =
        glGetUniformLocation(_program, "drawMode");
    _uniformEdgeSharpness =
        glGetUniformLocation(_program, "edgeSharpness");

    _attrPosition = glGetAttribLocation(_program, "position");
    _attrVertSharpness = glGetAttribLocation(_program, "vertSharpness");

    return true;
}

void
GLControlMeshDisplay::Draw(GLuint vbo, GLint stride,
                           const float *modelViewProjectionMatrix) {
    if (_program == 0) {
        createProgram();
        if (_program == 0) return;
    }

    if (_vao == 0) {
        glGenVertexArrays(1, &_vao);
    }
    glBindVertexArray(_vao);

    glUseProgram(_program);

    glUniformMatrix4fv(_uniformMvpMatrix,
                       1, GL_FALSE, modelViewProjectionMatrix);
    glUniform1i(_uniformEdgeSharpness, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, _edgeSharpnessTexture);

    // bind vbo to points
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(_attrPosition, 3, GL_FLOAT, GL_FALSE, stride, 0);
    glBindBuffer(GL_ARRAY_BUFFER, _vertSharpness);
    glVertexAttribPointer(_attrVertSharpness, 1, GL_FLOAT, GL_FALSE, 0, 0);

    glPointSize(10.0);

    // draw edges
    if (_displayEdges) {
        glUniform1i(_uniformDrawMode, 1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _edgeIndices);
        glDrawElements(GL_LINES, _numEdges*2, GL_UNSIGNED_INT, 0);
    }

    // draw vertices
    if (_displayVertices) {
        glUniform1i(_uniformDrawMode, 0);
        glDrawArrays(GL_POINTS, 0, _numPoints);
    }

    glPointSize(1.0f);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glUseProgram(0);
}

void
GLControlMeshDisplay::SetTopology(OpenSubdiv::Far::TopologyLevel const &level) {
    int nEdges = level.GetNumEdges();
    int nVerts = level.GetNumVertices();

    std::vector<int> edgeIndices;
    std::vector<float> edgeSharpnesses;
    std::vector<float> vertSharpnesses;

    edgeIndices.reserve(nEdges * 2);
    edgeSharpnesses.reserve(nEdges);
    vertSharpnesses.reserve(nVerts);

    for (int i = 0; i < nEdges; ++i) {
        OpenSubdiv::Far::ConstIndexArray verts = level.GetEdgeVertices(i);
        edgeIndices.push_back(verts[0]);
        edgeIndices.push_back(verts[1]);
        edgeSharpnesses.push_back(level.GetEdgeSharpness(i));
    }

    for (int i = 0; i < nVerts; ++i) {
        vertSharpnesses.push_back(level.GetVertexSharpness(i));
    }

    if (_vertSharpness == 0) glGenBuffers(1, &_vertSharpness);
    if (_edgeIndices == 0)   glGenBuffers(1, &_edgeIndices);
    if (_edgeSharpnessTexture == 0) glGenTextures(1, &_edgeSharpnessTexture);
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);

    glBindBuffer(GL_ARRAY_BUFFER, _vertSharpness);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*nVerts,
                 &vertSharpnesses[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, _edgeIndices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(int)*nEdges*2,
                 &edgeIndices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*nEdges,
                 &edgeSharpnesses[0], GL_STATIC_DRAW);

    glBindTexture(GL_TEXTURE_BUFFER, _edgeSharpnessTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, buffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDeleteBuffers(1, &buffer);

    _numEdges = nEdges;
    _numPoints = nVerts;
}

