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

#include "gl_fontutils.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

//------------------------------------------------------------------------------
GLFont::GLFont(GLuint fontTexture) :
    _dirty(false),
    _program(0),
    _transformBinding(0),
    _attrPosition(0),
    _attrData(0),
    _fontTexture(fontTexture),
    _scale(0) {

    _chars.reserve(500000);

    glGenVertexArrays(1, &_VAO);
    glGenBuffers(1, &_EAO);
    glGenBuffers(1, &_VBO);

    glBindVertexArray(_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EAO);

    glBindVertexArray(0);
}

//------------------------------------------------------------------------------
GLFont::~GLFont() {

    glDeleteVertexArrays(1, &_VAO);
    glGenBuffers(1, &_VBO);
}

//------------------------------------------------------------------------------
void GLFont::bindProgram() {

    static const char *shaderSource =
#include "fontShader.gen.h"
;
    // Update and bind transform state
    if (! _program) {

        _program = glCreateProgram();

        static char const versionStr[] = "#version 410\n",
                          vtxDefineStr[] = "#define VERTEX_SHADER\n",
                          geoDefineStr[] = "#define GEOMETRY_SHADER\n",
                          fragDefineStr[] = "#define FRAGMENT_SHADER\n";

        std::string vsSrc = std::string(versionStr) + vtxDefineStr + shaderSource,
                    gsSrc = std::string(versionStr) + geoDefineStr + shaderSource,
                    fsSrc = std::string(versionStr) + fragDefineStr + shaderSource;

        GLuint vertexShader =
            GLUtils::CompileShader(GL_VERTEX_SHADER, vsSrc.c_str()),
               geometryShader =
            GLUtils::CompileShader(GL_GEOMETRY_SHADER, gsSrc.c_str()),
               fragmentShader =
            GLUtils::CompileShader(GL_FRAGMENT_SHADER, fsSrc.c_str());

        glAttachShader(_program, vertexShader);
        glAttachShader(_program, geometryShader);
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
        }

    }
    glUseProgram(_program);

    if (! _scale) {
        _scale = glGetUniformLocation(_program, "scale");
    }

    if (! _attrPosition) {
        _attrPosition = glGetAttribLocation(_program, "position");
    }

    if (! _attrData) {
        _attrData = glGetAttribLocation(_program, "data");
    }
}

//------------------------------------------------------------------------------
void
GLFont::SetFontScale(float scale) {

    if (_scale) {
        glProgramUniform1f(_program, _scale, scale);
    }
}

//------------------------------------------------------------------------------
void
GLFont::Clear() {
    _chars.clear();
}

//------------------------------------------------------------------------------
void
GLFont::Print3D(float const pos[3], const char * str, int color) {

    int len = (int)strlen(str);

    for (int i=0; i<len; ++i) {

        GLFont::Char c;
        memcpy(&c.pos[0], pos, sizeof(float)*3);

        c.ofs[0]=2.0f*i;
        c.ofs[1]=0.0f;

        c.alpha = (float)str[i];
        c.color = (float)color;

        _chars.push_back(c);
    }
    _dirty=true;
}

//------------------------------------------------------------------------------
void GLFont::Draw(GLuint transformUB) {

    if ((int)_chars.size()==0) {
        return;
    }

    assert(_VAO && _VBO);

    glBindVertexArray(_VAO);

    bindProgram();

    if (! _transformBinding) {

        GLuint uboIndex = glGetUniformBlockIndex(_program, "Transform");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(_program, uboIndex, _transformBinding);

    }
    assert(transformUB);
    glBindBufferBase(GL_UNIFORM_BUFFER, _transformBinding, transformUB);

    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EAO);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _fontTexture);

    if (_dirty) {
        // generate element array indices for GL_POINTS
        std::vector<int> eao(_chars.size());
        for (int i=0; i<(int)_chars.size(); ++i) {
            eao[i]=i;
        }
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, eao.size()*sizeof(int), &eao[0], GL_STATIC_DRAW);

        // copy character data to VBO
        glBufferData(GL_ARRAY_BUFFER, _chars.size()*sizeof(Char), &_chars[0], GL_STATIC_DRAW);
        _dirty=false;
    }

    glEnableVertexAttribArray(_attrPosition);
    glVertexAttribPointer(_attrPosition, 3, GL_FLOAT, GL_FALSE, sizeof(Char), 0);

    glEnableVertexAttribArray(_attrData);
    glVertexAttribPointer(_attrData,     4, GL_FLOAT, GL_FALSE, sizeof(Char), (void*)12);

    glDrawElements(GL_POINTS, (int)_chars.size(), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    glUseProgram(0);
}

//------------------------------------------------------------------------------
