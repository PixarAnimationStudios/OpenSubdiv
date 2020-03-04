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

#include "glHud.h"
#include "glUtils.h"

#include "font_image.h"
#include "simple_math.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <iostream>

static const char *s_VS =
#if defined(GL_VERSION_3_1)
    "#version 150\n"
    "in vec2 position;\n"
    "in vec3 color;\n"
    "in vec2 uv;\n"
    "out vec4 fragColor;\n"
    "out vec2 fragUV;\n"
    "uniform mat4 ModelViewProjectionMatrix;\n"
    "void main() {\n"
    "  fragColor = vec4(color, 1);\n"
    "  fragUV = uv;\n"
    "  gl_Position = ModelViewProjectionMatrix * "
    "                  vec4(position.x, position.y, 0, 1);\n"
    "}\n";
#else
    "attribute vec2 position;\n"
    "attribute vec3 color;\n"
    "attribute vec2 uv;\n"
    "varying vec4 fragColor;\n"
    "varying vec2 fragUV;\n"
    "uniform mat4 ModelViewProjectionMatrix;\n"
    "void main() {\n"
    "  fragColor = vec4(color, 1);\n"
    "  fragUV = uv;\n"
    "  gl_Position = ModelViewProjectionMatrix * "
    "                  vec4(position.x, position.y, 0, 1);\n"
    "}\n";
#endif

static const char *s_FS =
#if defined(GL_VERSION_3_1)
    "#version 150\n"
    "in vec4 fragColor;\n"
    "in vec2 fragUV;\n"
    "out vec4 color;\n"
    "uniform sampler2D fontTexture;\n"
    "void main() {\n"
    "  vec4 c = texture(fontTexture, fragUV);\n"
    "  if (c.a == 0.0) discard;\n"
    "  color = c*fragColor;\n"
    "}\n";
#else
    "varying vec4 fragColor;\n"
    "varying vec2 fragUV;\n"
    "uniform sampler2D fontTexture;\n"
    "void main()\n"
    "{\n"
    "  vec4 c = texture2D(fontTexture, fragUV);\n"
    "  if (c.a == 0.0) discard;\n"
    "  gl_FragColor = c*fragColor;\n"
    "}\n";
#endif

static const char *s_BG_VS =
#if defined(GL_VERSION_3_1)
    "#version 150\n"
    "out vec2 uv;\n"
    "void main() {\n"
    "  vec4 pos[4] = vec4[] (vec4(-1,-1,0,1), vec4(1,-1,0,1), \n"
    "                        vec4(-1,1,0,1), vec4(1,1,0,1));  \n"
    "  uv = pos[gl_VertexID].xy;\n"
    "  gl_Position = pos[gl_VertexID];\n"
    "}\n";
#else
    "varying vec2 uv;\n"
    "void main() {\n"
    "  vec4 pos[4] = vec4[] (vec4(-1,-1,0,1), vec4(1,-1,0,1), \n"
    "                        vec4(-1,1,0,1), vec4(1,1,0,1));  \n"
    "  uv = pos[gl_VertexID].xy;\n"
    "  gl_Position = pos[gl_VertexID];\n"
    "}\n";
#endif

static const char *s_BG_FS =
#if defined(GL_VERSION_3_1)
    "#version 150\n"
    "in vec2 uv;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "  color = vec4(mix(0.1, 0.5, sin((uv.y*0.5+0.5)*3.14159)));\n"
    "  color.a = 1.0;\n"
    "}\n";
#else
    "varying vec2 uv;\n"
    "void main() {\n"
    "  gl_FragColor = vec4(mix(0.1, 0.5, sin((uv.y*0.5+0.5)*3.14159)));\n"
    "  gl_FragColor.a = 1.0;\n"
    "}\n";
#endif

GLhud::GLhud() : _fontTexture(0), _vbo(0), _staticVbo(0),
                 _vao(0), _staticVao(0), _program(0),
                 _aPosition(0), _aColor(0), _aUV(0), _bgProgram(0) {
}

GLhud::~GLhud() {
    if (_program)
        glDeleteProgram(_program);
    if (_fontTexture)
        glDeleteTextures(1, &_fontTexture);
    if (_vbo)
        glDeleteBuffers(1, &_vbo);
    if (_staticVbo)
        glDeleteBuffers(1, &_staticVbo);
    if (_vao)
        glDeleteVertexArrays(1, &_vao);
    if (_staticVao)
        glDeleteVertexArrays(1, &_staticVao);
    if (_bgVao)
        glDeleteVertexArrays(1, &_bgVao);
    if (_bgProgram)
        glDeleteProgram(_bgProgram);
}

void
GLhud::Init(int width, int height, int frameBufferWidth, int frameBufferHeight) {
    Hud::Init(width, height, frameBufferWidth, frameBufferHeight);

    glGenTextures(1, &_fontTexture);
    glBindTexture(GL_TEXTURE_2D, _fontTexture);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 FONT_TEXTURE_WIDTH, FONT_TEXTURE_HEIGHT,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, font_image);

    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_staticVbo);

    glGenVertexArrays(1, &_vao);
    glGenVertexArrays(1, &_staticVao);
    glGenVertexArrays(1, &_bgVao);

    GLuint vertexShader = GLUtils::CompileShader(GL_VERTEX_SHADER, s_VS);
    GLuint fragmentShader = GLUtils::CompileShader(GL_FRAGMENT_SHADER, s_FS);

    _program = glCreateProgram();
    glAttachShader(_program, vertexShader);
    glAttachShader(_program, fragmentShader);

    glLinkProgram(_program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint status;
    glGetProgramiv(_program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &infoLogLength);
        char *infoLog = new char[infoLogLength];
        glGetProgramInfoLog(_program, infoLogLength, NULL, infoLog);
        printf("%s\n", infoLog);
        delete[] infoLog;
    }

    _mvpMatrix = glGetUniformLocation(_program, "ModelViewProjectionMatrix");
    _aPosition = glGetAttribLocation(_program, "position");
    _aColor = glGetAttribLocation(_program, "color");
    _aUV = glGetAttribLocation(_program, "uv");

    glBindVertexArray(_vao);
    glEnableVertexAttribArray(_aPosition);
    glEnableVertexAttribArray(_aColor);
    glEnableVertexAttribArray(_aUV);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glVertexAttribPointer(_aPosition, 2, GL_FLOAT, GL_FALSE,
                          sizeof(GLfloat)*7, (void*)0);
    glVertexAttribPointer(_aColor, 3, GL_FLOAT, GL_FALSE,
                          sizeof(GLfloat)*7, (void*)(sizeof(GLfloat)*2));
    glVertexAttribPointer(_aUV, 2, GL_FLOAT, GL_FALSE,
                          sizeof(GLfloat)*7, (void*)(sizeof(GLfloat)*5));

    glBindVertexArray(_staticVao);
    glEnableVertexAttribArray(_aPosition);
    glEnableVertexAttribArray(_aColor);
    glEnableVertexAttribArray(_aUV);
    glBindBuffer(GL_ARRAY_BUFFER, _staticVbo);
    glVertexAttribPointer(_aPosition, 2, GL_FLOAT, GL_FALSE,
                          sizeof(GLfloat)*7, (void*)0);
    glVertexAttribPointer(_aColor, 3, GL_FLOAT, GL_FALSE,
                          sizeof(GLfloat)*7, (void*)(sizeof(GLfloat)*2));
    glVertexAttribPointer(_aUV, 2, GL_FLOAT, GL_FALSE,
                          sizeof(GLfloat)*7, (void*)(sizeof(GLfloat)*5));

    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ------ create bg program
    vertexShader = GLUtils::CompileShader(GL_VERTEX_SHADER, s_BG_VS);
    fragmentShader = GLUtils::CompileShader(GL_FRAGMENT_SHADER, s_BG_FS);

    _bgProgram = glCreateProgram();
    glAttachShader(_bgProgram, vertexShader);
    glAttachShader(_bgProgram, fragmentShader);

    glLinkProgram(_bgProgram);

    GLUtils::CheckGLErrors("GLhud::Init");
}

void
GLhud::Rebuild(int width, int height,
               int framebufferWidth, int framebufferHeight) {
    Hud::Rebuild(width, height, framebufferWidth, framebufferHeight);

    if (! _staticVbo)
        return;

    _staticVboSize = (int)getStaticVboSource().size();
    glBindBuffer(GL_ARRAY_BUFFER, _staticVbo);
    glBufferData(GL_ARRAY_BUFFER, _staticVboSize * sizeof(float),
                 &getStaticVboSource()[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool
GLhud::Flush() {
    if (!Hud::Flush())
        return false;

    // update dynamic text
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, getVboSource().size() * sizeof(float),
                 &getVboSource()[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // (x, y, r, g, b, u, v) = 7
    int numVertices = (int)getVboSource().size()/7;

    // reserved space of the vector remains for the next frame.
    getVboSource().clear();
    glUseProgram(_program);
    float proj[16];
    ortho(proj, 0, 0, float(GetWidth()), float(GetHeight()));
    glUniformMatrix4fv(_mvpMatrix, 1, GL_FALSE, proj);

    glDisable(GL_DEPTH_TEST);
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _fontTexture);

        glBindVertexArray(_vao);
        glDrawArrays(GL_TRIANGLES, 0, numVertices);

        glBindVertexArray(_staticVao);
        glDrawArrays(GL_TRIANGLES, 0, _staticVboSize/7);

        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glEnable(GL_DEPTH_TEST);

    return true;
}

void
GLhud::FillBackground() {
    glUseProgram(_bgProgram);
    glBindVertexArray(_bgVao);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glUseProgram(0);
    glBindVertexArray(0);
}

