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

#include "glFramebuffer.h"
#include "glUtils.h"
#include "glHud.h"

#include <cstdlib>
#include <cassert>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb_image_write.h"

#if _MSC_VER
#define snprintf _snprintf
#endif

GLFrameBuffer::GLFrameBuffer() :
_width(0), _height(0),
_frameBuffer(0),
_frameBufferColor(0),
_frameBufferNormal(0),
_frameBufferDepthTexture(0),
_program(0),
_vao(0),
_vbo(0) {
}

GLFrameBuffer::~GLFrameBuffer() {

    glDeleteFramebuffers(1, &_frameBuffer);
    glDeleteTextures(1, &_frameBufferColor);
    glDeleteTextures(1, &_frameBufferNormal);
    glDeleteTextures(1, &_frameBufferDepthTexture);

    glDeleteProgram(_program);

    glDeleteVertexArrays(1, &_vao);
    glDeleteBuffers(1, &_vbo);
}

static const char *g_framebufferShaderSource =
#include "framebuffer.gen.h"
;

void
GLFrameBuffer::Init(int width, int height) {

    _width = width;
    _height = height;

    if (not _program)
        _program = compileProgram(g_framebufferShaderSource);

    if (not _vao) {
        glGenVertexArrays(1, &_vao);
    }
    glBindVertexArray(_vao);
    
    if (not _vbo) {
        glGenBuffers(1, &_vbo);
        static float pos[] = { -1, -1, 1, -1, -1,  1, 1,  1 };
        glGenBuffers(1, &_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(pos), pos, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (not _frameBuffer) {
        glGenFramebuffers(1, &_frameBuffer);

        if (not _frameBufferColor) {
            _frameBufferColor=allocateTexture();
        }
        
        if (not _frameBufferNormal) {
            _frameBufferNormal=allocateTexture();
        }

        if (not _frameBufferDepthTexture) {
            _frameBufferDepthTexture=allocateTexture();
        }
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    GLUtils::CheckGLErrors("FrameBuffer::Init");
}

void
GLFrameBuffer::Bind() const {

    glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);
    GLenum buffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, buffers);
}

GLuint
GLFrameBuffer::allocateTexture() {
    GLuint texture;
    
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLUtils::CheckGLErrors("FrameBuffer::allocateTexture");
    
    return texture;
}

GLuint
GLFrameBuffer::compileProgram(char const * src, char const * defines) {

    GLuint program = glCreateProgram();

    const std::string &versionStr = GLUtils::GetShaderVersionInclude();
    static char const vtxDefineStr[] = "#define IMAGE_VERTEX_SHADER\n",
                      fragDefineStr[] = "#define IMAGE_FRAGMENT_SHADER\n";

    std::string vertexSrc = std::string(versionStr) + vtxDefineStr + (defines ? defines : "") + src;
    GLuint vs = GLUtils::CompileShader(GL_VERTEX_SHADER, vertexSrc.c_str());

    std::string fragmentSrc = std::string(versionStr) + fragDefineStr + (defines ? defines : "") + src;
    GLuint fs = GLUtils::CompileShader(GL_FRAGMENT_SHADER, fragmentSrc.c_str());

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &infoLogLength);
        char *infoLog = new char[infoLogLength];
        glGetProgramInfoLog(_program, infoLogLength, NULL, infoLog);
        printf("%s\n", infoLog);
        delete[] infoLog;
    }


    if (GLUtils::GL_ARBSeparateShaderObjectsOrGL_VERSION_4_1()){
        GLint colorMap = glGetUniformLocation(program, "colorMap");
        if (colorMap != -1)
            glProgramUniform1i(program, colorMap, 0);  // GL_TEXTURE0

        GLint normalMap = glGetUniformLocation(program, "normalMap");
        if (normalMap != -1)
            glProgramUniform1i(program, normalMap, 1);  // GL_TEXTURE1

        GLint depthMap = glGetUniformLocation(program, "depthMap");
        if (depthMap != -1)
            glProgramUniform1i(program, depthMap, 2);  // GL_TEXTURE2
    }
    else{
        glUseProgram(program);
        GLint colorMap = glGetUniformLocation(program, "colorMap");
        if (colorMap != -1)
            glUniform1i(colorMap, 0);  // GL_TEXTURE0

        GLint normalMap = glGetUniformLocation(program, "normalMap");
        if (normalMap != -1)
            glUniform1i(normalMap, 1);  // GL_TEXTURE1

        GLint depthMap = glGetUniformLocation(program, "depthMap");
        if (depthMap != -1)
            glUniform1i(depthMap, 2);  // GL_TEXTURE2
    }

    return program;
}

void
GLFrameBuffer::ApplyImageShader() {

    glDisable(GL_DEPTH_TEST);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, GetWidth(), GetHeight());

    glUseProgram(_program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _frameBufferColor);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _frameBufferNormal);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _frameBufferDepthTexture);

    glBindVertexArray(_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindVertexArray(0);
    glUseProgram(0);
}


void
GLFrameBuffer::Reshape(int width, int height) {

    // resize framebuffers
    glBindTexture(GL_TEXTURE_2D, _frameBufferColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glBindTexture(GL_TEXTURE_2D, _frameBufferNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_FLOAT, 0);

    glBindTexture(GL_TEXTURE_2D, _frameBufferDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, _frameBufferColor, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                           GL_TEXTURE_2D, _frameBufferNormal, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, _frameBufferDepthTexture, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        assert(false);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    _width=width;
    _height=height;
}


void
GLFrameBuffer::BuildUI(GLhud * /* hud */, int /* x */, int /* y */) {
}

struct PixelStoreState {

    void Push() {
        glGetIntegerv(GL_PACK_ROW_LENGTH,  &_packRowLength);
        glGetIntegerv(GL_PACK_ALIGNMENT,   &_packAlignment);
        glGetIntegerv(GL_PACK_SKIP_PIXELS, &_packSkipPixels);
        glGetIntegerv(GL_PACK_SKIP_ROWS,   &_packSkipRows);
    }

    void Pop() {
        Set(_packRowLength, _packAlignment, _packSkipPixels, _packSkipRows);
    }

    void Set(GLint packRowLength,
             GLint packAlignment,
             GLint packSkipPixels,
             GLint packSkipRows) {
        glPixelStorei(GL_PACK_ROW_LENGTH,  packRowLength);
        glPixelStorei(GL_PACK_ALIGNMENT,   packAlignment);
        glPixelStorei(GL_PACK_SKIP_PIXELS, packSkipPixels);
        glPixelStorei(GL_PACK_SKIP_ROWS,   packSkipRows);
    }
private:
    GLint _packRowLength,
          _packAlignment,
          _packSkipPixels,
          _packSkipRows;
};

void
GLFrameBuffer::Screenshot() const {

    int width = GetWidth(),
        height = GetHeight();

    std::vector<unsigned char> data(width*height*4 /*RGBA*/);

    PixelStoreState pixelStore;

    pixelStore.Push();

    pixelStore.Set( /* GL_PACK_ROW_LENGTH  */ 0, 
                    /* GL_PACK_ALIGNMENT   */ 1,
                    /* GL_PACK_SKIP_PIXELS */ 0,
                    /* GL_PACK_SKIP_ROWS   */ 0);

    GLint restoreBinding, restoreActiveTexture;
    glGetIntegerv( GL_TEXTURE_BINDING_2D, &restoreBinding );
    glGetIntegerv( GL_ACTIVE_TEXTURE, & restoreActiveTexture);

    glActiveTexture( GL_TEXTURE0 );

    glBindTexture( GL_TEXTURE_2D, _frameBufferColor );
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, &data[0]);

    glActiveTexture( restoreActiveTexture );
    glBindTexture( GL_TEXTURE_2D, restoreBinding );

    pixelStore.Pop();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    static int counter=0;
    char fname[64];
    snprintf(fname, 64, "screenshot.%d.png", counter++);

    // flip vertical
    stbi_write_png(fname, width, height, 4, &data[width*4*(height-1)], -width*4);

    fprintf(stdout, "Saved %s\n", fname);
}


//------------------------------------------------------------------------------



SSAOGLFrameBuffer::SSAOGLFrameBuffer() :
              _active(0),
              _radius(0),
              _scale(0),
              _gamma(0),
              _contrast(0) {
}

static const char *g_ssaoShaderSource =
#include "ssao.gen.h"
;

void
SSAOGLFrameBuffer::Init(int width, int height) {

    GLFrameBuffer::Init(width, height);

    if (_active) {
        setProgram( compileProgram(g_ssaoShaderSource) );

        GLuint program = getProgram();

        glUseProgram(program);
        _radius = glGetUniformLocation(program, "radius");
        _scale = glGetUniformLocation(program, "scale");
        _gamma = glGetUniformLocation(program, "gamma");
        _contrast = glGetUniformLocation(program, "contrast");

        SetRadius(0.01f);
        SetScale(300.0f);
        SetContrast(1.0f);
        SetGamma(1.0f);
    }

    GLUtils::CheckGLErrors("SSAOGLFrameBuffer::Init");
}

void
SSAOGLFrameBuffer::ApplyImageShader() {

    GLFrameBuffer::ApplyImageShader();
}


// nasty hack until callbacks carry pointers
static SSAOGLFrameBuffer * g_ssaofb=0;

static void
ssaoCallbackCheckbox(bool value, int /* data */) {

    g_ssaofb->SetActive(value);
}

static void
ssaoCallbackSlider(float value, int data) {

    switch (data) {
        case 0: g_ssaofb->SetRadius(value); break;
        case 1: g_ssaofb->SetScale(value); break;
        case 2: g_ssaofb->SetGamma(value); break;
        case 3: g_ssaofb->SetContrast(value); break;
        default:
            assert(0);
    }
}

void
SSAOGLFrameBuffer::BuildUI(GLhud * hud, int x, int y) {

    g_ssaofb = this;

    hud->AddCheckBox("SSAO", _active, x, y, ssaoCallbackCheckbox);

    hud->AddSlider("Radius",   0.0f,    0.1f,  0.01f, x+20, y+20,  20, false, ssaoCallbackSlider, 0);
    hud->AddSlider("Scale",    1.0f, 1000.0f, 300.0f, x+20, y+60,  20, false, ssaoCallbackSlider, 1);
    //hud->AddSlider("Gamma",    0.0f,    1.0f,   1.0f, x+20, y+100, 20, false, callbackSlider, 2);
    //hud->AddSlider("Contrast", 0.0f,    1.0f,   1.0f, x+20, y+140, 20, false, callbackSlider, 3);
}

void
SSAOGLFrameBuffer::SetActive(bool value) {

    if ( (_active = value) ) {
        SSAOGLFrameBuffer::Init(GetWidth(), GetHeight());
    } else {
        setProgram( compileProgram(g_framebufferShaderSource) );
    }
}

void
SSAOGLFrameBuffer::SetRadius(float value) {

    GLuint program = getProgram();
    if (_radius!=-1 and program>0) {
#if defined(GL_ARB_separate_shader_objects) or defined(GL_VERSION_4_1)
        glProgramUniform1f(program, _radius, value);
#else
        glUseProgram(program);
        glUniform1f(_radius, value);
#endif
    }
}

void
SSAOGLFrameBuffer::SetScale(float value) {

    GLuint program = getProgram();
    if (_scale!=-1 and program>0) {
#if defined(GL_ARB_separate_shader_objects) or defined(GL_VERSION_4_1)
        glProgramUniform1f(program, _scale, value);
#else
        glUseProgram(program);
        glUniform1f(_scale, value);
#endif
    }
}

void
SSAOGLFrameBuffer::SetGamma(float value) {

    GLuint program = getProgram();
    if (_gamma!=-1 and program>0) {
#if defined(GL_ARB_separate_shader_objects) or defined(GL_VERSION_4_1)
        glProgramUniform1f(program, _gamma, value);
#else
        glUseProgram(program);
        glUniform1f(_gamma, value);
#endif
    }
}

void
SSAOGLFrameBuffer::SetContrast(float value) {

    GLuint program = getProgram();
    if (_contrast!=-1 and program>0) {
#if defined(GL_ARB_separate_shader_objects) or defined(GL_VERSION_4_1)
        glProgramUniform1f(program, _contrast, value);
#else
        glUseProgram(program);
        glUniform1f(_contrast, value);
#endif
    }
}
