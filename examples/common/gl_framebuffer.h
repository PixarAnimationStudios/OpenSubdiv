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

#ifndef GL_FRAMEBUFFER_H
#define GL_FRAMEBUFFER_H

#include <osd/opengl.h>

class GLhud;

//------------------------------------------------------
class GLFrameBuffer {

public:
    GLFrameBuffer();

    int GetWidth() const {
        return _width;
    }

    int GetHeight() const {
        return _height;
    }

    virtual void Init(int width, int height);

    virtual void Reshape(int width, int height);

    virtual void ApplyImageShader();

    virtual void Bind() const;
    
    virtual void BuildUI(GLhud * hud, int x, int y);

    void Screenshot() const;

protected:
    friend class GLhud;

    virtual ~GLFrameBuffer();

    GLuint getFrameBuffer() {
        return _frameBuffer;
    }

    GLuint getProgram() {
        return _program;
    }
    
    void setProgram(GLuint program) {

        if (_program) {
            glDeleteProgram(_program);
        }
        _program = program;
    }
    
    GLuint compileProgram(char const * src, char const * defines=0);

    GLuint allocateTexture();

private:

    int _width, _height;
    GLuint _frameBuffer;
    GLuint _frameBufferColor;
    GLuint _frameBufferNormal;
    GLuint _frameBufferDepthTexture;

    GLuint _program;

    GLuint _vao;
    GLuint _vbo;
};

//------------------------------------------------------
class SSAOGLFrameBuffer : public GLFrameBuffer {

public:

    SSAOGLFrameBuffer();

    virtual void Init(int width, int height);

    virtual void ApplyImageShader();

    virtual void BuildUI(GLhud * hud, int x, int y);

    void SetActive(bool value);

    void SetRadius(float value);

    void SetScale(float value);

    void SetGamma(float value);

    void SetContrast(float value);

private:

    bool _active;

    GLint _radius,
          _scale,
          _gamma,
          _contrast;
};

#endif // GL_FRAMEBUFFER_H
