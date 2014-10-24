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

#ifndef GL_HUD_H
#define GL_HUD_H

#include "hud.h"

#include <osd/opengl.h>

#include "gl_framebuffer.h"

class GLhud : public Hud {

public:
    GLhud();
    ~GLhud();

    virtual void Init(int width, int height, int framebufferWidth, int framebufferHeight);

    virtual void Rebuild(int width, int height,
                         int framebufferWidth, int framebufferHeight);

    virtual bool Flush();

    void SetFrameBuffer(GLFrameBuffer * frameBuffer) {
        if (not _frameBuffer) {
            _frameBuffer = frameBuffer;
            _frameBuffer->Init(GetWidth(), GetHeight());
            _frameBuffer->BuildUI(this, 10, 600);
        }
    }

    GLFrameBuffer * GetFrameBuffer() {
        return _frameBuffer;
    }

    GLuint GetFontTexture() const {
        return _fontTexture;
    }

private:


    GLFrameBuffer * _frameBuffer;

    GLuint _fontTexture;
    GLuint _vbo, _staticVbo;
    GLuint _vao, _staticVao;
    int _staticVboSize;

    GLint _program;
    GLint _mvpMatrix;
    GLint _aPosition, _aColor, _aUV;
};

#endif // GL_HUD_H
