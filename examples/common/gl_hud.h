//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#ifndef GL_HUD_H
#define GL_HUD_H

#include "hud.h"

#include <osd/opengl.h>

class GLhud : public Hud
{
public:
    GLhud();
    ~GLhud();

    virtual void Init(int width, int height);

    virtual void Rebuild(int width, int height);

    virtual bool Flush();
    
private:
    GLuint _fontTexture;
    GLuint _vbo, _staticVbo;
    GLuint _vao, _staticVao;
    int _staticVboSize;

    GLint _program;
    GLint _mvpMatrix;
    GLint _aPosition, _aColor, _aUV;
};

#endif // GL_HUD_H
