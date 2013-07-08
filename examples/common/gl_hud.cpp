//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#include "gl_hud.h"

#include "font_image.h"
#include "simple_math.h"

#include <string.h>
#include <stdio.h>

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

GLhud::GLhud() : _fontTexture(0), _vbo(0), _staticVbo(0),
                 _vao(0), _staticVao(0), _program(0),
                 _aPosition(0), _aColor(0), _aUV(0)
{
}

GLhud::~GLhud()
{
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
}

static GLuint compileShader(GLenum shaderType, const char *source)
{
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    return shader;
}

void
GLhud::Init(int width, int height)
{
    Hud::Init(width, height);
    
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

    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, s_VS);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, s_FS);

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
}

void
GLhud::Rebuild(int width, int height)
{
    Hud::Rebuild(width, height);

    if (not _staticVbo)
        return;

    std::vector<float> vboSource;
    // add UI elements
    for (std::vector<RadioButton>::const_iterator it = getRadioButtons().begin();
         it != getRadioButtons().end(); ++it) {

        int x = it->x > 0 ? it->x : GetWidth() + it->x;
        int y = it->y > 0 ? it->y : GetHeight() + it->y;

        if (it->checked) {
            x = drawChar(vboSource, x, y, 1, 1, 1, FONT_RADIO_BUTTON_ON);
            drawString(vboSource, x, y, 1, 1, 0, it->label.c_str());
        } else {
            x = drawChar(vboSource, x, y, 1, 1, 1, ' ');
            drawString(vboSource, x, y, .5f, .5f, .5f, it->label.c_str());
        }
    }
    for (std::vector<CheckBox>::const_iterator it = getCheckBoxes().begin();
         it != getCheckBoxes().end(); ++it) {

        int x = it->x > 0 ? it->x : GetWidth() + it->x;
        int y = it->y > 0 ? it->y : GetHeight() + it->y;

        if( it->checked) {
            x = drawChar(vboSource, x, y, 1, 1, 1, FONT_CHECK_BOX_ON);
            drawString(vboSource, x, y, 1, 1, 0, it->label.c_str());
        } else {
            x = drawChar(vboSource, x, y, 1, 1, 1, FONT_CHECK_BOX_OFF);
            drawString(vboSource, x, y, .5f, .5f, .5f, it->label.c_str());
        }
    }

    drawString(vboSource, GetWidth()-80, GetHeight()-48, .5, .5, .5, "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f");
    drawString(vboSource, GetWidth()-80, GetHeight()-32, .5, .5, .5, "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f");

    _staticVboSize = (int)vboSource.size();
    glBindBuffer(GL_ARRAY_BUFFER, _staticVbo);
    glBufferData(GL_ARRAY_BUFFER, _staticVboSize * sizeof(float),
                 &vboSource[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool
GLhud::Flush()
{
    if (!Hud::Flush()) 
        return false;

    // update dynamic text
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, getVboSource().size() * sizeof(float),
                 &getVboSource()[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    /* (x, y, r, g, b, u, v) = 7*/
    int numVertices = (int)getVboSource().size()/7;

    // reserved space of the vector remains for the next frame.
    getVboSource().clear();

    glUseProgram(_program);
    float proj[16];
    ortho(proj, 0, 0, float(GetWidth()), float(GetHeight()));
    glUniformMatrix4fv(_mvpMatrix, 1, GL_FALSE, proj);

    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _fontTexture);

        glBindVertexArray(_vao);
        glDrawArrays(GL_TRIANGLES, 0, numVertices);

        glBindVertexArray(_staticVao);
        glDrawArrays(GL_TRIANGLES, 0, _staticVboSize/7);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    return true;
}
