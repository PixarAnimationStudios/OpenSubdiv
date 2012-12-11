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
#include <GL/glew.h>
#include <string.h>
#include <stdio.h>
#include "gl_hud.h"
#include "font_image.h"

GLhud::GLhud() : _fontTexture(0), _vbo(0), _staticVbo(0)
{
}

GLhud::~GLhud()
{
    if (_fontTexture)
        glDeleteTextures(1, &_fontTexture);
    if (_vbo)
        glDeleteBuffers(1, &_vbo);
    if (_staticVbo)
        glDeleteBuffers(1, &_staticVbo);
}

void
GLhud::Init(int width, int height)
{
    Hud::Init(width, height);
    
    glGenTextures(1, &_fontTexture);
    glBindTexture(GL_TEXTURE_2D, _fontTexture);
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 FONT_TEXTURE_WIDTH, FONT_TEXTURE_HEIGHT,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, font_image);

    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_staticVbo);
}

void
GLhud::Rebuild(int width, int height)
{
    Hud::Rebuild(width, height);

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
    glBufferData(GL_ARRAY_BUFFER, _staticVboSize * sizeof(float), &vboSource[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool
GLhud::Flush()
{
    if (!Hud::Flush()) 
        return false;

    // update dynamic text
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, getVboSource().size() * sizeof(float), &getVboSource()[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    int numVertices = (int)getVboSource().size()/7;  /* (x, y, r, g, b, u, v) = 7*/

    // reserved space of the vector remains for the next frame.
    getVboSource().clear();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, GetWidth(), GetHeight(), 0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glPushAttrib(GL_ENABLE_BIT|GL_POLYGON_BIT);
    {
        glEnable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, 0);
        glDisable(GL_CULL_FACE);
        glColor4f(1, 1, 1, 1);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _fontTexture);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glVertexPointer(2, GL_FLOAT, 7*sizeof(float), (void*)0);
        glColorPointer(3, GL_FLOAT, 7*sizeof(float), (void*)(2*sizeof(float)));
        glTexCoordPointer(2, GL_FLOAT, 7*sizeof(float), (void*)(5*sizeof(float)));

        glDrawArrays(GL_TRIANGLES, 0, numVertices);

        glBindBuffer(GL_ARRAY_BUFFER, _staticVbo);
        glVertexPointer(2, GL_FLOAT, 7*sizeof(float), (void*)0);
        glColorPointer(3, GL_FLOAT, 7*sizeof(float), (void*)(2*sizeof(float)));
        glTexCoordPointer(2, GL_FLOAT, 7*sizeof(float), (void*)(5*sizeof(float)));

        glDrawArrays(GL_TRIANGLES, 0, _staticVboSize/7);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glPopAttrib();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    return true;
}
