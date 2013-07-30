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

#include "../osd/glPtexTexture.h"
#include "../osd/ptexTextureLoader.h"

#include "../osd/opengl.h"

#include <Ptexture.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLPtexTexture::OsdGLPtexTexture()
    : _width(0), _height(0), _depth(0), _pages(0), _layout(0), _texels(0) {
}

OsdGLPtexTexture::~OsdGLPtexTexture() {

    // delete pages lookup ---------------------------------
    if (glIsTexture(_pages))
       glDeleteTextures(1, &_pages);

    // delete layout lookup --------------------------------
    if (glIsTexture(_layout))
       glDeleteTextures(1, &_layout);

    // delete textures lookup ------------------------------
    if (glIsTexture(_texels))
       glDeleteTextures(1, &_texels);
}

static GLuint
genTextureBuffer(GLenum format, GLsizeiptr size, GLvoid const * data) {

    GLuint buffer, result;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_TEXTURE_BUFFER, buffer);
    glBufferData(GL_TEXTURE_BUFFER, size, data, GL_STATIC_DRAW);

    glGenTextures(1, & result);
    glBindTexture(GL_TEXTURE_BUFFER, result);
    glTexBuffer(GL_TEXTURE_BUFFER, format, buffer);

    // need to reset texture binding before deleting the source buffer.
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glDeleteBuffers(1, &buffer);

    return result;
}

OsdGLPtexTexture *
OsdGLPtexTexture::Create(PtexTexture * reader,
                      unsigned long int targetMemory,
                      int gutterWidth,
                      int pageMargin) {

    OsdGLPtexTexture * result = NULL;

    // Read the ptexture data and pack the texels
    OsdPtexTextureLoader ldr(reader, gutterWidth, pageMargin);

    unsigned long int nativeSize = ldr.GetNativeUncompressedSize(),
           targetSize = targetMemory;

    if (targetSize != 0 && targetSize != nativeSize)
        ldr.OptimizeResolution(targetSize);

    GLint maxnumpages = 0;
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxnumpages);

    ldr.OptimizePacking(maxnumpages);

    if (!ldr.GenerateBuffers())
        return result;

    // Setup GPU memory
    unsigned long int nfaces = ldr.GetNumBlocks();

    GLuint pages = genTextureBuffer(GL_R32I,
                                    nfaces * sizeof(GLint),
                                    ldr.GetIndexBuffer());

    GLuint layout = genTextureBuffer(GL_RGBA32F,
                                     nfaces * 4 * sizeof(GLfloat),
                                     ldr.GetLayoutBuffer());

    GLenum format, type;
    switch (reader->dataType()) {
        case Ptex::dt_uint16 : type = GL_UNSIGNED_SHORT; break;
        case Ptex::dt_float  : type = GL_FLOAT; break;
        case Ptex::dt_half   : type = GL_HALF_FLOAT; break;
        default              : type = GL_UNSIGNED_BYTE; break;
    }

    switch (reader->numChannels()) {
        case 1 : format = GL_RED; break;
        case 2 : format = GL_RG; break;
        case 3 : format = GL_RGB; break;
        case 4 : format = GL_RGBA; break;
        default: format = GL_RED; break;
    }

    // actual texels texture array
    GLuint texels;
    glGenTextures(1, &texels);
    glBindTexture(GL_TEXTURE_2D_ARRAY, texels);

    // XXXX for the time being, filtering is off - once cross-patch filtering
    // is in place, we will use glGenSamplers to dynamically access these settings.
    if (gutterWidth > 0) {
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0,
                 (type == GL_FLOAT) ? GL_RGBA32F : GL_RGBA,
                 ldr.GetPageSize(),
                 ldr.GetPageSize(),
                 ldr.GetNumPages(),
                 0, format, type,
                 ldr.GetTexelBuffer());

    ldr.ClearBuffers();

    // Return the Osd Ptexture object
    result = new OsdGLPtexTexture;

    result->_width = ldr.GetPageSize();
    result->_height = ldr.GetPageSize();
    result->_depth = ldr.GetNumPages();

    result->_format = format;

    result->_pages = pages;
    result->_layout = layout;
    result->_texels = texels;

    return result;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
