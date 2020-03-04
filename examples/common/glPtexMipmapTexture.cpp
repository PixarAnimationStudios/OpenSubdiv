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

#include "glPtexMipmapTexture.h"
#include "ptexMipmapTextureLoader.h"

GLPtexMipmapTexture::GLPtexMipmapTexture()
    : _width(0), _height(0), _depth(0), _layout(0), _texels(0), _memoryUsage(0)
{
}

GLPtexMipmapTexture::~GLPtexMipmapTexture()
{
    if (glIsTexture(_layout))
       glDeleteTextures(1, &_layout);

    if (glIsTexture(_texels))
       glDeleteTextures(1, &_texels);
}

/*static*/
const char *
GLPtexMipmapTexture::GetShaderSource()
{
    static const char *ptexShaderSource =
#include "glslPtexCommon.gen.h"
        ;
    return ptexShaderSource;
}

static GLuint
genTextureBuffer(GLenum format, GLsizeiptr size, GLvoid const * data)
{
    GLuint buffer = 0;
    GLuint result = 0;

#if defined(GL_ARB_direct_state_access)
    if (OSD_OPENGL_HAS(ARB_direct_state_access)) {
        glCreateBuffers(1, &buffer);
        glNamedBufferData(buffer, size, data, GL_STATIC_DRAW);
        glCreateTextures(GL_TEXTURE_BUFFER, 1, &result);
        glTextureBuffer(result, format, buffer);
    } else
#endif
    {
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_TEXTURE_BUFFER, buffer);
        glBufferData(GL_TEXTURE_BUFFER, size, data, GL_STATIC_DRAW);

        glGenTextures(1, & result);
        glBindTexture(GL_TEXTURE_BUFFER, result);
        glTexBuffer(GL_TEXTURE_BUFFER, format, buffer);

        // need to reset texture binding before deleting the source buffer.
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
    }

    glDeleteBuffers(1, &buffer);

    return result;
}

GLPtexMipmapTexture *
GLPtexMipmapTexture::Create(PtexTexture * reader,
                               int maxLevels,
                               size_t targetMemory)
{
    GLPtexMipmapTexture * result = NULL;

    GLint maxNumPages = 0;
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxNumPages);

    // Read the ptexture data and pack the texels
    PtexMipmapTextureLoader loader(reader,
                                   maxNumPages,
                                   maxLevels,
                                   targetMemory);

    // Setup GPU memory
    int numFaces = loader.GetNumFaces();

    GLuint layout = genTextureBuffer(GL_R16I,
                                     numFaces * 6 * sizeof(GLshort),
                                     loader.GetLayoutBuffer());

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

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0,
                 (type == GL_FLOAT) ? GL_RGBA32F : GL_RGBA,
                 loader.GetPageWidth(),
                 loader.GetPageHeight(),
                 loader.GetNumPages(),
                 0, format, type,
                 loader.GetTexelBuffer());

//    loader.ClearBuffers();

    // Return the Osd Ptexture object
    result = new GLPtexMipmapTexture;

    result->_width = loader.GetPageWidth();
    result->_height = loader.GetPageHeight();
    result->_depth = loader.GetNumPages();

    result->_format = format;

    result->_layout = layout;
    result->_texels = texels;
    result->_memoryUsage = loader.GetMemoryUsage();

    return result;
}
