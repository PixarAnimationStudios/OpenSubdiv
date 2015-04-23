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

#ifndef OSD_GL_PTEX_MIPMAP_TEXTURE_H
#define OSD_GL_PTEX_MIPMAP_TEXTURE_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/opengl.h"

#include <stdlib.h>

class PtexTexture;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class GLPtexMipmapTexture : NonCopyable<GLPtexMipmapTexture> {
public:
    static GLPtexMipmapTexture * Create(PtexTexture * reader,
                                           int maxLevels=-1,
                                           size_t targetMemory=0);

    /// Returns the texture buffer containing the layout of the ptex faces
    /// in the texels texture array.
    GLuint GetLayoutTextureBuffer() const { return _layout; }

    /// Returns the texels texture array.
    GLuint GetTexelsTexture() const { return _texels; }

    /// Returns the amount of allocated memory (in byte)
    size_t GetMemoryUsage() const { return _memoryUsage; }

    ~GLPtexMipmapTexture();

private:
    GLPtexMipmapTexture();

    GLsizei _width,   // widht / height / depth of the 3D texel buffer
            _height,
            _depth;

    GLint   _format;  // texel color format

    GLuint _layout,   // per-face lookup table
           _texels;   // texel data

    size_t _memoryUsage;  // total amount of memory used (estimate)
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GL_PTEX_MIPMAP_TEXTURE_H
